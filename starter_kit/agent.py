"""
agent.py - Mode Dificil (bateria 6.000 kWh / 1.500 kW, solar x12, mercat a 3 h)

Resum del funcionament:
  1. Memoria: guarda cada hora el que arriba a obs (preu, CO2, consum, solar, meteo).
  2. Prediccio de les properes 24 h:
       - perfils base per mes / tipus de dia / hora (amb festius i dies de campus tancat),
         la solar en hora solar (corregeix el canvi d'hora) i interpolada entre mesos;
       - correccions lineals per horitzo i hora amb les 3 h de mercat conegudes, la darrera
         hora mesurada, el que va passar fa 24 h i la meteo;
       - desviacio tipica de l'error de consum i solar per a cada hora.
  3. Planificacio per programacio dinamica sobre l'estat de carrega: minimitza
     0,7*CO2 + 0,3*cost de la importacio ESPERADA de la xarxa (te en compte la incertesa:
     descarregar mes del consum o carregar mes que l'excedent no serveix de res).
  4. Aplica nomes la decisio de l'hora actual, amb limits fisics exactes (0 infraccions).
"""
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ETA = math.sqrt(0.90)
H = 24
HORA = timedelta(hours=1)
THR = 50.0          # kW d'envolupant solar a partir dels quals "hi ha sol"
AVARIA = 150.0      # kW: per sota, la lectura del comptador es considera una avaria (consum real ~0 a efectes del simulador)
RHO_AVARIA = 0.9    # persistencia horaria de l'avaria

# Festius nacionals de data fixa: el PVPC (peatge 2.0TD) els tracta com a cap de setmana.
NACIONALS = {(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)}
# Fraccio de "campus tancat" en dies especials (1 = com un festiu, 0 = feiner normal).
ESPECIALS = {
    (2025, 9, 11): 1.0, (2025, 9, 24): 1.0, (2025, 12, 8): 1.0,
    (2025, 12, 24): 0.5, (2025, 12, 25): 1.0, (2025, 12, 26): 1.0,
    (2025, 12, 29): 0.8, (2025, 12, 30): 0.8, (2025, 12, 31): 0.8,
    (2026, 1, 1): 1.0, (2026, 1, 2): 0.8, (2026, 1, 5): 0.8, (2026, 1, 6): 1.0,
    (2026, 3, 30): 0.5, (2026, 3, 31): 0.5, (2026, 4, 1): 0.5, (2026, 4, 2): 0.5,
    (2026, 4, 3): 1.0, (2026, 4, 6): 1.0, (2026, 5, 1): 1.0, (2026, 5, 25): 1.0,
    (2026, 6, 24): 1.0, (2026, 9, 11): 1.0, (2026, 9, 24): 1.0,
}


def _dim(y, m):
    return ((datetime(y + (m == 12), m % 12 + 1, 1)) - timedelta(days=1)).day


def _ultim_diumenge(y, m):
    d = datetime(y, m, _dim(y, m))
    return d - timedelta(days=(d.weekday() + 1) % 7)


def frac_tancat(d):
    if d.weekday() >= 5 or (d.month, d.day) in NACIONALS:
        return 1.0
    return ESPECIALS.get((d.year, d.month, d.day), 0.0)


def hkey(ts):
    return ts.toordinal() * 24 + ts.hour


class BatteryAgent:
    NS, NA = 151, 101

    def __init__(self, params_path=None, params=None):
        if params is None:
            p = Path(params_path) if params_path else Path(__file__).parent / "parametres.json"
            with open(p, encoding="utf-8") as f:
                params = json.load(f)
        P = self.P = params
        self.TP, self.TC, self.TL = P["TP"], P["TC"], P["TL"]
        self.RP, self.RC, self.RL = P["RP"], P["RC"], P["RL"]
        self.TV, self.TVX, self.TI, self.TT = P["TV"], P["TVX"], P["TI"], P["TT"]
        self.coef = P.get("coef", {})
        self.sig = P.get("sig", {})
        self.centre = P.get("centre", 12.0)
        self.g_p, self.g_c, self.g_v = P.get("g_p", 0.0), P.get("g_c", 1.0), P.get("g_v", 1.0)
        self.CB, self.PB = P.get("CB", 0.0933), P.get("PB", 0.1528)
        self.KM = P.get("KM")                     # normalitzadors per mes (opcional)
        self.sigma_k = P.get("sigma_k", 1.0)
        self.lam = P.get("lam", 0.8)
        self.kappa = P.get("kappa", 0.0)          # valor d'opcio: optimisme sobre el mercat futur
        self._sw = None
        self._pn = None
        self._pv = None
        self.bloc = P.get("bloc")                 # None = ultims 7 dies del mes; o [dia_ini, dia_fi]
        self.salt = bool(P.get("salt", True))     # el test salta d'una setmana a la del mes seguent
        ft = P.get("final_ts", "2026-08-31 23:00:00")
        self.final_ts = datetime.fromisoformat(ft) if ft else None
        self.mode_reset = bool(P.get("mode_reset", False))
        self._cc = {}
        self.n_err = 0
        self._grid_for = None
        self._reset_mem()

    def _reset_mem(self):
        self.rp, self.rc = {}, {}                 # preu / CO2 reals
        self.ap, self.ac = {}, {}                 # anomalies respecte al perfil
        self.rl, self.al = {}, {}                 # consum real i anomalia
        self.av, self.qv, self.qi = {}, {}, {}    # anomalia solar, claredat solar i d'irradiancia
        self.at = {}                              # anomalia de temperatura
        self.sun = None                           # (hk, qv, qi) de la darrera hora amb sol observada
        self.avaria = None                        # darrera hora amb el comptador a zero
        self.last_hk = None
        self.soc_esp = None
        self.nstep = 0

    # ------------------------------------------------------------------ calendari
    def _ctx(self, ts):
        key = ts.toordinal()
        c = self._cc.get(key)
        if c is not None:
            return c
        y, m, d = ts.year, ts.month, ts.day
        dia = datetime(y, m, d)
        f = frac_tancat(dia)
        fa = frac_tancat(dia - timedelta(days=1))
        nac = (m, d) in NACIONALS
        tp = 1 if (dia.weekday() >= 5 or nac) else 0
        dow = 6 if nac else dia.weekday()
        ml = 9 if (m == 8 and d >= 24) else m
        combo = 2 * (1 if fa >= 0.5 else 0) + (1 if f >= 0.5 else 0)
        dst = 1 if _ultim_diumenge(y, 3) <= dia < _ultim_diumenge(y, 10) else 0
        a = (d - self.centre) / 30.0
        mb = (m % 12 + 1) if a > 0 else ((m - 2) % 12 + 1)
        a = min(1.0, abs(a))
        c = (m - 1, tp, dow, ml - 1, f, combo, dst, mb - 1, a)
        self._cc[key] = c
        return c

    def _bp(self, c, h):
        # feiners: el preu segueix l'hora oficial (peatges); caps de setmana: l'hora solar
        hs = (h - c[6]) % 24 if c[1] == 1 else h
        v = self.TP[c[0]][c[1]][hs]
        a = self.g_p * c[8]
        if a > 0 and not (c[0] == 7 and c[7] == 8):
            v = (1 - a) * v + a * self.TP[c[7]][c[1]][hs]
        return v + self.RP[c[2]][h]

    def _bc(self, c, h):
        # el CO2 de la xarxa segueix el sol: taula en hora solar (corregeix el canvi d'hora)
        hs = (h - c[6]) % 24
        v = self.TC[c[0]][c[1]][hs]
        a = self.g_c * c[8]
        if a > 0 and not (c[0] == 7 and c[7] == 8):
            v = (1 - a) * v + a * self.TC[c[7]][c[1]][hs]
        return v + self.RC[c[2]][h]

    def _bl(self, c, h):
        f = c[4]
        t = self.TL[c[3]]
        if f <= 0.0:
            v = t[0][h]
        elif f >= 1.0:
            v = t[1][h]
        else:
            v = (1 - f) * t[0][h] + f * t[1][h]
        return v + self.RL[c[5]][h]

    def _sol(self, T, c, h):
        """taula solar/meteo en hora solar, interpolada entre mesos"""
        hs = h - c[6]
        if hs < 0:
            return T[c[0]][0]
        a = self.g_v * c[8]
        return (1 - a) * T[c[0]][hs] + a * T[c[7]][hs]

    # ------------------------------------------------------------------ blocs del test
    def _fi_bloc(self, ts):
        dim = _dim(ts.year, ts.month)
        return dim if self.bloc is None else min(self.bloc[1], dim)

    def _ini_bloc(self, y, m):
        dim = _dim(y, m)
        return dim - 6 if self.bloc is None else min(self.bloc[0], dim)

    def _dins_bloc(self, ts):
        return self._ini_bloc(ts.year, ts.month) <= ts.day <= self._fi_bloc(ts)

    def _seguent(self, ts):
        if (self.salt or self.mode_reset) and ts.hour == 23 and ts.day == self._fi_bloc(ts):
            y, m = (ts.year + 1, 1) if ts.month == 12 else (ts.year, ts.month + 1)
            return datetime(y, m, self._ini_bloc(y, m)), True
        return ts + HORA, False

    def _linia_temps(self, ts, n):
        """[(ts, hk, ctx, mateix_bloc)] de les properes n hores del test (saltant entre setmanes)."""
        out, t, same = [], ts, True
        for _ in range(n):
            out.append((t, hkey(t), self._ctx(t), same))
            if self.final_ts is not None and t == self.final_ts:
                break
            t, salt = self._seguent(t)
            if salt:
                if self.mode_reset:
                    break
                same = False
        return out

    # ------------------------------------------------------------------ memoria
    def _obs_mercat(self, tl, preus, co2s):
        for j in range(min(len(preus), len(tl))):
            t, k, c, _ = tl[j]
            self.rp[k] = preus[j]
            self.ap[k] = preus[j] - self._bp(c, t.hour)
        for j in range(min(len(co2s), len(tl))):
            t, k, c, _ = tl[j]
            self.rc[k] = co2s[j]
            self.ac[k] = co2s[j] - self._bc(c, t.hour)

    def _obs_campus(self, tprev, load=None, pv=None, irr=None, temp=None):
        k = hkey(tprev)
        c = self._ctx(tprev)
        h = tprev.hour
        if load is not None:
            if load < AVARIA:
                self.avaria = k           # comptador a zero: no es una mesura valida del consum
            else:
                self.rl[k] = load
                self.al[k] = load - self._bl(c, h)
        if temp is not None:
            self.at[k] = temp - self._sol(self.TT, c, h)
        if pv is not None:
            self.av[k] = pv - self._sol(self.TV, c, h)
            mx = self._sol(self.TVX, c, h)
            if mx > THR:
                q = (pv - self._sol(self.TV, c, h)) / mx
                qi = 0.0
                if irr is not None:
                    ti = self._sol(self.TI, c, h)
                    qi = (irr - ti) / max(ti, 50.0)
                self.qv[k] = q
                self.sun = (k, q, qi)

    def _neteja(self, hk):
        if self.nstep % 48:
            return
        lim = hk - 80
        for d in (self.rp, self.rc, self.ap, self.ac, self.rl, self.al, self.av, self.qv, self.at):
            for k in [k for k in d if k < lim]:
                del d[k]

    # ------------------------------------------------------------------ variables explicatives
    @staticmethod
    def _mitja24(d, ref):
        v = [d[k] for k in range(ref - 23, ref + 1) if k in d]
        return (sum(v) / len(v)) if len(v) >= 12 else None

    def features(self, tl, m):
        """Retorna perfils base i, per a cada variable, [(k, variant, x)] de les hores a predir."""
        n = len(tl)
        hk = tl[0][1]
        base_p = [self._bp(c, t.hour) for t, _, c, _ in tl]
        base_c = [self._bc(c, t.hour) for t, _, c, _ in tl]
        base_l = [self._bl(c, t.hour) for t, _, c, _ in tl]
        base_v = [self._sol(self.TV, c, t.hour) for t, _, c, _ in tl]
        mx_v = [self._sol(self.TVX, c, t.hour) for t, _, c, _ in tl]
        F = {"p": [], "c": [], "l": [], "v": []}

        # --- mercat
        js = [j for j in range(min(m, n)) if tl[j][3]]
        for nom, A, Ao in (("p", self.ap, self.ac), ("c", self.ac, self.ap)):
            an = [A[tl[j][1]] for j in js if tl[j][1] in A]
            ao = [Ao[tl[j][1]] for j in js if tl[j][1] in Ao]
            if not an:
                continue
            a0, a2 = sum(an) / len(an), an[-1]
            o0 = (sum(ao) / len(ao)) if ao else 0.0
            d24 = self._mitja24(A, tl[js[-1]][1])
            for k in range(m, n):
                if not tl[k][3]:
                    continue
                g24 = A.get(tl[k][1] - 24)
                if g24 is not None and d24 is not None:
                    F[nom].append((k, "lag", (1.0, a0, a2, o0, g24, d24)))
                else:
                    F[nom].append((k, "nolag", (1.0, a0, a2, o0)))

        # --- consum
        lprev = self.rl.get(hk - 1)
        e1 = self.al.get(hk - 1)
        tmp = self.at.get(hk - 1, 0.0)
        dl24 = self._mitja24(self.al, hk - 1)
        for k in range(n):
            if not tl[k][3]:
                continue
            e24 = self.al.get(tl[k][1] - 24)
            if lprev is not None:
                p1 = lprev - base_l[k]
                if e24 is not None and dl24 is not None:
                    F["l"].append((k, "full", (1.0, e1, p1, e24, tmp, dl24)))
                else:
                    F["l"].append((k, "prev", (1.0, e1, p1, tmp)))
            elif e24 is not None:
                F["l"].append((k, "lag", (1.0, e24)))

        # --- solar
        sun = self.sun if (self.sun is not None and hk - self.sun[0] <= 30) else None
        for k in range(n):
            if not tl[k][3] or mx_v[k] <= THR:
                continue
            s24 = self.av.get(tl[k][1] - 24)
            b, mx = base_v[k], mx_v[k]
            if sun is not None and s24 is not None:
                F["v"].append((k, "full", (b, sun[1] * mx, sun[2] * mx, s24)))
            elif sun is not None:
                F["v"].append((k, "q", (b, sun[1] * mx, sun[2] * mx)))
            elif s24 is not None:
                F["v"].append((k, "lag", (b, s24)))
        return base_p, base_c, base_l, base_v, mx_v, F

    def _aplica(self, nom, F, tl, out):
        C = self.coef.get(nom, {})
        usat = {}
        for k, var, x in F[nom]:
            cv = C.get(var)
            if cv is None:
                continue
            b = cv[k][tl[k][0].hour]
            out[k] += sum(bi * xi for bi, xi in zip(b, x))
            usat[k] = var
        return usat

    def forecast(self, tl, m, preus, co2s):
        n = len(tl)
        base_p, base_c, base_l, base_v, mx_v, F = self.features(tl, m)
        pr, co, ld, pv = list(base_p), list(base_c), list(base_l), list(base_v)
        up = self._aplica("p", F, tl, pr)
        uc = self._aplica("c", F, tl, co)
        ul = self._aplica("l", F, tl, ld)
        uv = self._aplica("v", F, tl, pv)
        for k in range(min(m, n)):
            if k < len(preus):
                pr[k] = preus[k]
            if k < len(co2s):
                co[k] = co2s[k]
        sd = [0.0] * n
        self._sw = np.zeros(n)
        SP, SC = self.sig.get("p", {}), self.sig.get("c", {})
        for k in range(min(m, n), n):
            h = tl[k][0].hour
            sp_ = SP.get(up.get(k, "none"), SP.get("none"))
            sc_ = SC.get(uc.get(k, "none"), SC.get("none"))
            if sp_ is not None and sc_ is not None:
                self._sw[k] = math.hypot(0.3 * sp_[k][h] / self.PB, 0.7 * sc_[k][h] / self.CB)
        SL, SV = self.sig.get("l", {}), self.sig.get("v", {})
        self._pn = np.ones(n)
        en_avaria = self.avaria is not None and self.avaria == tl[0][1] - 1
        for k in range(n):
            h = tl[k][0].hour
            pv[k] = min(max(pv[k], 0.0), 1.15 * mx_v[k]) if mx_v[k] > 1.0 else 0.0
            ld[k] = max(ld[k], 0.0)
            if en_avaria and tl[k][3]:
                self._pn[k] = 1.0 - RHO_AVARIA ** (k + 1)   # probabilitat que el comptador torni a mesurar
            sl = SL.get(ul.get(k, "none"), SL.get("none"))
            s_l = sl[k][h] if sl is not None else 100.0
            sv = SV.get(uv.get(k, "none"), SV.get("none"))
            s_v = (sv[k] * mx_v[k]) if (sv is not None and mx_v[k] > THR) else 0.0
            sd[k] = math.hypot(s_l, s_v)
        self._pv = np.array(pv)
        return (np.maximum(np.array(pr), 0.0), np.maximum(np.array(co), 0.0),
                np.array(ld), self._pv, np.array(sd))

    # ------------------------------------------------------------------ planificador (DP)
    def _build(self, cap, pmax):
        self._grid_for = (cap, pmax)
        self.S = np.linspace(0.0, cap, self.NS)
        dS = self.S[1] - self.S[0]
        self.A = np.linspace(-pmax, pmax, self.NA)
        delta = np.where(self.A > 0, self.A * ETA, self.A / ETA)[:, None]
        sn = self.S[None, :] + delta
        valid = (sn >= -1e-9) & (sn <= cap + 1e-9)
        sn = np.clip(sn, 0.0, cap)
        pos = sn / dS
        self.i0 = np.minimum(np.floor(pos).astype(np.int64), self.NS - 2)
        self.i1 = self.i0 + 1
        self.fr = pos - self.i0
        self.pen = np.where(valid, 0.0, 1e9)

    @staticmethod
    def _emax(mu, s):
        """E[max(0, X)] amb X ~ N(mu, s^2)"""
        if s < 1e-6:
            return np.maximum(0.0, mu)
        z = mu / s
        return mu * ndtr(z) + s * 0.3989422804014327 * np.exp(-0.5 * z * z)

    def _import(self, t, a, net, sd):
        """importacio esperada de la xarxa a l'hora t amb consigna a (barreja normal / comptador a zero)"""
        f = self._emax(net[t] + a, self.sigma_k * sd[t])
        pn = self._pn[t]
        if pn < 1.0:
            f = pn * f + (1.0 - pn) * np.maximum(0.0, a - self._pv[t])
        return f

    def _plan(self, soc, w, net, sd, lam, cap, pmax):
        if self._grid_for != (cap, pmax):
            self._build(cap, pmax)
        n = len(w)
        V = -lam * self.S
        for t in range(n - 1, 0, -1):
            f = self._import(t, self.A, net, sd)
            c = w[t] * f
            if self.kappa > 0 and self._sw is not None and self._sw[t] > 0:
                f0 = float(self._import(t, np.array([0.0]), net, sd)[0])
                c = c - self.kappa * self._sw[t] * np.abs(f - f0)
            Vn = V[self.i0] * (1.0 - self.fr) + V[self.i1] * self.fr
            V = (c[:, None] + Vn + self.pen).min(axis=0)
        max_c = max(0.0, min(pmax, (cap - soc) / ETA) - 1e-6)
        max_d = max(0.0, min(pmax, soc * ETA) - 1e-6)
        cand = np.concatenate([self.A, np.linspace(-max_d, max_c, 121), [0.0, -net[0]]])
        cand = np.clip(cand, -max_d, max_c)
        sn = soc + np.where(cand > 0, cand * ETA, cand / ETA)
        q = w[0] * self._import(0, cand, net, sd) + np.interp(sn, self.S, V)
        q = q + 1e-9 * np.abs(cand)          # en cas d'empat, la consigna mes petita
        return float(cand[int(np.argmin(q))])

    # ------------------------------------------------------------------ decisio
    @staticmethod
    def _num(v):
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    def _serie(self, v):
        out = []
        try:
            for x in list(v)[:H]:
                x = self._num(x)
                if x is None:
                    break
                out.append(x)
        except TypeError:
            pass
        return out

    def act(self, obs: dict) -> float:
        cap = self._num(obs.get("cap_max_kwh")) or 6000.0
        pmax = self._num(obs.get("power_max_kw")) or 1500.0
        soc = self._num(obs.get("soc_kwh"))
        if soc is None:
            soc = (self._num(obs.get("soc_pct")) or 0.0) * cap
        soc = min(cap, max(0.0, soc))
        a = 0.0
        try:
            a = self._decideix(obs, soc, cap, pmax)
        except Exception:
            self.n_err += 1
            a = 0.0
        max_c = max(0.0, min(pmax, (cap - soc) / ETA) - 1e-6)
        max_d = max(0.0, min(pmax, soc * ETA) - 1e-6)
        if not math.isfinite(a):
            a = 0.0
        a = max(-max_d, min(max_c, a))
        if abs(a) < 1e-6:
            a = 0.0
        self.soc_esp = soc + a * ETA if a > 0 else soc + a / ETA
        return float(a)

    def _decideix(self, obs, soc, cap, pmax):
        ts = obs["timestamp"]
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts))
        ts = datetime(ts.year, ts.month, ts.day, ts.hour)
        hk = hkey(ts)
        consecutiu = self.last_hk is not None and hk == self.last_hk + 1
        if self.last_hk is not None and not consecutiu and self.soc_esp is not None:
            # salt de temps: si el SOC ha tornat al 50 %, el simulador reinicia la bateria cada setmana
            if abs(soc - 0.5 * cap) < 1e-3 and abs(self.soc_esp - soc) > 1.0:
                self.mode_reset = True
        if not self._dins_bloc(ts):
            self.salt = False
        self.last_hk = hk
        self.nstep += 1
        self._neteja(hk)

        preus = self._serie(obs.get("preus_properes_24h"))
        co2s = self._serie(obs.get("co2_properes_24h"))
        pn, cn = self._num(obs.get("preu_actual")), self._num(obs.get("co2_actual"))
        if not preus and pn is not None:
            preus = [pn]
        if not co2s and cn is not None:
            co2s = [cn]
        m = min(len(preus), len(co2s))
        if m == 0:
            return 0.0
        tl = self._linia_temps(ts, H)
        if m < 3:                     # la llista s'escurca: s'acaba la simulacio
            tl = tl[:m]
        self._obs_mercat(tl, preus, co2s)
        nd = obs.get("noves_dades_5min")
        if isinstance(nd, dict) and consecutiu:
            self._obs_campus(ts - HORA, self._num(nd.get("ultim_consum_kw")), self._num(nd.get("ultima_solar_kw")),
                             self._num(nd.get("irradiancia_wm2")), self._num(nd.get("temp_ambient_c")))

        pr, co, ld, pv, sd = self.forecast(tl, m, preus, co2s)
        if self.KM is not None:
            kc, kp = self.KM[ts.month - 1]
        else:
            kc, kp = self.CB, self.PB
        w = 0.7 * co / kc + 0.3 * pr / kp
        net = ld - pv
        acaba = len(tl) < H
        lam = 0.0 if acaba else self.lam * ETA * float(np.mean(w))
        return self._plan(soc, w, net, sd, lam, cap, pmax)
