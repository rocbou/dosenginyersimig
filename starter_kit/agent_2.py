"""
agent.py - Repte UPC Sostenible, Mode Dificil
(bateria 6.000 kWh / +-1.500 kW, solar x12, nomes 3 h de mercat a obs)

Tot el que l'agent ha apres del train es a model.json, que genera train.py llegint
directament els CSV de data/train. Son dues coses:
  - PERFILS: la mitjana de preu, CO2, consum i solar per mes, tipus de dia i hora.
  - PERSISTENCIA: per a cada horitzo (1 h endavant, 2 h endavant, ...), quina part de la
    desviacio d'avui respecte al perfil es mante. Un numero per horitzo i variable.

Com decideix, cada hora:
  1. PREDICCIO de les properes 24 h:   valor = perfil + persistencia x desviacio d'avui
       mercat (preu, CO2): desviacio de les 3 h conegudes (la mitjana i l'ultima)
       consum:             desviacio de la darrera hora mesurada
       solar:              "claredat" de l'ultima hora amb sol mesurada
  2. PLA: programacio dinamica sobre l'estat de carrega que minimitza
        0,7 * CO2 + 0,3 * cost   de la importacio ESPERADA de la xarxa,
     comptant amb l'error de la prediccio de consum i solar (descarregar mes del que
     consumeix el campus, o esperar un excedent solar que no arriba, no serveix de res).
  3. ACCIO: aplica nomes la consigna de l'hora actual, retallada als limits fisics.

Regles que surten de l'enunciat o de mirar les dades (no son parametres ajustats):
  - El test acaba el 31 d'agost de 2026 a les 23 h: l'energia que quedi llavors no compta.
  - Calendari: festius i dies de campus tancat (consum de cap de setmana).
  - La solar i el CO2 de la xarxa segueixen l'hora solar, no l'oficial (canvi d'hora).
  - De 22 h a 5 h el comptador del campus repeteix el mateix valor: de nit, el consum
    de les hores seguents es l'ultim mesurat.
  - Si el comptador marca ~0 kW es una avaria: descarregar no estalvia res.
"""
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ETA = math.sqrt(0.90)       # rendiment de carrega i de descarrega
H = 24                      # hores que es planifiquen
HORA = timedelta(hours=1)
SOL_MIN = 50.0              # kW de solar maxima a partir dels quals "hi ha sol"
EDAT_SOL = 12               # hores durant les quals serveix l'ultima mesura amb sol
AVARIA = 150.0              # kW: per sota, la lectura del comptador es una avaria
RHO_AVARIA = 0.9            # probabilitat que una avaria continui l'hora seguent
NIT = (22, 23, 0, 1, 2, 3, 4, 5)   # el consum mesurat de 22 h a 4 h es repeteix fins a les 5 h

# Festius nacionals de data fixa: el PVPC (peatge 2.0TD) els tracta com a cap de setmana.
NACIONALS = {(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)}
# Dies especials del campus: 1 = tancat (com un festiu), valors intermedis = activitat reduida.
ESPECIALS = {
    (2025, 9, 11): 1.0, (2025, 9, 24): 1.0, (2025, 12, 8): 1.0,
    (2025, 12, 24): 0.5, (2025, 12, 25): 1.0, (2025, 12, 26): 1.0,
    (2025, 12, 29): 0.8, (2025, 12, 30): 0.8, (2025, 12, 31): 0.8,
    (2026, 1, 1): 1.0, (2026, 1, 2): 0.8, (2026, 1, 5): 0.8, (2026, 1, 6): 1.0,
    (2026, 3, 30): 0.5, (2026, 3, 31): 0.5, (2026, 4, 1): 0.5, (2026, 4, 2): 0.5,
    (2026, 4, 3): 1.0, (2026, 4, 6): 1.0, (2026, 5, 1): 1.0, (2026, 5, 25): 1.0,
    (2026, 6, 24): 1.0, (2026, 9, 11): 1.0, (2026, 9, 24): 1.0,
}


def dies_del_mes(y, m):
    return (datetime(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).day


def ultim_diumenge(y, m):
    d = datetime(y, m, dies_del_mes(y, m))
    return d - timedelta(days=(d.weekday() + 1) % 7)


def frac_tancat(d):
    """0 = feiner normal, 1 = campus tancat."""
    if d.weekday() >= 5 or (d.month, d.day) in NACIONALS:
        return 1.0
    return ESPECIALS.get((d.year, d.month, d.day), 0.0)


def hkey(ts):
    """numero d'hora absolut"""
    return ts.toordinal() * 24 + ts.hour


class BatteryAgent:
    NS, NA = 151, 101       # malla de la programacio dinamica (estat de carrega, consigna)

    def __init__(self, params_path=None, params=None):
        if params is None:
            p = Path(params_path) if params_path else Path(__file__).parent / "model.json"
            with open(p, encoding="utf-8") as f:
                params = json.load(f)
        P = self.P = params
        T = P["perfils"]
        self.TP, self.TC, self.TL, self.TV, self.TVX = T["preu"], T["co2"], T["consum"], T["solar"], T["solar_max"]
        self.coef = P.get("persistencia", {})     # {"preu": [k][2], "co2": [k][2], "consum": [k][1], "solar": [edat][1]}
        self.sig = P.get("error", {})             # desviacio tipica de l'error de prediccio
        self.centre = P.get("centre", 12.0)       # dia mitja de les dades de cada mes al train
        self.CB, self.PB = P.get("co2_mitja", 0.0933), P.get("preu_mitja", 0.1528)
        self.sigma_k = P.get("sigma_k", 1.0)
        self.kappa = P.get("kappa", 0.0)
        self.lam = P.get("valor_final", 0.8)
        self.bloc = P.get("bloc")                 # None = ultims 7 dies del mes; o [dia_ini, dia_fi] (validacio)
        ft = P.get("final_ts", "2026-08-31 23:00:00")
        self.final_ts = datetime.fromisoformat(ft) if ft else None
        self.mode_reset = False
        self._cc = {}
        self.n_err = 0
        self._malla = None
        # memoria: nomes l'ultima mesura de cada cosa
        self.consum = None                        # (hora, kW, desviacio respecte al perfil)
        self.sol = None                           # (hora, claredat) de l'ultima hora amb sol
        self.avaria = None                        # hora de l'ultima lectura ~0 del comptador
        self.last_hk = None
        self.soc_esp = None

    # ================================================================ calendari i perfils
    def _ctx(self, ts):
        key = ts.toordinal()
        c = self._cc.get(key)
        if c is None:
            y, m, d = ts.year, ts.month, ts.day
            dia = datetime(y, m, d)
            tp = 1 if (dia.weekday() >= 5 or (m, d) in NACIONALS) else 0   # tipus de dia per al mercat
            ml = 9 if (m == 8 and d >= 24) else m                          # el 24 d'agost el campus ja torna
            dst = 1 if ultim_diumenge(y, 3) <= dia < ultim_diumenge(y, 10) else 0
            a = (d - self.centre) / 30.0                                   # posicio dins del mes -> mes vei
            mb = (m % 12 + 1) if a > 0 else ((m - 2) % 12 + 1)
            c = (m - 1, tp, ml - 1, frac_tancat(dia), dst, mb - 1, min(1.0, abs(a)))
            self._cc[key] = c
        return c

    def _bp(self, c, h):
        """perfil de preu: feiners en hora oficial (peatges), caps de setmana en hora solar"""
        return self.TP[c[0]][c[1]][(h - c[4]) % 24 if c[1] == 1 else h]

    def _bc(self, c, h):
        """perfil de CO2 en hora solar, interpolat cap al mes vei (el test es a final de mes)"""
        hs = (h - c[4]) % 24
        v = self.TC[c[0]][c[1]][hs]
        if c[6] > 0 and not (c[0] == 7 and c[5] == 8):      # agost 2026 -> setembre 2025 no son veins
            v = (1 - c[6]) * v + c[6] * self.TC[c[5]][c[1]][hs]
        return v

    def _bl(self, c, h):
        f, t = c[3], self.TL[c[2]]
        return (1 - f) * t[0][h] + f * t[1][h]

    def _bv(self, T, c, h):
        """perfil solar en hora solar, interpolat cap al mes vei"""
        hs = max(h - c[4], 0)
        return (1 - c[6]) * T[c[0]][hs] + c[6] * T[c[5]][hs]

    # ================================================================ hores a planificar
    def _fi_setmana(self, ts):
        """ultim dia de la setmana de test (l'enunciat: els ultims 7 dies de cada mes)"""
        dim = dies_del_mes(ts.year, ts.month)
        return dim if self.bloc is None else min(self.bloc[1], dim)

    def _linia_temps(self, ts, n):
        """[(ts, context)] de les properes hores, fins a n o fins que s'acabi la simulacio."""
        out, t = [], ts
        for _ in range(n):
            out.append((t, self._ctx(t)))
            if self.final_ts is not None and t == self.final_ts:
                break                               # final del test: el que quedi a la bateria no compta
            if self.mode_reset and t.hour == 23 and t.day == self._fi_setmana(t):
                break                               # la bateria es reinicia en acabar cada setmana
            t = t + HORA
        return out

    # ================================================================ memoria
    def _mesura(self, tprev, load=None, pv=None):
        """guarda el consum i la solar mesurats durant l'hora anterior"""
        k, c, h = hkey(tprev), self._ctx(tprev), tprev.hour
        if load is not None:
            if load < AVARIA:
                self.avaria = k
            else:
                self.consum = (k, load, load - self._bl(c, h))
        if pv is not None:
            mx = self._bv(self.TVX, c, h)
            if mx > SOL_MIN:
                self.sol = (k, (pv - self._bv(self.TV, c, h)) / mx)

    # ================================================================ prediccio
    def variables(self, tl, preus, co2s):
        """Perfils base i, per variable, [(k, index del coeficient, x)] de cada hora a predir.
        train.py fa servir aquest mateix codi per ajustar els coeficients."""
        n, hk, m = len(tl), hkey(tl[0][0]), min(len(preus), len(co2s))
        base = {"preu": [self._bp(c, t.hour) for t, c in tl], "co2": [self._bc(c, t.hour) for t, c in tl],
                "consum": [self._bl(c, t.hour) for t, c in tl], "solar": [self._bv(self.TV, c, t.hour) for t, c in tl]}
        mx = [self._bv(self.TVX, c, t.hour) for t, c in tl]
        X = {"preu": [], "co2": [], "consum": [], "solar": []}
        fix = {}
        # mercat: desviacio de les hores conegudes (la mitjana i l'ultima)
        for nom, vals in (("preu", preus), ("co2", co2s)):
            an = [vals[j] - base[nom][j] for j in range(min(m, n))]
            if an:
                x = (sum(an) / len(an), an[-1])
                X[nom] = [(k, k, x) for k in range(m, n)]
        # consum: de nit es repeteix l'ultim valor; la resta, desviacio de la darrera hora mesurada
        if self.consum is not None and self.consum[0] == hk - 1:
            k0 = 0
            if (tl[0][0] - HORA).hour in NIT[:-1]:
                while k0 < n and tl[k0][0].hour in NIT[1:]:
                    fix[k0] = self.consum[1]
                    k0 += 1
            X["consum"] = [(k, k, (self.consum[2],)) for k in range(k0, n)]
        # solar: claredat de l'ultima hora amb sol; el coeficient depen de l'antiguitat de la mesura
        if self.sol is not None:
            for k in range(n):
                edat = hk + k - self.sol[0]
                if mx[k] > SOL_MIN and 1 <= edat <= EDAT_SOL:
                    X["solar"].append((k, edat, (self.sol[1] * mx[k],)))
        return base, mx, X, fix

    def prediu(self, tl, preus, co2s):
        n, m = len(tl), min(len(preus), len(co2s))
        base, mx, X, fix = self.variables(tl, preus, co2s)
        out = {nom: list(v) for nom, v in base.items()}
        S = self.sig
        s = {nom: [S[nom][-1]] * n for nom in out}            # l'ultim valor es l'error sense cap mesura
        for nom, files in X.items():
            C = self.coef.get(nom)
            if C is None:
                continue
            for k, i, x in files:
                out[nom][k] += sum(b * xi for b, xi in zip(C[i], x))
                s[nom][k] = S[nom][i]
        sd, sw = np.zeros(n), np.zeros(n)
        self._pn = np.ones(n)
        en_avaria = self.avaria is not None and self.avaria == hkey(tl[0][0]) - 1
        for k in range(n):
            if k < m:
                out["preu"][k], out["co2"][k] = preus[k], co2s[k]
            else:
                sw[k] = math.hypot(0.3 * s["preu"][k] / self.PB, 0.7 * s["co2"][k] / self.CB)
            if k in fix:
                out["consum"][k], s["consum"][k] = fix[k], 0.0
            out["consum"][k] = max(out["consum"][k], 0.0)
            hi_ha_sol = mx[k] > SOL_MIN
            out["solar"][k] = min(max(out["solar"][k], 0.0), 1.15 * mx[k]) if mx[k] > 1.0 else 0.0
            sd[k] = math.hypot(s["consum"][k], s["solar"][k] * mx[k] if hi_ha_sol else 0.0)
            if en_avaria:
                self._pn[k] = 1.0 - RHO_AVARIA ** (k + 1)
        self._pv = np.array(out["solar"])
        return (np.maximum(np.array(out["preu"]), 0.0), np.maximum(np.array(out["co2"]), 0.0),
                np.array(out["consum"]), self._pv, sd, sw)

    # ================================================================ pla (programacio dinamica)
    def _fes_malla(self, cap, pmax):
        self._malla = (cap, pmax)
        self.S = np.linspace(0.0, cap, self.NS)
        dS = self.S[1] - self.S[0]
        self.A = np.linspace(-pmax, pmax, self.NA)
        sn = self.S[None, :] + np.where(self.A > 0, self.A * ETA, self.A / ETA)[:, None]
        self.pen = np.where((sn >= -1e-9) & (sn <= cap + 1e-9), 0.0, 1e9)
        pos = np.clip(sn, 0.0, cap) / dS
        self.i0 = np.minimum(np.floor(pos).astype(np.int64), self.NS - 2)
        self.fr = pos - self.i0

    def _importacio(self, t, a, net, sd):
        """importacio esperada de la xarxa a l'hora t amb consigna a: E[max(0, net + a)], net ~ N(net_t, sd_t)"""
        mu, s = net[t] + a, self.sigma_k * sd[t]
        if s < 1e-6:
            f = np.maximum(0.0, mu)
        else:
            z = mu / s
            f = mu * ndtr(z) + s * 0.3989422804014327 * np.exp(-0.5 * z * z)
        if self._pn[t] < 1.0:                     # comptador avariat: amb prob. 1-pn el consum "es" zero
            f = self._pn[t] * f + (1.0 - self._pn[t]) * np.maximum(0.0, a - self._pv[t])
        return f

    def _pla(self, soc, w, net, sd, sw, lam, cap, pmax):
        if self._malla != (cap, pmax):
            self._fes_malla(cap, pmax)
        V = -lam * self.S                         # valor de l'energia que queda al final de l'horitzo
        for t in range(len(w) - 1, 0, -1):
            f = self._importacio(t, self.A, net, sd)
            c = w[t] * f
            if self.kappa > 0 and sw[t] > 0:      # el mercat futur es incert i s'hi podra reaccionar: val una mica mes
                c = c - self.kappa * sw[t] * np.abs(f - float(self._importacio(t, np.zeros(1), net, sd)[0]))
            Vn = V[self.i0] * (1.0 - self.fr) + V[self.i0 + 1] * self.fr
            V = (c[:, None] + Vn + self.pen).min(axis=0)
        max_c = max(0.0, min(pmax, (cap - soc) / ETA) - 1e-6)
        max_d = max(0.0, min(pmax, soc * ETA) - 1e-6)
        cand = np.clip(np.concatenate([self.A, np.linspace(-max_d, max_c, 121), [0.0, -net[0]]]), -max_d, max_c)
        sn = soc + np.where(cand > 0, cand * ETA, cand / ETA)
        q = w[0] * self._importacio(0, cand, net, sd) + np.interp(sn, self.S, V) + 1e-9 * np.abs(cand)
        return float(cand[int(np.argmin(q))])

    # ================================================================ decisio
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
        try:
            a = self._decideix(obs, soc, cap, pmax)
        except Exception:
            self.n_err += 1
            a = 0.0
        # limits fisics exactes: mai cap infraccio
        max_c = max(0.0, min(pmax, (cap - soc) / ETA) - 1e-6)
        max_d = max(0.0, min(pmax, soc * ETA) - 1e-6)
        a = max(-max_d, min(max_c, a)) if math.isfinite(a) else 0.0
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
        seguit = self.last_hk is not None and hk == self.last_hk + 1
        if self.last_hk is not None and not seguit and self.soc_esp is not None:
            # salt de temps i la bateria torna a ser al 50 %: el simulador la reinicia cada setmana
            if abs(soc - 0.5 * cap) < 1e-3 and abs(self.soc_esp - soc) > 1.0:
                self.mode_reset = True
        self.last_hk = hk

        preus = self._serie(obs.get("preus_properes_24h"))
        co2s = self._serie(obs.get("co2_properes_24h"))
        pn, cn = self._num(obs.get("preu_actual")), self._num(obs.get("co2_actual"))
        preus = preus or ([pn] if pn is not None else [])
        co2s = co2s or ([cn] if cn is not None else [])
        m = min(len(preus), len(co2s))
        if m == 0:
            return 0.0
        tl = self._linia_temps(ts, H)
        if m < 3:                                 # la llista s'escurca: s'acaba la simulacio
            tl = tl[:m]
        nd = obs.get("noves_dades_5min")
        if isinstance(nd, dict) and seguit:
            self._mesura(ts - HORA, self._num(nd.get("ultim_consum_kw")), self._num(nd.get("ultima_solar_kw")))

        pr, co, ld, pv, sd, sw = self.prediu(tl, preus, co2s)
        w = 0.7 * co / self.CB + 0.3 * pr / self.PB          # com de dolent es importar 1 kWh a cada hora
        lam = 0.0 if len(tl) < H else self.lam * ETA * float(np.mean(w))
        return self._pla(soc, w, ld - pv, sd, sw, lam, cap, pmax)
