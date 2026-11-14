"""
agent.py - BatteryAgent MODE DIFICIL (Datathon UPC Sostenible)

Bateria 6.000 kWh / +-1.500 kW, solar x12 (excedente a 0 EUR/kWh), mercado solo 3 h conocido.

Estrategia (control predictivo con prevision):
 1. PREDICCION: precio, CO2, consumo y solar de las proximas 24 h =
    perfil historico (mes x tipo de dia x hora, aprendido del train) + correcciones
    lineales por horizonte a partir de las desviaciones recientes (ultima hora conocida,
    mismo instante del dia anterior y media de las ultimas 24 h). La solar se modela como
    envolvente de cielo despejado x indice de nubosidad con persistencia.
 2. PLANIFICACION: programacion dinamica (solo numpy) sobre 24 h minimizando
    0.70*CO2 + 0.30*coste normalizados por el baseline, con coste esperado de importacion
    E[max(0, consumo - solar + p)] para tener en cuenta la incertidumbre de la prevision
    (evita descargar de mas y verter energia a 0 EUR/0 kg).
 3. EJECUCION: se aplica solo la primera consigna, con margen de seguridad fisico (0 infracciones).
"""
import math
from datetime import datetime, timedelta

import numpy as np
from tables import TABLES

THR = 100.0     # kW: umbral de "hay sol"
SMAX = 30
NK = 24


def _dtype(wd):
    return 0 if wd < 5 else wd - 4


class BatteryAgent:
    ETA = math.sqrt(0.90)
    W_CO2, W_COST = 0.70, 0.30
    SAFETY = 1e-3
    LAM = 0.7          # valor terminal del kWh sobrante (x eta x peso medio)
    SIGMA_K = 1.0      # escala de la incertidumbre en el coste esperado
    NS, NA = 91, 41    # rejilla de SOC y de acciones
    N0 = 72.0          # peso del prior en la normalizacion adaptativa

    def __init__(self):
        self.T = TABLES
        self.CAP = 6000.0
        self.PMAX = 1500.0
        self._build()
        self._reset_state()

    def _build(self):
        e = self.ETA
        self.S = np.linspace(0.0, self.CAP, self.NS)
        dS = self.S[1] - self.S[0]
        self.A = np.linspace(-self.PMAX, self.PMAX, self.NA)
        delta = np.where(self.A > 0, self.A * e, self.A / e)[:, None]
        sn = self.S[None, :] + delta
        valid = (sn >= -1e-9) & (sn <= self.CAP + 1e-9)
        sn = np.clip(sn, 0.0, self.CAP)
        pos = sn / dS
        self.i0 = np.minimum(np.floor(pos).astype(np.int64), self.NS - 2)
        self.i1 = self.i0 + 1
        self.fr = pos - self.i0
        self.pen = np.where(valid, 0.0, 1e9)

    def _reset_state(self):
        self.last_hk = None
        self.dev = {"price": {}, "co2": {}, "load": {}}
        self.raw = {"price": {}, "co2": {}}
        self.cdev = {}
        self.tlast = None
        self.crec = 0.0
        self.n_base = 0
        self.s_bc = 0.0
        self.s_bp = 0.0

    # ---------------- calendario / perfiles ----------------
    @staticmethod
    def _hk(ts):
        return ts.toordinal() * 24 + ts.hour

    def _prof(self, name, ts):
        return self.T[name]["P"][ts.month][_dtype(ts.weekday())][ts.hour]

    def _store(self, hk, ts_h, price=None, co2=None, load=None, pv=None):
        if price is not None:
            self.raw["price"][hk] = price
            self.dev["price"][hk] = price - self._prof("price", ts_h)
        if co2 is not None:
            self.raw["co2"][hk] = co2
            self.dev["co2"][hk] = co2 - self._prof("co2", ts_h)
        if load is not None:
            self.dev["load"][hk] = load - self._prof("load", ts_h)
        if pv is not None:
            env = self.T["pv"]["env"][ts_h.month][ts_h.hour]
            if env > THR:
                c = min(1.5, max(0.0, pv / env))
                cd = c - self.T["pv"]["cc"][ts_h.month]
                self.cdev[hk] = cd
                self.tlast = hk
                self.crec = cd

    def _prune(self, hk):
        lim = hk - 80
        for dct in (self.dev["price"], self.dev["co2"], self.dev["load"],
                    self.raw["price"], self.raw["co2"], self.cdev):
            old = [h for h in dct if h < lim]
            for h in old:
                del dct[h]

    # ---------------- decision ----------------
    def act(self, obs):
        try:
            p = self._act(obs)
        except Exception:
            p = 0.0
        if not math.isfinite(p):
            p = 0.0
        return float(p)

    def _act(self, obs):
        cap = float(obs.get("cap_max_kwh", self.CAP) or self.CAP)
        pmax = float(obs.get("power_max_kw", self.PMAX) or self.PMAX)
        if cap != self.CAP or pmax != self.PMAX:
            self.CAP, self.PMAX = cap, pmax
            self._build()

        ts = obs["timestamp"]
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts))
        hk = self._hk(ts)
        if self.last_hk is not None and hk - self.last_hk != 1:
            self._reset_state()
        self.last_hk = hk
        self._prune(hk)

        soc = float(obs["soc_kwh"])
        price = list(obs.get("preus_properes_24h") or [])
        co2 = list(obs.get("co2_properes_24h") or [])
        pn = float(obs["preu_actual"])
        cn = float(obs["co2_actual"])
        if not price or abs(price[0] - pn) > 1e-12:
            price.insert(0, pn)
        if not co2 or abs(co2[0] - cn) > 1e-12:
            co2.insert(0, cn)
        nd = obs.get("noves_dades_5min") or {}
        ll = float(nd.get("ultim_consum_kw", 0.0))
        lp = float(nd.get("ultima_solar_kw", 0.0))

        # guardar lo observado
        np_, nc_ = len(price), len(co2)
        for j in range(np_):
            self._store(hk + j, ts + timedelta(hours=j), price=price[j])
        for j in range(nc_):
            self._store(hk + j, ts + timedelta(hours=j), co2=co2[j])
        self._store(hk - 1, ts - timedelta(hours=1), load=ll, pv=lp)

        # normalizacion adaptativa (suma base_grid*co2 y base_grid*precio del baseline)
        if (hk - 1) in self.raw["co2"]:
            bg = max(0.0, ll - lp)
            self.s_bc += bg * self.raw["co2"][hk - 1]
            self.s_bp += bg * self.raw["price"].get(hk - 1, pn)
            self.n_base += 1
        n0 = self.N0
        Kc = (self.s_bc + n0 * self.T["Kc_m"][ts.month]) / (self.n_base + n0)
        Kp = (self.s_bp + n0 * self.T["Kp_m"][ts.month]) / (self.n_base + n0)

        H = NK
        pr = np.empty(H); co = np.empty(H); ld = np.empty(H); pv = np.empty(H); sd = np.empty(H)
        kp = self.T["price"]["coef"]; kc = self.T["co2"]["coef"]; kl = self.T["load"]["coef"]
        pcoef = self.T["pv"]["coef"]; psig = self.T["pv"]["sig"]
        lsig = self.T["load"]["sig"]
        ref_p = hk + min(np_, 3) - 1
        ref_c = hk + min(nc_, 3) - 1
        lastp = self.dev["price"].get(ref_p, 0.0)
        lastc = self.dev["co2"].get(ref_c, 0.0)
        lastl = self.dev["load"].get(hk - 1, 0.0)

        def m24(name, ref):
            v = [self.dev[name][h] for h in range(ref - 23, ref + 1) if h in self.dev[name]]
            return (sum(v) / len(v)) if len(v) >= 12 else 0.0

        mp = m24("price", ref_p); mc = m24("co2", ref_c); ml = m24("load", hk - 1)

        for k in range(H):
            Tk = ts + timedelta(hours=k)
            hT = hk + k
            if k < np_:
                pr[k] = price[k]
            else:
                c = kp[k]
                pr[k] = (self._prof("price", Tk) + c[0] + c[1] * lastp
                         + c[2] * self.dev["price"].get(hT - 24, 0.0) + c[3] * mp)
            if k < nc_:
                co[k] = co2[k]
            else:
                c = kc[k]
                co[k] = (self._prof("co2", Tk) + c[0] + c[1] * lastc
                         + c[2] * self.dev["co2"].get(hT - 24, 0.0) + c[3] * mc)
            c = kl[k]
            ld[k] = (self._prof("load", Tk) + c[0] + c[1] * lastl
                     + c[2] * self.dev["load"].get(hT - 24, 0.0) + c[3] * ml)
            env = self.T["pv"]["env"][Tk.month][Tk.hour]
            pv_sd = 0.0
            if env > THR:
                tl = self.tlast if self.tlast is not None else hT - SMAX
                s = min(SMAX, max(1, hT - tl))
                cd = pcoef[s][0] * self.crec + pcoef[s][1] * self.cdev.get(hT - 24, 0.0)
                cc = self.T["pv"]["cc"][Tk.month]
                pv[k] = env * min(1.3, max(0.0, cc + cd))
                pv_sd = psig[s] * env
            else:
                pv[k] = 0.0
            sd[k] = math.hypot(lsig[k], pv_sd)
        pr = np.maximum(pr, 0.0)
        co = np.maximum(co, 0.0)
        net = ld - pv
        w = self.W_CO2 * co / Kc + self.W_COST * pr / Kp
        lam = self.LAM * self.ETA * float(w.mean())
        return self._plan(soc, w, net, lam, sd)

    # ---------------- planificacion (DP) ----------------
    @staticmethod
    def _emax(m, s):
        z = m / s
        phi_cdf = 0.5 * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
        return m * phi_cdf + s * np.exp(-0.5 * z * z) * 0.3989422804

    def _cost(self, w, net, sd, t):
        m = net[t] + self.A
        if self.SIGMA_K > 0:
            return w[t] * self._emax(m, max(1e-6, self.SIGMA_K * sd[t]))
        return w[t] * np.maximum(0.0, m)

    def _plan(self, soc, w, net, lam, sd):
        H = len(w)
        e = self.ETA
        V = -lam * self.S
        for t in range(H - 1, 0, -1):
            c = self._cost(w, net, sd, t)
            Vn = V[self.i0] * (1.0 - self.fr) + V[self.i1] * self.fr
            V = (c[:, None] + Vn + self.pen).min(axis=0)
        max_c = max(0.0, (self.CAP - soc) / e - self.SAFETY)
        max_d = max(0.0, soc * e - self.SAFETY)
        hi = min(self.PMAX, max_c)
        lo = -min(self.PMAX, max_d)
        cand = np.concatenate([self.A, [max_c, -max_d, 0.0, -net[0]]])
        cand = np.clip(cand[(cand <= hi + 1e-9) & (cand >= lo - 1e-9)], lo, hi)
        sn = soc + np.where(cand > 0, cand * e, cand / e)
        m0 = net[0] + cand
        if self.SIGMA_K > 0:
            q = w[0] * self._emax(m0, max(1e-6, self.SIGMA_K * sd[0])) + np.interp(sn, self.S, V)
        else:
            q = w[0] * np.maximum(0.0, m0) + np.interp(sn, self.S, V)
        p = float(cand[int(np.argmin(q))])
        return 0.0 if abs(p) < 1e-6 else p