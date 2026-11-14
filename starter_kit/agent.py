"""
Hard Mode battery controller for the UPC Datathon.

System assumptions: 6,000 kWh capacity, +/-1,500 kW inverter, PV generation
scaled by 12, and only three hours of market data available in each observation.

The controller forecasts the next 24 hours from month/day-type/hour profiles
and online residual corrections. It then uses a finite-horizon dynamic program
to minimize the official 70% CO2 / 30% cost objective. Expected grid imports
are used to reduce over-discharge when load and PV forecasts are uncertain.
Only the first action is applied; physical limits are enforced before return.

All learned profiles and coefficients are embedded below, so no external model
file or training-data access is required at evaluation time.
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
    NS, NA = 151, 101  # SOC and action grid resolution
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

    # ---------------- calendar and profiles ----------------
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

    # ---------------- control decision ----------------
    def act(self, obs):
        try:
            p = self._act(obs)
        except Exception:
            p = 0.0
        if not math.isfinite(p):
            p = 0.0
        return float(p)

    @staticmethod
    def _number(value, default=None):
        """Convert an optional observation to a finite float."""
        try:
            x = float(value)
            return x if math.isfinite(x) else default
        except (TypeError, ValueError, OverflowError):
            return default

    @staticmethod
    def _series(value, limit=24):
        """Normalize JSON/list/numpy forecast inputs without ambiguous truth tests."""
        if value is None:
            return []
        try:
            values = list(value)[:limit]
        except TypeError:
            return []
        out = []
        for item in values:
            x = BatteryAgent._number(item)
            if x is None:
                break
            out.append(x)
        return out

    def _act(self, obs):
        cap = self._number(obs.get("cap_max_kwh"), self.CAP)
        pmax = self._number(obs.get("power_max_kw"), self.PMAX)
        if cap is None or cap <= 0:
            cap = self.CAP
        if pmax is None or pmax <= 0:
            pmax = self.PMAX
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

        soc = self._number(obs.get("soc_kwh"), 0.0)
        soc = min(self.CAP, max(0.0, soc))
        price = self._series(obs.get("preus_properes_24h"))
        co2 = self._series(obs.get("co2_properes_24h"))
        pn = self._number(obs.get("preu_actual"), 0.0)
        cn = self._number(obs.get("co2_actual"), 0.0)
        if not price or abs(price[0] - pn) > 1e-12:
            price.insert(0, pn)
        if not co2 or abs(co2[0] - cn) > 1e-12:
            co2.insert(0, cn)
        price = [max(0.0, x) for x in price[:NK]]
        co2 = [max(0.0, x) for x in co2[:NK]]
        pn, cn = max(0.0, pn), max(0.0, cn)
        nd = obs.get("noves_dades_5min")
        nd = nd if isinstance(nd, dict) else {}
        # Missing telemetry is NOT a zero measurement. Treating it as zero
        # corrupts the online load/PV residuals and can destabilize forecasts.
        ll = self._number(nd.get("ultim_consum_kw"))
        lp = self._number(nd.get("ultima_solar_kw"))

        # Store observations available at the current step.
        np_, nc_ = len(price), len(co2)
        for j in range(np_):
            self._store(hk + j, ts + timedelta(hours=j), price=price[j])
        for j in range(nc_):
            self._store(hk + j, ts + timedelta(hours=j), co2=co2[j])
        if ll is not None or lp is not None:
            self._store(hk - 1, ts - timedelta(hours=1), load=ll, pv=lp)

        # Adaptive normalization using observed baseline CO2 and electricity cost.
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

    # ---------------- dynamic programming planner ----------------
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