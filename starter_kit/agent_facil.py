"""
BatteryAgent: control predictiu (MPC) amb programacio lineal exacta.
A cada hora resol el pla optim per a les properes 24 h i n'aplica nomes la primera consigna.
"""
import numpy as np
from scipy.optimize import linprog

HORITZO = 24


class BatteryAgent:
    def __init__(self):
        self.cap = 2000.0                 # kWh
        self.pmax = 500.0                 # kW
        self.eta = 0.9 ** 0.5             # rendiment de carrega i de descarrega
        self.avg_co2 = 0.085              # kg/kWh mitja (train)
        self.avg_price = 0.138            # EUR/kWh mitja (train)
        self.term = 0.0                   # valor terminal de l'energia sobrant
        self._cache = {}
        self.k_preu = 0.9                 # >1 dona més pes al preu, <1 dona més pes al CO2

    def _matrius(self, n):
        # Matrius del problema, nomes dependents de n (es calculen un cop)
        if n not in self._cache:
            L = np.tril(np.ones((n, n)))
            Sc = np.hstack([self.eta * L, -L / self.eta])   # SOC_k - SOC_0
            A = np.vstack([Sc, -Sc])
            self._cache[n] = (Sc, A, [(0.0, self.pmax)] * (2 * n))
        return self._cache[n]

    def act(self, obs):
        soc = float(obs["soc_kwh"])
        pr = np.asarray(obs["preus_properes_24h"], float)
        co = np.asarray(obs["co2_properes_24h"], float)
        n = len(pr)
        if n == 0:
            return 0.0

        # Pes per kWh importat: replica el 70% CO2 / 30% cost de la puntuacio
        w = 0.7 * co / self.avg_co2 + 0.3 * self.k_preu * pr / self.avg_price

        _, A, bounds = self._matrius(n)
        eta = self.eta
        # variables: [c_1..c_n (carrega), d_1..d_n (descarrega)]
        cost = np.concatenate([w, -w])
        # valor terminal: l'energia que queda a la bateria al final te un valor
        cost = cost - self.term * w.mean() * np.concatenate(
            [eta * np.ones(n), -np.ones(n) / eta])
        b = np.concatenate([np.full(n, self.cap - soc), np.full(n, soc)])

        p = 0.0
        res = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if res.status == 0:
            p = float(res.x[0] - res.x[n])

        # Retallada de seguretat: mai infraccions fisiques
        max_c = min(self.pmax, (self.cap - soc) / eta)
        max_d = min(self.pmax, soc * eta)
        return float(max(-max_d, min(max_c, p)))
