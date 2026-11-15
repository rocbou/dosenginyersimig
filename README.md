# BatteryAgent: control de bateria i fotovoltaica (UPC Sostenible, Datathon 2026, Mode Difícil)

Agent de control (`BatteryAgent`) per al repte **UPC Sostenible**: decidir hora a hora la potència de càrrega o descàrrega d'una bateria industrial al Campus Nord per **reduir emissions de CO₂ (70 %)** i **cost de la factura (30 %)**.

> **Equip:** Dos enginyers i mig · **Autors:** Andoni Pon, Didac Munne, Joan Garrido, Roc Bou
> **Resultat públic (Mode Difícil):** 23,73 pts.

---

## 1. Resum

- **Planificació òptima** per programació dinàmica sobre l'estat de càrrega, amb un objectiu idèntic a la fórmula oficial del Score.
- **Previsió de 24 h** de preu, CO₂, consum i solar amb perfils per mes/tipus de dia/hora i **regressions per horitzó** sobre les anomalies, perquè el mercat només és visible 3 h.
- **Incertesa explícita:** es minimitza la importació *esperada* de la xarxa, no una previsió puntual.
- **Seguretat física estricta:** límits conservadors per no cometre infraccions (−0,20 pts cadascuna).
- **Robust a l'estructura del test:** detecta sol si el test salta de setmana o reinicia la bateria.
- **Ràpid:** ≈ 4 s per simular 2016 h (límit: 60 s).

## 2. El repte en una taula

| | |
|---|---|
| Bateria | 6.000 kWh, ±1.500 kW, rendiment round-trip 90 %, SOC inicial 50 % |
| Solar | ×12 (fins a ≈ 1.650 kW) |
| Mercat visible | **només 3 h** (hora actual + 2) |
| Avaluació | 12 setmanes, 2.016 hores |
| Score | `max(0, 0,70·ΔCO₂(%) + 0,30·ΔCost(%) − 0,20·infraccions)` |
| Excedent solar | s'aboca a la xarxa a 0 € i 0 kg CO₂ (no es cobra) |
| Límits d'execució | 60 s, 25 MB, escàner AST (sense `eval`, `exec`, xarxa ni sistema) |
| Biblioteques | numpy, pandas, scipy, scikit-learn, joblib, xgboost, lightgbm, math, datetime, collections, itertools, random, json, pathlib, typing |

## 3. Estructura del repositori

```
.
├── agent.py               # BatteryAgent (el que s'envia)
├── parametres.json        # taules de perfils, coeficients i hiperparàmetres (el que s'envia)
├── README.md              # aquest document
```

**Lliurament:** `submission.zip` amb `agent.py` i `parametres.json` **a l'arrel** (sense subcarpetes).

## 4. Com funciona

```
obs ──► memòria ──► previsió 24 h ──► pes w = 0,7·CO2/CB + 0,3·preu/PB
                                          │
                              programació dinàmica (SOC)
                                          │
                             decisió de l'hora actual ──► clamp físic ──► kW
```

1. **Memòria.** Cada hora desa preu, CO₂, consum, solar i meteo, i calcula l'**anomalia** de cada un respecte al seu perfil.
2. **Previsió.** Per a cada hora futura: `perfil + Σ coeficient·variables`, amb coeficients per **horitzó** i **hora del dia**. Les variables són les anomalies més recents, les de fa 24 h, l'anomalia de l'altra variable de mercat, la temperatura i la claredat solar.
3. **Incertesa.** Cada hora té una desviació típica (consum i solar) que depèn de quines dades hi ha disponibles.
4. **Planificació.** Es minimitza `Σ w_t · E[max(0, consum − solar + a_t)]`. L'esperança d'una normal truncada té forma tancada: `μ·Φ(μ/σ) + σ·φ(μ/σ)`.
5. **Valor terminal.** L'energia que sobra a l'hora 24 es valora a `lam·η·mitjana(w)` per kWh; `lam = 0` quan l'horitzó coincideix amb el final d'una setmana amb reinici o amb la simulació.
6. **Acció.** Només s'aplica la decisió de l'hora actual (horitzó lliscant) i es passa pel clamp físic.

Explicació detallada, línia per línia: **[`EXPLICACIO_CODI.md`](EXPLICACIO_CODI.md)**.

## 5. `parametres.json`: què conté i per què

| Clau | Forma | Què representa | Justificació |
|---|---|---|---|
| `TP`, `TC` | 12×2×24 | Preu i CO₂ típics per mes, tipus de dia (feiner/festiu) i hora | L'estacionalitat i el calendari expliquen la major part de la variància |
| `TL` | 12×2×24 | Consum típic per mes, obert/tancat i hora | El campus té una forta dependència acadèmica |
| `TV`, `TVX`, `TI`, `TT` | 12×24 | Solar típica, **envolupant** solar, irradiància i temperatura per mes i hora solar | Se separa el "sol segur" de l'efecte dels núvols |
| `RP`, `RC` | 7×24 | Residus per dia de la setmana (preu, CO₂) | Corregeix patrons setmanals; el valor és petit (±0,009) |
| `RL` | 4×24 | Residus de consum per combinació ahir/avui tancat | Captura transicions (dilluns, post-festiu) |
| `coef` | per variable/variant/horitzó/hora | Coeficients de regressió sobre anomalies | Converteix la informació recent en una correcció de perfil |
| `sig` | idem | Desviació típica de l'error per cas | Alimenta la incertesa del planificador |
| `centre`, `g_p`, `g_c`, `g_v` | escalars | Pivot i guany d'interpolació entre mesos (preu 0, CO₂ 1, solar 1) | Suavitza el canvi de mes |
| `CB`, `PB` | escalars | Normalitzadors per kWh de CO₂ i de cost | Fan que `w` reprodueixi el pes relatiu de la fórmula del Score |
| `sigma_k`, `lam`, `kappa` | escalars | Escala de la incertesa, valor terminal, valor d'opció | Hiperparàmetres heurístics (vegeu §8) |
| `salt`, `bloc`, `final_ts`, `mode_reset` | | Estructura esperada del test | S'autoajusten en execució |

## 6. Seguretat, infraccions i normativa

- **Límits físics.** `max_càrrega = min(pmax − ε, (cap − SOC)/0,9487 − ε)` i `max_descàrrega = min(pmax − ε, SOC·√0,9 − ε)`, amb `ε = 1e-3`. Es fa servir el rendiment **més estricte** en cada sentit, per no fallar encara que l'entorn arrodoneixi η.
- **Capacitat i potència** llegides de `obs["cap_max_kwh"]` i `obs["power_max_kw"]`, mai deduïdes de `soc_pct`.
- **Errors.** Qualsevol excepció a `act()` retorna 0 kW en lloc de fer fallar l'avaluació.
- **Compliment.** Només es fan servir biblioteques permeses; no hi ha `eval`, `exec`, xarxa ni crides de sistema; el temps és ≈ 4 s per a 2016 h.

## 7. Historial del projecte i lliçons

| Fase | Què es va fer | Resultat / lliçó |
|---|---|---|
| **1. Agent inicial (LP)** | Perfils mensuals + persistència + programació lineal a 30 h amb 3 escenaris | Funcionava, però rebia 3 infraccions (−0,6 pts) al Mode Difícil |
| **2. Diagnòstic d'infraccions** | Es va deduir la capacitat de `soc/soc_pct`. Si `soc_pct` ve arrodonit el límit queda mal calculat | A l'entorn de proves apareixien 42–101 infraccions. **Solució:** usar `cap_max_kwh` |
| **3. Mercat de 3 h** | L'enunciat del Mode Difícil només mostra 3 h de mercat | Cal predir +3…+24 h: la previsió passa a ser el nucli |
| **4. Escàner de seguretat** | La web va rebutjar la crida `stack()` | Es van eliminar `np.stack`, `np.linalg.solve` i noms sospitosos. **Lliçó:** auditar les crides abans de lliurar |
| **5. Agent actual (DP + regressions)** | Programació dinàmica amb importació esperada, regressions per horitzó, festius, meteo, detecció d'estructura del test | **23,73 pts** públics |
| **6. Auditoria i endurir el clamp** | Auditoria del codi i dels paràmetres; marge de seguretat reforçat | Mateixa puntuació; **0 infraccions** davant d'un entorn amb η arrodonit (abans: 79) |

## 8. Experiments i resultats negatius

Tot sobre un **entorn sintètic** construït amb les nostres pròpies taules més soroll AR. Serveix per **comparar variants entre elles**, **no** per predir la nota real.

| Experiment | Resultat | Decisió |
|---|---|---|
| Sostre amb previsió perfecta (LP d'oracle) | L'agent assoleix ≈ **95 %** del sostre | El marge per millorar la previsió és petit |
| Sensibilitat a `lam`, `kappa`, `sigma_k` | Variacions ≤ 0,05 pts | Els hiperparàmetres no són el punt crític |
| DP amb resolució 4× | +0,01 pts | Ja està saturat; es manté 151×101 |
| Capa de **nivell lent** (mitjana mòbil de 7 dies) | Empitjora els escenaris amb desplaçament de distribució (fins a −1,3 pts) | **Descartada** |
| Suavitzat de coeficients entre hores | L'efecte esperat és ≈ 0,5–2 % de l'error de previsió i no es pot validar sense dades | **No s'aplica** |
| Robustesa a l'estructura del test (setmanes diferents, sense reinici) | Diferències < 0,1 pts | L'autodetecció funciona |
| Rendiment d'entorn η = 0,9487 | 79 infraccions → 0 amb el clamp nou | **Aplicat** |

**Diagnòstic de coeficients.** Els pesos `a0` i `a2` de la regressió de preu i CO₂ són multicolinears (correlació −0,75 i −0,86) i entre el 8 % i el 25 % de les cel·les tenen |pes| > 1. Un Monte-Carlo controlat indica que això només afegeix ≈ 0,5 % (365 mostres per cel·la) a ≈ 2 % (100 mostres) d'error de previsió.

## 9. Limitacions (honestedat tècnica)

1. **No hi ha validació sobre dades reals en aquest repositori.** Les xifres de §8 vénen d'un entorn sintètic; són relatives, no una predicció de la nota privada.
2. **Calendari fixat a mà.** `ESPECIALS` i `NACIONALS` són dates i fraccions de tancament estimades. Un dia especial no previst es tracta com un dia feiner.
3. **Taules d'un període limitat.** Si el mercat o el consum del període privat difereix del d'entrenament, la previsió pot desviar-se.
4. **Hiperparàmetres heurístics** (`lam`, `kappa`, `sigma_k`, `THR`, `AVARIA`) afinats amb el rànquing públic.
5. **El rànquing públic no és el privat.** Una puntuació molt per sobre de la resta pot indicar sobreajust a les setmanes públiques; el sostre d'oracle és una bona prova de realitat.

## 10. Treball futur

- **Reentrenar** les regressions amb validació per blocs temporals (reservant setmanes) i amb regularització (ridge) per reduir la multicolinealitat.
- **Calibrar** l'error de previsió fora de mostra i ajustar `sig` en conseqüència.
- **Estimar** les fraccions de `ESPECIALS` a partir de les dades en comptes de fixar-les a mà.
- **Comparar** amb models de gradient boosting per horitzó (lightgbm/xgboost, permesos per la normativa) i desar-ne només els coeficients o arbres en format segur.
- **Eines opcionals** (`eines/`): `validar.py` calcula la puntuació oficial i el sostre d'oracle sobre setmanes locals, i `explorar.py` accepta una variant només si guanya de manera consistent en la majoria de setmanes.

## 11. Mode d'ús local (opcional)

```python
import json
from agent import BatteryAgent

agent = BatteryAgent()              # llegeix parametres.json de la mateixa carpeta
kw = agent.act(obs)                 # obs: diccionari del simulador; retorna kW (float)
```

Requisits: `numpy`, `scipy` (i `pandas` només per a les eines locals).

## 12. Llicència i ús

MIT License

Copyright (c) 2026 Dos enginyers i mig

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
