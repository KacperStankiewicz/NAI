"""
=====================================
        Adaptacyjny Tempomat
=====================================

Aby uruchomić program należy zainstalować następujące biblioteki
pip install scikit-fuzzy
pip install matplotlib

Adaptacyjny Tempomat
--------------------

* Wejście
   - `odległość`
      * crisp: Jak daleko jestem od pojazdu przede mną w metrach (0-500)?
      * fuzzy: blisko, optymalnie, daleko
   - `prędkość aktualna`
      * crisp: Z jaką prędkością się poruszam (0 - 150) w kilometrach na godzinę?
      * fuzzy: mała, średnia, wysoka
   -  `masa pojazdu`
      * crisp: Ile waży pojazd w tonach (0.8-40)?
      * fuzzy: lekki, średni, cięzki
* Wyjście
   - `przepustnica`
      * crisp: jak szerko otworzyć przepustnicę (0-100%)?
      * fuzzy: zamknięta, lekko otwarta, średnio otwarta, mocno otwarta
* Zasady
   - jeśli odległość = blisko => przepustnica = zamknięta
   - jeśli odległość = optymalnie i prędkość = średnia => przepustnica = lekko otwarta
   - jeśli odległość = optymalnie i prędkość = wysoka => przepustnica = zamknięta
   - jeśli odległość = daleko i prędkość = mała => przepustnica = mocno otwarta
   - jeśli odległość = daleko i prędkość = średnia => przepustnica = lekko otwarta
   - jeśli odległość = optymalnie i prędkość = mała i masa = lekki  => przepustnica = średnio otwarta
   - jeśli odległość = optymalnie i prędkość = mała i masa = średni  => przepustnica = średnio otwarta
   - jeśli odległość = optymalnie i prędkość = mała i masa = ciężki  => przepustnica = mocno otwarta
   - jeśli odległość = daleko i prędkość = wysoka i masa = lekki  => przepustnica = lekko otwarta
   - jeśli odległość = daleko i prędkość = wysoka i masa = średni  => przepustnica = lekko otwarta
   - jeśli odległość = daleko i prędkość = wysoka i masa = cięzki  => przepustnica = zamknięta

* Przykład użycia
   - Jeśli danymi wejściowymi będą:
      * odległość = 500m i
      * prędkość aktualna = 90 km/h i
      * masa pojazdu = 1.5t
   - Przepustnica powinna być:
      * otwarta w 33%
"""
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# stworzenie tablic zawierających wartości rzeczywiste
distance = ctrl.Antecedent(np.arange(0, 501, 1), 'distance')
velocity = ctrl.Antecedent(np.arange(0, 151, 1), 'velocity')
mass = ctrl.Antecedent(np.arange(0.8, 41, 0.1), 'mass')
throttle = ctrl.Consequent(np.arange(0, 101, 1), 'throttle')

# stworzenie membership functions dla danych wejściowych i wyjściowych
distance.automf(names=['blisko', 'optymalnie', 'daleko'])
distance['blisko'] = fuzz.trapmf(distance.universe, [0,0,50,250])

velocity.automf(names=['mała', 'średnia', 'wysoka'])

mass.automf(names=['lekki', 'średni', 'ciężki'])
mass['lekki'] = fuzz.trimf(mass.universe, [0.8,0.8, 1.6])
mass['średni'] = fuzz.trimf(mass.universe, [1.0, 3.0, 4.5])
mass['ciężki'] = fuzz.trimf(mass.universe, [3.0,40.0,40.0])

throttle.automf(names=['zamknięta', 'lekko otwarta', 'średnio otwarta', 'mocno otwarta'])

throttle['zamknięta'] = fuzz.trapmf(throttle.universe, [0, 0, 5, 33])

distance.view()
velocity.view()
mass.view()
throttle.view()

# zasady

rule1 = ctrl.Rule(distance['blisko'], throttle['zamknięta'])
rule2 = ctrl.Rule(distance['optymalnie'] & velocity['średnia'], throttle['lekko otwarta'])
rule3 = ctrl.Rule(distance['optymalnie'] & velocity['wysoka'], throttle['zamknięta'])
rule4 = ctrl.Rule(distance['daleko'] & velocity['mała'], throttle['mocno otwarta'])
rule5 = ctrl.Rule(distance['daleko'] & velocity['średnia'], throttle['lekko otwarta'])
rule6 = ctrl.Rule(distance['optymalnie'] & velocity['mała'] & mass['lekki'], throttle['średnio otwarta'])
rule7 = ctrl.Rule(distance['optymalnie'] & velocity['mała'] & mass['średni'], throttle['średnio otwarta'])
rule8 = ctrl.Rule(distance['optymalnie'] & velocity['mała'] & mass['ciężki'], throttle['mocno otwarta'])
rule9 = ctrl.Rule(distance['daleko'] & velocity['wysoka'] & mass['lekki'], throttle['lekko otwarta'])
rule10 = ctrl.Rule(distance['daleko'] & velocity['wysoka'] & mass['średni'], throttle['lekko otwarta'])
rule11 = ctrl.Rule(distance['daleko'] & velocity['wysoka'] & mass['ciężki'], throttle['zamknięta'])

# stworzenie kontrolera z zestawem zasad
thrt_ctrl = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11])

# stworzenie symulacji dla kontrolera
throttle_simulation = ctrl.ControlSystemSimulation(thrt_ctrl)

# sprawdzenie symulacji
throttle_simulation.input['distance'] = 500
throttle_simulation.input['velocity'] = 90
throttle_simulation.input['mass'] = 1.5

throttle_simulation.compute()

print(throttle_simulation.output['throttle'])
throttle.view(sim=throttle_simulation)
plt.show()