import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

def calc_film_thickness(T, L, n):
    Tm = 273.15
    return L * (Tm / (Tm - T))**(1/n)

Ts = np.arange(-30, 0, 0.1)
Ts += 273.15

plt.plot(Ts, calc_film_thickness(Ts, 1e-9, 3/2), label = '$\\nu=3/2$')
plt.plot(Ts, calc_film_thickness(Ts, 1e-9, 3), label = '$\\nu=3$')
plt.plot(Ts, calc_film_thickness(Ts, 1e-8, 3), label = '$\\lambda=1e-8$')
plt.plot(Ts, calc_film_thickness(Ts, 1e-10, 3), label = '$\\lambda=1e-10$')
plt.plot(Ts, calc_film_thickness(Ts, 7e-11, 1.3), label = 'Gilpin')
plt.xlabel('Temperature (K)')
plt.ylabel('Film thickness (m)')
plt.legend()
plt.yscale('log')
plt.show()