import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import cmcrameri.cm as cm

sns.set(style = 'whitegrid', font_scale = 1.1)

def calc_film_thickness(T, L, n):
    Tm = 273.15
    return L * (Tm / (Tm - T))**(1/n)

Ts = np.arange(-15, 0, 0.1)
Ts += 273.15

fig, ax = plt.subplots(figsize = (12, 8))
cmap = cm.roma

plt.plot(Ts, calc_film_thickness(Ts, 1e-8, 3), label = '$\\lambda=10^{-8},\\;\\nu=3$', color = 'firebrick')
plt.plot(Ts, calc_film_thickness(Ts, 1e-9, 3/2), label = '$\\lambda=10^{-9},\\;\\nu=\\frac{3}{2}$', color = 'orangered')
plt.plot(Ts, calc_film_thickness(Ts, 1e-9, 3), label = '$\\lambda=10^{-9},\\;\\nu=3$', color = 'orange')
plt.plot(Ts, calc_film_thickness(Ts, 5.5e-10, 3), label = 'This study: $\\lambda=2\\AA,\\;\\nu=3$', color = 'mediumorchid')
plt.plot(Ts, calc_film_thickness(Ts, 7e-11, 1.3), label = 'Gilpin: $\\lambda=7\\times10^{-11},\\;\\nu=1.3$', color = 'cornflowerblue')

plt.xlabel('Temperature (K)')
plt.ylabel('Film thickness (m)')
plt.legend(loc = 'upper left')
plt.yscale('log')
plt.savefig('results/figures/film_thickness.png', dpi = 300)
