import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

sns.set_context('talk')

isotopes = pd.read_csv('data/isotopes.csv', header = 0, sep = '\s+')

borehole = pd.read_csv('data/borehole-temperature.csv')

fit = linregress(isotopes['d18 O'], isotopes['dD'])
plt.plot(isotopes['d18 O'], fit.intercept + fit.slope * isotopes['d18 O'], color = 'red', linestyle = ':')
plt.scatter(isotopes['d18 O'], isotopes['dD'])
plt.xlabel('δ18O (‰)')
plt.ylabel('δD (‰)')
plt.tight_layout()
plt.savefig('results/figures/co-isotopic-plot.png', dpi = 300)
plt.show()
print(fit)

# quit()

fig, ax = plt.subplots(1, 3, figsize = (16, 12), sharey = True)
plt.gca().invert_yaxis()

ax[0].axhline(3040.1, color = 'black', linestyle = ':')
ax[0].scatter(isotopes['dD'], isotopes['Top depth'], color = 'red')
ax[0].plot(isotopes['dD'], isotopes['Top depth'], color = 'red')
ax[0].set_xlabel('δD (‰)')
ax[0].set_ylabel('Depth (m)')

ax[1].axhline(3040.1, color = 'black', linestyle = ':')
ax[1].scatter(isotopes['d18 O'], isotopes['Top depth'], color = 'dodgerblue')
ax[1].plot(isotopes['d18 O'], isotopes['Top depth'], color = 'dodgerblue')
ax[1].set_xlabel('δ18O (‰)')

idx = borehole['depth [m]'] > 3039
ax[2].plot(borehole['GISP2'][idx], borehole['depth [m]'][idx], color = 'violet')
ax[2].axhline(3040.1, color = 'black', linestyle = ':')
ax[2].set_xlabel('Temperature ($^\circ$C)')

plt.tight_layout()
plt.savefig('results/figures/water_isotopes_with_borehole.png', dpi = 300)