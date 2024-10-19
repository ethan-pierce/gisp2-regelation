import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from adjustText import adjust_text

sns.set_context('talk')

sed_conc = pd.DataFrame(columns = ['Depth', 'Concentration']) # from Gow et al., 1996
sed_conc['Depth'] = [3042.85, 3044.0, 3046.0, 3048.0, 3051.06, 3053.0]
sed_conc['Concentration'] = [0.65, 0.24, 0.43, 0.35, 0.38, 0.30]

isotopes = pd.read_csv('data/isotopes.csv', header = 0, sep = '\s+')

# fig, ax = plt.subplots(1, 3, figsize = (16, 12), sharey = True)
# plt.gca().invert_yaxis()

# ax[0].axhline(3040.1, color = 'black', linestyle = ':')
# ax[0].scatter(isotopes['dD'], isotopes['Top depth'], color = 'firebrick')
# ax[0].plot(isotopes['dD'], isotopes['Top depth'], color = 'firebrick')
# ax[0].set_xlabel('δD (‰)')
# ax[0].set_ylabel('Depth (m)')

# ax[1].axhline(3040.1, color = 'black', linestyle = ':')
# ax[1].scatter(isotopes['d18 O'], isotopes['Top depth'], color = 'dodgerblue')
# ax[1].plot(isotopes['d18 O'], isotopes['Top depth'], color = 'dodgerblue')
# ax[1].set_xlabel('δ$^{18}$O (‰)')

# ax[2].scatter(sed_conc['Concentration'], sed_conc['Depth'], color = 'forestgreen')
# ax[2].plot(sed_conc['Concentration'], sed_conc['Depth'], color = 'forestgreen', linestyle = '--')
# ax[2].axhline(3040.1, color = 'black', linestyle = ':')
# ax[2].set_xlabel('Sediment concentration (% mass)')
# ax[2].set_xlim(0, 1)

# plt.tight_layout()
# plt.savefig('results/figures/water_isotopes_with_sed_conc.png', dpi = 300)
# quit()

fig, ax = plt.subplots(figsize = (10, 7))

fit = linregress(isotopes['d18 O'], isotopes['dD'])
plt.plot(isotopes['d18 O'], fit.intercept + fit.slope * isotopes['d18 O'], color = 'red', linestyle = ':', label = 'Meteoric Water Line')
plt.scatter(isotopes['d18 O'], isotopes['dD'], color = 'black')

xs = np.linspace(np.min(isotopes['d18 O']), np.max(isotopes['d18 O']), 100)
ys = -47.75 + 6.5 * xs
plt.plot(xs, ys, color = 'cyan', linestyle = '--', label = 'Freezing slope')

texts = [plt.text(isotopes['d18 O'][i] + 0.25, isotopes['dD'][i] - 6, isotopes['Top depth'][i]) for i in range(len(isotopes))]
adjust_text(texts)

plt.xlabel('δ18O (‰)')
plt.ylabel('δD (‰)')

plt.legend()

plt.tight_layout()
plt.savefig('results/figures/co-isotopic-plot.png', dpi = 300)
plt.show()
print(fit)
quit()