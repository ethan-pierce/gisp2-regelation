"""Models the surface temperature history at Summit."""

import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

alley_temp = pd.read_csv('data/alley2000-gisp.txt', skiprows = 74, nrows = 1632, header = 0, sep = '\s+')
alley_temp = alley_temp.dropna(axis = 1)
alley_acc = pd.read_csv('data/alley2000-gisp.txt', skiprows = 1716, nrows = 3414-1717+1, header = 0, sep = '\s+')

markle_all = loadmat('data/markle.mat')
print(markle_all.keys())
markle_grip = pd.DataFrame(markle_all['GRIP_T_surf'], columns = ['Temperature'])
markle_grip['Age'] = markle_all['GRIP_age']
markle_grip = markle_grip[markle_grip['Age'] <= 120]

kindler = pd.read_csv(
    'data/Kindler2014-ngrip.csv', skiprows = 12, sep = '\s+', usecols = [0, 1, 2, 3, 4], nrows = 5663-12,
    names = ['Depth', 'Age ss09sea06bm', 'Age', 'Accumulation', 'Temperature']
)
kindler['Age'] = kindler['Age'] * 1e-3
lapse_rate = 6 / 1000
elevation_diff = 3232 - 2917
kindler['Temperature'] = kindler['Temperature'] - lapse_rate * elevation_diff

sns.set_style('whitegrid')

# fig, ax = plt.subplots(figsize = (12, 4))
# plt.plot(markle_grip['Age'], markle_grip['Temperature'], label = 'Markle (GRIP)', color = 'dodgerblue', lw = 0.8)
# plt.plot(kindler['Age'], kindler['Temperature'], label = 'Kindler (NGRIP)', color = 'orangered', lw = 0.8)
# plt.plot(alley_temp['Age'], alley_temp['Temperature'], label = 'Alley (GISP-2)', color = 'darkblue', lw = 0.8)  
# plt.xlabel('Age (k.a.)')
# plt.ylabel('Surface temperature (°C)')
# plt.legend()
# plt.show()



H = 3053
dz = 0.1
zs = np.arange(0, H + dz, dz)
zs = np.flip(zs)
rho = 917
k = 2.1
c = 2090
G = 0.05 / -k
b0 = 0.2 / 31556926
kappa = k / (rho * c)
c1 = np.sqrt(2 * kappa * H / b0)
Ts0 = 273 - 30

T0 = Ts0 + (np.sqrt(np.pi) / 2) * c1 * G * (erf((np.max(zs) - zs) / c1) - erf(H / c1))





fig, ax = plt.subplots(figsize = (12, 8))
plt.plot(T0, zs)
plt.xlabel('Temperature (°K)')
plt.ylabel('Depth (m)')
ax.invert_yaxis()
plt.show()
