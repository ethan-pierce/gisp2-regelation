"""Model basal temperature using the Kindler (2014) reconstruction."""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import matplotlib.pyplot as plt
import seaborn as sns


class BasalTemperatureModel(eqx.Module):
    """Input units are meters/years/Celsius."""

    time: jax.Array = eqx.field(converter = jnp.asarray)
    zs: jax.Array = eqx.field(converter = jnp.asarray)
    temperature: jax.Array = eqx.field(converter = jnp.asarray, init = False)
    surface_temperature: jax.Array = eqx.field(converter = jnp.asarray)
    accumulation: jax.Array = eqx.field(converter = jnp.asarray)
    geothermal_flux: jax.Array = eqx.field(converter = jnp.asarray)
    surface_slope: float = eqx.field(converter = jnp.asarray)
    
    sec_per_a: float = 31556926 # seconds per year
    Tm: float = 273.15 # melting temperature
    thermal_conductivity: float = 2.1 # W m^-1 K^-1
    heat_capacity: float = 2090 # J kg^-1 K^-1
    density: float = 917 # kg m^-3
    ice_fluidity: float = 2.4e-24 # Pa^-3 s^-1
    gravity: float = 9.81 # m s^-2
    thickness: float = eqx.field(converter = jnp.asarray, init = False)
    thermal_diffusivity: float = eqx.field(converter = jnp.asarray, init = False)
    basal_heat_flux: float = eqx.field(converter = jnp.asarray, init = False)
    strain_dissipation: jax.Array = eqx.field(converter = jnp.asarray, init = False)

    def __post_init__(self):
        self.thickness = jnp.max(self.zs)
        self.thermal_diffusivity = self.thermal_conductivity / (self.density * self.heat_capacity)
        self.basal_heat_flux = -self.geothermal_flux / self.thermal_conductivity
        self.strain_dissipation = self.calc_strain()
        self.temperature = self.set_initial_conditions()

    def set_initial_conditions(self) -> jax.Array:
        initial_acc_per_sec = self.accumulation[0] / self.sec_per_a
        initial_temp = self.Tm + self.surface_temperature[0]
        c1 = jnp.sqrt(2 * self.thermal_diffusivity * self.thickness / initial_acc_per_sec)

        return (
            initial_temp
            + (jnp.sqrt(jnp.pi) / 2) 
            * c1 
            * self.basal_heat_flux
            * (
                jax.scipy.special.erf(self.zs / c1)
                - jax.scipy.special.erf(self.thickness / c1)
            )
        )

    def calc_steady_state(self, t: int) -> jax.Array:
        T = self.Tm + self.surface_temperature[t]
        b = self.accumulation[t] / self.sec_per_a
        c = jnp.sqrt(2 * self.thermal_diffusivity * self.thickness / b)

        return (
            T
            + (jnp.sqrt(jnp.pi) / 2) 
            * c 
            * self.basal_heat_flux
            * (
                jax.scipy.special.erf(self.zs / c)
                - jax.scipy.special.erf(self.thickness / c)
            )
        )

    def calc_strain(self) -> jax.Array:
        return (
            2 * self.ice_fluidity 
            * (self.density * self.gravity * self.surface_slope)**4 
            * (self.thickness - self.zs)**4
        )

    def calc_dT(self, temperature: jax.Array, args: tuple) -> jax.Array:
        surface_temperature, accumulation = args

        temperature = temperature.at[-1].set(self.Tm + surface_temperature)
        gradient = jnp.gradient(temperature, self.zs)
        gradient = gradient.at[0].set(self.basal_heat_flux)
        divergence = jnp.gradient(gradient, self.zs)

        accumulation = accumulation / self.sec_per_a
        z_normalized = self.zs / self.thickness
        advection = -accumulation * z_normalized * gradient

        diffusion = self.thermal_diffusivity * divergence

        return diffusion - advection

    @eqx.filter_jit
    def run_one_step(self, t: int):
        args = (self.surface_temperature[t], self.accumulation[t])
        dt = (self.time[t - 1] - self.time[t]) * self.sec_per_a
        residual = lambda T, _: (
            self.temperature - T + dt * self.calc_dT(T, args)
        )

        solver = optx.Newton(rtol = 1e-3, atol = 1e-3)
        solution = optx.root_find(
            fn = residual,
            solver = solver,
            y0 = self.temperature,
            args = None
        )

        result = jnp.where(
            solution.value > self.Tm,
            self.Tm,
            solution.value
        )

        result = result.at[-1].set(self.Tm + self.surface_temperature[t])

        return eqx.tree_at(
            lambda t: t.temperature,
            self,
            result
        )



if __name__ == '__main__':

    # sns.set_style('darkgrid')
    # sns.set(font_scale = 1.2)
    # borehole = pd.read_csv('data/borehole-temperature.csv')

    # idx = borehole['depth [m]'] > 3039

    # fig, ax = plt.subplots(figsize = (5, 12))
    # plt.plot(borehole['GISP2'][idx], borehole['depth [m]'][idx])
    # plt.xlabel('Temperature ($^\circ$C)')
    # plt.ylabel('Depth (m)')

    # plt.gca().invert_yaxis()
    # plt.tight_layout()
    # plt.savefig('results/figures/lowest_borehole_temperature_profile.png')
    # quit()

    kindler = pd.read_csv(
        'data/Kindler2014-ngrip.csv', skiprows = 12, sep = '\s+', usecols = [0, 1, 2, 3, 4], nrows = 5663-12,
        names = ['Depth', 'Age ss09sea06bm', 'Age', 'Accumulation', 'Temperature']
    )
    lapse_rate = 6 / 1000
    elevation_diff = 3232 - 2917
    kindler['Temperature'] = kindler['Temperature'] - lapse_rate * elevation_diff

    alley_temp = pd.read_csv('data/alley2000-gisp.txt', skiprows = 74, nrows = 1632, header = 0, sep = '\s+')
    alley_temp = alley_temp.dropna(axis = 1)
    alley_acc = pd.read_csv('data/alley2000-gisp.txt', skiprows = 1716, nrows = 3414-1717+1, header = 0, sep = '\s+')
    alley_int = np.interp(alley_temp['Age'], alley_acc['Age'], alley_acc['Accumulation'])
    alley = alley_temp.copy()
    alley['Age'] = alley['Age'] * 1e3
    alley['Accumulation'] = alley_int

    idx = np.argmin(np.abs(kindler['Age'][0] - alley['Age'])) - 1

    recon = pd.DataFrame(columns = ['Age', 'Temperature', 'Accumulation'])
    recon['Age'] = np.hstack([alley['Age'][:idx], kindler['Age']])
    recon['Temperature'] = np.hstack([alley['Temperature'][:idx], kindler['Temperature']])
    recon['Accumulation'] = np.hstack([alley['Accumulation'][:idx], kindler['Accumulation']])

    dz = 1
    pts = np.arange(0, 3053 + dz, dz)

    model = BasalTemperatureModel(
        np.flip(recon['Age']),
        pts,
        np.flip(recon['Temperature']),
        np.flip(recon['Accumulation']),
        0.05,
        0.002
    )

    # Q = 6e4
    # R = 8.314
    # A0 = 3.5e-25
    # C0 = 7.4e-8
    # Tcorr = (1 / (model.temperature + C0 * model.thickness * 917 * 9.81)) - (1 / (263 + C0 * model.thickness * 917 * 9.81))
    # A = A0 * np.exp(-(Q/ R) * Tcorr)
    # strain = A * (model.density * model.gravity * model.surface_slope)**4 * (model.thickness - model.zs)**4
    # plt.plot(strain / (model.density * model.heat_capacity), model.thickness - model.zs)
    # plt.gca().invert_yaxis()
    # plt.show()
    # quit()

    ts = jnp.arange(recon.shape[0] - 1) + 1
    Ts = np.empty((ts.shape[0] + 1, model.zs.shape[0]), dtype = np.float32)
    Ts[0] = model.temperature

    fig, ax = plt.subplots(figsize = (12, 8))

    for t in ts:
        # model = model.run_one_step(t)
        # Ts[t] = model.temperature
        
        Ts[t] = model.calc_steady_state(t)

        if t % 100 == 0:
            print(f'Iteration {t} complete.')
            
            ax.plot(Ts[t], model.thickness - model.zs, alpha = 0.5, color = 'firebrick')

    np.savetxt('results/steady_temperature_profiles.csv', Ts)

    ax.invert_yaxis()
    plt.xlabel('Temperature (K)')
    plt.ylabel('Depth (m)')
    plt.show()

    Tb = [Ts[i][0] for i in range(len(Ts))]
    Tb = np.flip(np.array(Tb))
    recon['Basal temperature'] = Tb
    recon.to_csv('results/basal_temperature_reconstruction.csv', index = False)

    plt.plot(model.time * 1e-3, Tb)
    plt.xlabel('Age (k.a.)')
    plt.ylabel('Basal temperature (K)')
    plt.show()

    np.savetxt('results/steady_basal_temperature.csv', Tb)
    plt.plot(recon['Age'] * 1e-3, recon['Temperature'], label = 'Surface temperature', color = 'dodgerblue', lw = 0.8)
    plt.plot(recon['Age'] * 1e-3, np.array(Tb) - 273.15, label = 'Basal temperature', color = 'orangered', lw = 0.8)
    plt.legend()
    plt.xlabel('Age (k.a.)')
    plt.ylabel('Temperature (°C)')
    plt.show()

    # Ts = np.array(Ts)
    # gradT = np.gradient(Ts, dz, axis = 1, edge_order = 2)
    # basal_dT = np.mean(gradT[:, 0:10], axis = 1)
    # plt.plot(recon['Age'] * 1e-3, basal_dT, label = 'Basal temperature gradient', color = 'firebrick', lw = 0.8)
    # plt.xlabel('Age (k.a.)')
    # plt.ylabel('Basal temperature gradient (K/m)')
    # plt.show()
