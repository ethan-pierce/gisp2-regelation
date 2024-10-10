"""Model entrainment by pressurized vein flow."""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

sns.set_context('talk')


class VeinFlowModel(eqx.Module):

    zs: jax.Array = eqx.field(converter = jnp.asarray)
    ds: jax.Array = eqx.field(converter = jnp.asarray)

    gravity: float = 9.81 # m s^-2
    ice_density: float = 917 # kg m^-3
    water_density: float = 1000 # kg m^-3
    sediment_density: float = 2700 # kg m^-3
    water_viscosity: float = 1.8e-3 # Pa s
    ice_conductivity: float = 2.1 # W m^-1 K^-1
    latent_heat: float = 3.34e5 # J kg^-1
    melt_temperature: float = 273.15 # K
    surface_energy: float = 0.03 # J m^-2
    liquidus_slope: float = 1.9e-3 # K Pa^-1
    permeability: float = 1.75e-11 # m^2 (ranges from 1e-12 to 1e-15)
    alpha: float = 0.1 # geometric coefficient
    diffusivity: float = 4e-10 # m^2 s^-1

    Rv0: float = 140e-6 # m
    d0: float = 1e-3 # m
    k0 = 2e-12 # m^2
    phi0: float = eqx.field(converter = jnp.asarray, init = False)

    def __post_init__(self):
        self.phi0 = 3 * self.alpha * (self.Rv0 / self.d0)**2

    def calc_liquid_fraction(self, T: jax.Array, c: jax.Array) -> jax.Array:
        leading_coeff = 3 * self.alpha * self.surface_energy**2 / self.ds**2
        first_term = (self.water_density * self.latent_heat / self.melt_temperature) * (self.melt_temperature - T - self.liquidus_slope * c)
        second_term = (self.water_density - self.ice_density) * self.gravity * self.zs
        return leading_coeff * (first_term - second_term)**(-2)

    def calc_permeability(self, phi: jax.Array) -> jax.Array:
        return self.k0 * (phi / self.phi0)**2
        # return 1.7e-11 * (self.ds * 1e3)**(-3.4)

    def calc_flux(self, k: jax.Array, T: jax.Array, c: jax.Array) -> jax.Array:
        leading_coeff = self.water_density * self.latent_heat * k / (self.water_viscosity * self.melt_temperature)
        gradT = jnp.gradient(T, self.zs)
        gradC = self.liquidus_slope * jnp.gradient(c, self.zs)
        return -leading_coeff * (gradT + gradC)

    def calc_mobile_grain_size(self, q: jax.Array, phi: jax.Array) -> float:
        return jnp.sqrt(
            (9 * self.water_viscosity * jnp.abs(q))
            /
            (2 * phi * (self.sediment_density - self.water_density) * self.gravity)
        )

    def calc_dphi_dt(self, T: jax.Array) -> jax.Array:
        gradT = jnp.gradient(T, self.zs)
        divT = jnp.gradient(gradT, self.zs)
        return (
            (self.ice_conductivity / (self.ice_density * self.latent_heat)) * divT
        )

    def calc_dc_dt(self, c: jax.Array, phi: jax.Array, dphi: jax.Array, q: jax.Array) -> jax.Array:
        gradC = jnp.gradient(c, self.zs)
        divC = jnp.gradient(phi * gradC, self.zs)
        diffusion = self.diffusivity / phi * divC
        advection = q * gradC
        source = c / phi * dphi
        return diffusion - advection - source

    def run_one_step(self, dt: float, Ts: jax.Array, Cs: jax.Array) -> jax.Array:
        phi = self.calc_liquid_fraction(Ts, Cs)
        k = self.calc_permeability(phi)
        q = self.calc_flux(k, Ts, Cs)

        dphi = self.calc_dphi_dt(Ts)
        dC = self.calc_dc_dt(Cs, phi, dphi, q)

        return Cs + dt * dC

    def update(self, Ts: jax.Array, Cs: jax.Array) -> jax.Array:
        phi = self.calc_liquid_fraction(Ts, Cs)
        k = self.calc_permeability(phi)
        q = self.calc_flux(k, Ts, Cs)
        dphi = self.calc_dphi_dt(Ts)
        dC = self.calc_dc_dt(Cs, phi, dphi, q)
        return dC

    def run_one_backward(self, dt: float, Ts: jax.Array, Cs: jax.Array) -> jax.Array:
        residual = lambda c, _: c - dt * self.update(Ts, c) - Cs
        solver = optx.Newton(atol = 1e-6, rtol = 1e-6)
        solution = optx.root_find(
            fn = residual,
            solver = solver,
            y0 = Cs,
            args = None
        )
        return solution.value


if __name__ == '__main__':
    sns.set_context('talk')

    df = pd.read_csv('results/basal_temperature_reconstruction.csv')
    Tb = jnp.array(df['Basal temperature'])
    age = jnp.array(df['Age']) * 1e-3
    
    zs = np.linspace(0, 15, 1000)
    ds = np.where(zs < 13.3, 1e-3, 1e-2)
    cs = jnp.zeros_like(zs)
    G = -0.023
    Ts = 264 + G * zs
    
    model = VeinFlowModel(zs, ds)
    phi = model.calc_liquid_fraction(Ts, cs)
    k = model.calc_permeability(phi)
    q = model.calc_flux(k, Ts, cs)
    R = model.calc_mobile_grain_size(q, phi)





    # temperatures = np.linspace(260, 272.9, 100)
    # Rs = []
    # for Tb in temperatures:
    #     Ts = Tb + G * zs
    #     phi = model.calc_liquid_fraction(Ts, cs)
    #     k = model.calc_permeability(phi)
    #     q = model.calc_flux(k, Ts, cs)
    #     R = model.calc_mobile_grain_size(q, phi)
    #     Rs.append(np.max(R))

    # fig, ax = plt.subplots(figsize = (8, 6))
    # plt.plot(temperatures, Rs)
    # plt.xlabel('Basal temperature (K)')
    # plt.ylabel('Mobile grain size (m)')
    # plt.tight_layout()
    # plt.savefig('results/figures/mobile_grain_size.png', dpi = 300)
    # plt.show()





    temperature = Tb + G * zs[:, None]
    phis = np.zeros_like(temperature)
    qs = np.zeros_like(temperature)
    Rs = np.zeros_like(temperature)

    for i in range(temperature.shape[1]):
        T = temperature[:, i]
        phi = model.calc_liquid_fraction(T, cs)
        phis[:, i] = phi

        k = model.calc_permeability(phi)
        q = model.calc_flux(k, T, cs)
        qs[:, i] = q

        R = model.calc_mobile_grain_size(q, phi)
        Rs[:, i] = R

    # plt.imshow(phis, aspect = 'auto', norm = LogNorm())
    # plt.gca().invert_yaxis()
    # plt.colorbar()
    # plt.show()

    fig, ax = plt.subplots(figsize = (18, 8))
    im = ax.imshow(Rs, aspect = 'auto', norm = LogNorm())
    plt.colorbar(im, label = 'Grain size (m)')
    ax.invert_yaxis()

    ax.set_xticks(np.linspace(0, len(age), 10))
    ax.set_xticklabels(np.round(np.linspace(0, np.max(age), 10), 2))
    ax.set_xlabel('Age (ka)')

    ax.set_yticks(np.linspace(0, len(zs), 10))
    ax.set_yticklabels(np.round(np.linspace(0, np.max(zs), 10), 2))
    ax.set_ylabel('Height above bed (m)')

    plt.title('Mobile grain size (m)')
    plt.tight_layout()
    # plt.savefig('results/figures/vein_flux.png', dpi = 300)
    # plt.savefig('results/figures/grain_transport.png', dpi = 300)
    plt.show()




    # sns.set_context('talk')
    # fig, ax = plt.subplots(2, 1, figsize = (18, 6), sharex = True)

    # ax[0].plot(age, phi_b, label = '$\phi|_{z=0}$')
    # ax[0].plot(age, phi_t, label = '$\phi|_{z=15}$')
    # ax[0].set_yscale('log')
    # ax[0].set_ylabel('Liquid fraction')
    # ax[0].legend()

    # ax[1].plot(age, q_b, label = '$q|_{z=0}$')
    # ax[1].plot(age, q_t, label = '$q|_{z=15}$')
    # ax[1].set_yscale('log')
    # ax[1].set_xlabel('Age (ka)')
    # ax[1].set_ylabel('Flux (m$^3$ s$^{-1}$)')
    # ax[1].legend()

    # plt.tight_layout()
    # plt.show()





    ## this doesn't work yet, check stablity criterion
    # nt = 10000
    # results = np.zeros((nt, len(zs)))
    # for i in range(nt):
    #     cs = cs.at[0].set(10)
    #     cs = jnp.where(zs >= 13.3, 0.0, cs)
    #     cs = model.run_one_step(100, Ts, cs)
    #     results[i] = cs

    #     if i % (nt / 20) == 0:
    #         print(f'{i} / {nt}')

    # plt.plot(results[0], zs)
    # for i in range(10):
    #     idx = nt // (i + 2)
    #     plt.plot(results[idx], zs)
    # plt.plot(results[-1], zs)
    # plt.show()



    # sns.set_context('talk')
    # fig, ax = plt.subplots(1, 4, figsize = (18, 8), sharey = True)

    # ax[0].plot(Ts, model.zs)
    # ax[0].set_xlabel('Temperature (K)')
    # ax[0].set_ylabel('Depth (m)')

    # ax[1].plot(model.ds, zs)
    # ax[1].set_xlabel('Grain size (m)')

    # ax[2].plot(phi, zs)
    # ax[2].set_xlabel('Liquid fraction')

    # ax[3].plot(q, zs)
    # ax[3].set_xlabel('Flux (m$^3$ s$^{-1}$)')

    # plt.tight_layout()
    # plt.savefig('results/figures/vein_flow_properties.png', dpi = 300)
    # plt.show()
