"""Model particle regelation in a basal ice layer."""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import seaborn as sns


class RegelationModel(eqx.Module):

    film_coeff: float = 1e-9 # m
    film_exp: int = 3 # n-r van der Waals

    sec_per_a: float = 31556926 # seconds per year
    gravity: float = 9.81 # m s^-2
    melt_temperature: float = 273.15 # K
    ice_density: float = 917 # kg m^-3
    sediment_density: float = 2700 # kg m^-3
    water_density: float = 1000 # kg m^-3
    water_viscosity: float = 1e-3 # Pa s
    clapeyron_slope: float = 7.4e-8 # K Pa^-1
    latent_heat: float = 3.34e5 # J kg^-1
    thermal_conductivity: float = 3 # W m^-1 K^-1
    ice_conductivity: float = 2.1 # W m^-1 K^-1
    particle_conductivity: float = 3.6 # W m^-1 K^-1

    def calc_force(self, grain_sizes: jax.Array) -> jax.Array:
        return grain_sizes * self.sediment_density * self.gravity

    def calc_film_thickness(self, temperatures: jax.Array) -> jax.Array:
        return (
            self.film_coeff 
            * (self.melt_temperature / (self.melt_temperature - temperatures))**(1 / self.film_exp)
        )

    def calc_velocity_meyer(self, film_thickness: float, grain_radius: float) -> jax.Array:
        force = self.calc_force(grain_radius)
        leading_coeff = (
            (film_thickness**3 * self.water_density**2)
            /
            (12 * self.ice_density**2 * self.water_viscosity * grain_radius**2)
        )
        numerator = leading_coeff * force
        second_coeff = (
            (self.ice_density**2 * self.latent_heat**2 * grain_radius)
            /
            (2 * self.melt_temperature * self.thermal_conductivity)
        )
        denominator = (1 - leading_coeff * second_coeff)
        return numerator / denominator

    def calc_velocity_rempel(self, film_thickness: float, gradient: float, grain_radius: float) -> jax.Array:
        force = self.calc_force(grain_radius)
        a = (
            (self.water_density * film_thickness**3) 
            / 
            (6 * self.ice_density * self.water_viscosity * grain_radius**2)
        )
        b = self.sediment_density * self.gravity * grain_radius
        c = self.water_density * self.latent_heat / self.melt_temperature
        d = grain_radius**3 * self.ice_density * self.latent_heat / (self.particle_conductivity + 2 * self.ice_conductivity)
        e = grain_radius**3 * (self.ice_density - self.sediment_density) * gradient / (self.particle_conductivity + 2 * self.ice_conductivity)
        g = gradient * grain_radius

        return (a*b + a*c*e - a*c*g) / (1 + a*c*d)

    def calc_velocity_worster(self, T: jax.Array) -> jax.Array:
        film_thickness = self.calc_film_thickness(T)
        leading_coeff = (
            film_thickness**3 / (6 * self.water_viscosity * self.grain_radius)
        )
        temp_diff = self.melt_temperature / (self.melt_temperature - T)
        force = 2 * self.force
        return leading_coeff * temp_diff * -force



if __name__ == '__main__':

    recon = pd.read_csv('results/basal_temperature_reconstruction.csv')

    ages = np.array(recon['Age']) * 1e-3
    Ts = np.array(recon['Basal temperature'])
    
    model = RegelationModel()    


    # m1 = eqx.tree_at(lambda t: t.film_coeff, model, 5e-9)
    # m2 = eqx.tree_at(lambda t: t.film_coeff, model, 1e-9)
    # m3 = eqx.tree_at(lambda t: (t.film_coeff, t.film_exp), model, (1e-10, 3/2))
    # l1 = m1.calc_film_thickness(Ts)
    # l2 = m2.calc_film_thickness(Ts)
    # l3 = m3.calc_film_thickness(Ts)

    # sns.set_context('talk')
    # fig, ax = plt.subplots(figsize = (12, 8))
    # ax.plot(ages, l1, label = '$\lambda = 5e-9$, $\\nu = 3$', lw = 1.2)
    # ax.plot(ages, l2, label = '$\lambda = 1e-9$, $\\nu = 3$', lw = 1.2)
    # ax.plot(ages, l3, label = '$\lambda = 1e-10$, $\\nu = \\frac{3}{2}$', lw = 1.2)
    # ax.set_xlabel('Age (ka)')
    # ax.set_ylabel('Film thickness (m)')
    # ax.set_yscale('log')
    # ax.legend(loc = 'center right')
    # plt.tight_layout()
    # plt.savefig('results/figures/film_thickness_over_time.png')
    # plt.show()






    # model = eqx.tree_at(lambda t: t.film_coeff, model, 1e-8)

    # sns.set_context('talk')
    # time_labels = np.linspace(0, 125, 1000)
    # times = time_labels * 1e3 * model.sec_per_a
    # gradients = np.linspace(-0.05, 0.05, 1000)

    # h1 = model.calc_film_thickness(264)
    # h2 = model.calc_film_thickness(273)

    # velocities = -model.calc_velocity_rempel(h1, gradients, 1e-2)
    # positions = np.outer(velocities, times)

    # warmvels = -model.calc_velocity_rempel(h2, gradients, 1e-2)
    # warmpos = np.outer(warmvels, times)

    # fig, ax = plt.subplots(1, 2, figsize = (16, 8))
    # im = ax[0].imshow(positions, aspect = 'auto', cmap = 'RdBu', norm = TwoSlopeNorm(vmin = np.min(positions), vcenter = 0, vmax = np.max(positions)))
    # ax[0].set_xlabel('Time (ka)')
    # ax[0].set_ylabel('Gradient (K m${^-1}$)')
    # ax[0].set_xticks(np.arange(0, 1000, 200))
    # ax[0].set_xticklabels([f'{t:.0f}' for t in time_labels[::200]])
    # ax[0].set_yticks(np.arange(0, 1000, 100))
    # ax[0].set_yticklabels([f'{g:.2f}' for g in gradients[::100]])
    # plt.colorbar(im, ax = ax[0], format=lambda x, _: f"{x:.1e}", label = 'meters above bed')
    # ax[0].set_title('Particle position | 264 K | 1 cm grain')

    # im = ax[1].imshow(warmpos, aspect = 'auto', cmap = 'RdBu', norm = TwoSlopeNorm(vmin = np.min(warmpos), vcenter = 0, vmax = np.max(warmpos)))
    # ax[1].set_xlabel('Time (ka)')
    # ax[1].set_ylabel('Gradient (K m$^{-1}$)')
    # ax[1].set_xticks(np.arange(0, 1000, 200))
    # ax[1].set_xticklabels([f'{t:.0f}' for t in time_labels[::200]])
    # ax[1].set_yticks(np.arange(0, 1000, 100))
    # ax[1].set_yticklabels([f'{g:.2f}' for g in gradients[::100]])
    # plt.colorbar(im, ax = ax[1], format=lambda x, _: f"{x:.1e}", label = 'meters above bed')
    # ax[1].set_title('Particle position | 273 K | 1 cm grain')

    # plt.tight_layout()
    # plt.savefig('results/figures/regelation_particle_positions.png')
    # plt.show()






#################
# Plotting code #
#################

# # Grain size vs. overburden force
# plt.plot(grain_sizes, model.force)
# plt.xscale('log')
# plt.xlabel('Grain radius (m)')
# plt.ylabel('Overburden force (Pa)')
# plt.savefig('results/figures/overburden_force.png')
# plt.close('all')
