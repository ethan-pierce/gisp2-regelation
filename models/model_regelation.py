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

    film_coeff: float = 5e-9 # m
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

    def calc_pressure_melting_point(self) -> jax.Array:
        return self.melt_temperature - self.clapeyron_slope * self.gravity * self.ice_density * 3053

    def calc_force(self, grain_sizes: jax.Array) -> jax.Array:
        return grain_sizes * self.sediment_density * self.gravity

    def calc_film_thickness(self, temperatures: jax.Array) -> jax.Array:
        Tm = self.melt_temperature
        return (
            self.film_coeff 
            * (Tm / (Tm - temperatures))**(1 / self.film_exp)
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
    dts = np.diff(ages, prepend = 0)
    Tbs = np.array(recon['Basal temperature'])
    
    model = RegelationModel()
    Pmp = model.calc_pressure_melting_point()
    films = model.calc_film_thickness(Tbs)
    gradient = -0.023



    # How fast does a particle fall when it's cold?
    sns.set_context('talk')
    fig, ax = plt.subplots(figsize = (12, 8))

    films = model.calc_film_thickness(Tbs)
    grain_sizes = [1e-4, 1e-3, 1e-2]

    for g in grain_sizes:
        velocities = np.zeros_like(ages)

        for i in np.arange(ages.shape[0])[::-1]:
            h = films[i]
            v = model.calc_velocity_rempel(h, gradient, g)

            if Tbs[i] >= 274:
                velocities[i] = 0.0
            else:
                velocities[i] = v
    
        position = np.cumsum(velocities[::-1] * dts[::-1] * 1e3 * model.sec_per_a)

        ax.plot(ages[::-1], 0 - position, label = f'{g * 1000} mm')
        ax.plot(ages[::-1], np.where(velocities[::-1] == 0, 0 - position, np.nan), color = 'white', lw = 4)
    
    ax.set_xlabel('Age (ka)')
    ax.set_ylabel('Position (m)')
    ax.legend(loc = 'center right', title = 'Grain size')
    plt.title('Thermal regelation in cold ice')
    plt.tight_layout()
    plt.savefig('results/figures/cold_regelation_particle_positions.png', dpi = 300)
    plt.show()



    # How high can you lift a particle when it's warm?
    sns.set_context('talk')
    fig, ax = plt.subplots(figsize = (14, 8))

    warm_period = np.linspace(0, 5, 100)
    gradients = np.linspace(0, 0.1, 100)
    film_h = model.calc_film_thickness(273.145)
    print(film_h)

    grain_sizes = [1e-4, 1e-3, 1e-2]
    for g in grain_sizes:
        velocity = model.calc_velocity_rempel(film_h, gradients, g)
        position = np.outer(-velocity * 31556926 * 1e3, warm_period)
        
        im = ax.imshow(position, aspect = 'auto', cmap = 'RdBu_r', norm = TwoSlopeNorm(vmin = np.min(position), vcenter = 0, vmax = np.max(position)))
        pad = 0.03 if g != 1e-4 else 0.005
        cbar = plt.colorbar(im, ax = ax, label = 'meters above bed', pad = pad)
        cbar.ax.set_title(str(g * 1000) + ' mm')

    ax.axhline(0.021 * 1000, color = 'black', lw = 2, ls = '--')
    ax.annotate('Particles move up $\\uparrow$', (0, 19), xytext = (0, 19), textcoords = 'offset points', ha = 'left', va = 'center', fontsize = 16)
    ax.annotate('Particles move down $\\downarrow$', (0, 15), xytext = (0, 15), textcoords = 'offset points', ha = 'left', va = 'center', fontsize = 16)

    ax.axvline(20.4, color = 'purple', lw = 2, ls = ':')
    ax.annotate('Modeled warm period', (15, 50), xytext = (15, 50), textcoords = 'offset points', ha = 'left', va = 'center', fontsize = 16, color = 'purple', rotation = 90)

    ax.set_xlabel('Time (ka)')
    ax.set_ylabel('Gradient (K m$^{-1}$)')
    ax.set_xticks(np.arange(0, 100, 20))
    ax.set_xticklabels([f'{t:.0f}' for t in warm_period[::20]])
    ax.set_yticks(np.arange(0, 100, 10))
    ax.set_yticklabels([f'{g:.2f}' for g in gradients[::10]])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.title('Thermal regelation in temperate ice $\\minus$ clast size:')
    plt.savefig('results/figures/warm_regelation_particle_positions.png', dpi = 300)
    plt.show()



    m1 = eqx.tree_at(lambda t: t.film_coeff, model, 5e-9)
    m2 = eqx.tree_at(lambda t: t.film_coeff, model, 1e-9)
    m3 = eqx.tree_at(lambda t: (t.film_coeff, t.film_exp), model, (1e-10, 3/2))
    l1 = m1.calc_film_thickness(Tbs)
    l2 = m2.calc_film_thickness(Tbs)
    l3 = m3.calc_film_thickness(Tbs)

    sns.set_context('talk')
    fig, ax = plt.subplots(figsize = (12, 9))
    ax.plot(ages, l1, label = '$\lambda = 5\\times10^{-9}$ m, $\\nu = 3$', lw = 1.2)
    ax.plot(ages, l2, label = '$\lambda = 1\\times10^{-9}$ m, $\\nu = 3$', lw = 1.2)
    ax.plot(ages, l3, label = '$\lambda = 1\\times10^{-10}$ m, $\\nu = \\frac{3}{2}$', lw = 1.2)
    ax.set_xlabel('Age (ka)')
    ax.set_ylabel('Film thickness (m)')
    ax.set_yscale('log')
    ax.legend(loc = 'upper right')
    plt.tight_layout()
    plt.savefig('results/figures/film_thickness_over_time.png', dpi = 300)
    plt.show()






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
