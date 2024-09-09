"""Model vertical temperature and enthalpy distribution.

Uses the enthalpy-gradient model for englacial drainage introduced by 
Hewitt and Schoof (2017). 
"""

import numpy as np
import matplotlib.pyplot as plt

class EnthalpyGradientModel:

    def __init__(self, z):
        self.z = z

        self.params = {
            'gravity': 9.81,
            'ice_density': 917,
            'water_density': 1000,
            'clapeyron_slope': 7.5e-8,
            'reference_temperature': 273.15,
            'heat_capacity': 2009,
            'thermal_conductivity': 2.1,
            'latent_heat': 3.34e5,
            'permeability': 1e-12,
            'water_viscoisty': 1.8e-3,
            'diffusivity': 1.1e-8
        }

        self.enthalpy = np.zeros_like(self.z)
        self.temperature = np.zeros_like(self.z)
        self.porosity = np.zeros_like(self.z)

        self.melt_temperature = self.params['reference_temperature'] - (
            self.params['clapeyron_slope'] 
            * self.params['gravity'] 
            * self.params['ice_density']
            * (np.max(self.z) - self.z)
        )

    def mean(a):
        return (a[1:] + a[:-1]) / 2

    def partition_domain(self):
        """Return zero for cold ice and one for temperate ice."""
        temp_corrected = self.melt_temperature - self.params['reference_temperature']
        threshold = self.params['ice_density'] * self.params['heat_capacity'] * temp_corrected
        return np.where(self.enthalpy <  threshold, 0, 1)

    def calc_advection(self, mass_balance: float):
        velocity = -mass_balance * self.z / np.max(self.z)
        gradient = np.gradient(self.enthalpy, self.z)
        return velocity * gradient

    def calc_diffusion(self):
        partition = self.partition_domain()
        temperature = np.where(
            partition == 0,
            self.enthalpy / (self.params['ice_density'] * self.params['heat_capacity']),
            self.melt_temperature
        )
        gradient = np.diff(temperature, append = -10) / np.diff(self.z, append = np.diff(self.z)[-1])
        geotherm = 0.04 / (self.params['ice_density'] * self.params['heat_capacity'])
        div = np.diff(-self.params['thermal_conductivity'] * gradient, prepend = geotherm) / np.diff(self.z, prepend = np.diff(self.z)[0])
        return div

    def calc_drainage(self):
        drainage = (
            -self.params['permeability'] 
            * self.porosity**2 
            * (self.params['water_density'] - self.params['ice_density']) 
            * self.params['gravity']
            / self.params['water_viscoisty']
        )

        partition = self.partition_domain()
        porosity = np.where(
            partition == 0,
            0,
            self.enthalpy / (self.params['water_density'] * self.params['latent_heat'])
        )
        gradient = np.diff(porosity, prepend = 0) / np.diff(self.z, append = np.diff(self.z)[0])
        return -self.params['diffusivity'] * gradient + drainage

    def calc_dhdt(self, mass_balance: float):
        advection = self.calc_advection(mass_balance)
        diffusion = self.calc_diffusion()
        drainage = self.calc_drainage()
        return -advection + diffusion - drainage

    def run_one_step(self, dt: float, mass_balance: float):
        dhdt = self.calc_dhdt(mass_balance)
        self.enthalpy += dhdt * dt
        self.temperature = (
            self.params['reference_temperature'] 
            + self.enthalpy / (self.params['ice_density'] * self.params['heat_capacity'])
        )
        self.porosity = np.where(
            self.partition_domain() == 0,
            0,
            (
                self.enthalpy 
                - (self.params['ice_density'] * self.params['heat_capacity']) 
                * (self.temperature - self.params['reference_temperature'])
            ) / (self.params['water_density'] * self.params['latent_heat'])
        )

if __name__ == '__main__':
    z = np.arange(0, 3053.1, 0.1)
    model = EnthalpyGradientModel(z)
    model.temperature = model.params['reference_temperature']

    plt.plot(model.calc_dhdt(-0.02), z)
    plt.show()