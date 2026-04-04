import numpy as np


def schmidt_number(sst):
    return 2116.8 - 136.25 * sst + 4.7353 * sst**2 - 0.092307 * sst**3 + 0.0007555 * sst**4


def co2_solubility(sst, salinity):
    temperature_kelvin = sst + 273.15
    ln_k0 = (
        -58.0931
        + 90.5069 * (100 / temperature_kelvin)
        + 22.2940 * np.log(temperature_kelvin / 100)
        + salinity
        * (
            0.027766
            - 0.025888 * (temperature_kelvin / 100)
            + 0.0050578 * (temperature_kelvin / 100) ** 2
        )
    )
    return np.exp(ln_k0)


def compute_co2_flux(pco2_ocean, pco2_atm, sst, wind_speed, salinity):
    sc = schmidt_number(sst)
    k = 0.251 * wind_speed**2 * (sc / 660) ** (-0.5)
    k = k * 24 * 365 / 100
    k0 = co2_solubility(sst, salinity) * 1000
    delta_pco2 = (pco2_ocean - pco2_atm) * 1e-6
    return k * k0 * delta_pco2
