import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import pandas as pd
import time

import os
import sys

def v_normed(vi, planet='Earth'):
    # velocities of circular orbits just above surface
    V_norms = {'Mercury': 3.01,
               'Venus': 7.33,
               'Earth': 7.91,
               'Mars': 3.55,
               'Jupiter': 42.1,
               'Saturn': 25.1,
               'Uranus': 15.0,
               'Neptune': 16.6}

    V_norm = V_norms[planet]
    vi_norm = vi / V_norm

    return vi_norm

def F_factor(vi, r, planet):
    # vi: initial velocity (km/s)
    # r: distance of the observer from the planet (planet radii)
    vi_norm = v_normed(vi, planet=planet)

    return (vi_norm ** 2.0) * r

def big_B(xi, F):
    t1 = (1.0 - np.cos(xi)) / 4.0
    t2 = 1.0 - np.cos(xi) + (4.0 / F)
    return np.sqrt(t1 * t2)

def sin_phi_function(vi, xi, F):
    vi_norm = v_normed(vi, planet='Earth')

    v = vi_norm * np.sqrt((F + 2.0) / F)  # speed, v(F), of the particles at the observer

    v_radial_short = vi_norm * (((1.0 + np.cos(xi)) / 2.0) - big_B(xi, F))
    v_radial_long = vi_norm * (((1.0 + np.cos(xi)) / 2.0) + big_B(xi, F))

    v_transverse_short = np.sqrt((v**2.0) - (v_radial_short**2.0))
    v_transverse_long = np.sqrt((v**2.0) - (v_radial_long**2.0))

    sin_phi_short = v_transverse_short / v
    sin_phi_long = v_transverse_long / v

    return sin_phi_short, sin_phi_long

def b_parameter(r, F, sin_phi):
    return r * sin_phi * np.sqrt((F + 2.0) / F)

def minimum_q(r, F, sin_phi):
    # closest distance of approach to the centre of the planet

    # If q > 1 there is no shielding and ηS = 1. If q  1, the particle
    # is stopped as soon as it reaches the surface of the planet so that
    # ηS = 0, unless it reaches the target point P before periapsis.

    term = F * (F + 2.0) * (sin_phi ** 2.0)
    q = r * (np.sqrt(1.0 + term) - 1.0) / F

    return q

def local_enhancement_factor(vi, xi, F):
    sin_phi_short, sin_phi_long = sin_phi_function(vi, xi, F)

    # equation 6 from Jones & Poole 2007
    term1S = sin_phi_short / np.sin(xi)
    term1L = sin_phi_long / np.sin(xi)
    term2 = (F + 1.0) / F
    term3S = 1 / (2.0 * np.sin(xi) * sin_phi_short)
    term3L = 1 / (2.0 * np.sin(xi) * sin_phi_long)
    term4 = 1.0 + np.cos(xi)
    term5a = np.cos(xi) - (2.0 * big_B(xi, F))
    term5b = np.cos(xi) + (2.0 * big_B(xi, F))
    term6 = F + 2
    term7 = 1.0 - (np.cos(xi)**2.0)
    term8 = 2.0 * F * big_B(xi, F)

    nu_short = (term1S * term2) - term3S * (((term4 * term5a) / term6) + (term7/term8))
    nu_long = (term1L * term2) - term3L * (((term4 * term5b) / term6) - (term7/term8))

    return nu_short, nu_long

def shielding_factor(q, xi, F):
    if q > 1.0:
        return 1.0
    elif q <= 1.0:
        return 0.0
    elif np.cos(xi) > 1.0 / (F + 1.0):
        return 0.0

def total_enhancement(N, distribution_array, lat_mids, vel_mids, p='Earth'):
    # distribution_array is a numpy array of shape (N_lat, N_lon, N_vel) and the values are the weights
    distro = np.sum(distribution_array, axis=0)  # sum over all longitudes
    concatenated_d = np.concatenate(distro)

    idx_array = np.reshape(np.arange(np.size(distro)), np.shape(distro))
    concatenated_idx = np.concatenate(idx_array)

    # set up grid
    r = np.linspace(1.0, 60.0, 1000)  # 60 Earth radii = 1 lunar sistance
    nu_value = np.zeros(1000)

    # iterate along grid
    for j, ri in enumerate(r):
        for i in range(N):
            idx = np.random.choice(concatenated_idx, p=concatenated_d/np.sum(concatenated_d))
            lat_idx, vel_idx = np.where(idx_array==idx)
            xi = np.abs(lat_mids[lat_idx] * np.pi / 180.0)  # convert to radians
            vi = vel_mids[vel_idx]

            F = F_factor(vi, ri, planet=p)

            sin_phi_short, sin_phi_long = sin_phi_function(vi, xi, F)

            shielding_short = shielding_factor(minimum_q(ri, F, sin_phi_short), xi, F)
            shielding_long = shielding_factor(minimum_q(ri, F, sin_phi_long), xi, F)

            nu_short, nu_long = local_enhancement_factor(vi, xi, F)

            nu_value[j] += (shielding_short * nu_short) + (shielding_long * nu_long)

        nu_value[j] = nu_value[j] / N

    return r, nu_value
