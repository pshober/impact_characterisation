import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import pandas as pd
import time
import os
import sys

def F_factor(vi, r, planet='Earth'):
    # vi: initial velocity (km/s)
    # r: distance of the observer from the planet (planet radii)

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

    return (vi_norm ** 2.0) * r

def sin_phi(vi, xi, F):

    v = vi * np.sqrt((F + 2.0) / F)  # speed, v(F), of the particles at the observer

    v_radial_short = vi * (((1.0 + np.cos(xi)) / 2.0) - big_B(xi, F))
    v_radial_long = vi * (((1.0 + np.cos(xi)) / 2.0) + big_B(xi, F))

    v_transverse_short = np.sqrt((v**2.0) - (v_radial_short**2.0))
    v_transverse_long = np.sqrt((v**2.0) - (v_radial_long**2.0))

    sin_phi_short = v_transverse_short / v
    sin_phi_long = v_transverse_long / v

    return sin_phi_short, sin_phi_long

def b_parameter(r, F, sin_phi):
    return r * sin_phi * np.sqrt((F + 2.0) / F)

def big_B(xi, F):
    t1 = (1.0 - np.cos(xi)) / 4.0
    t2 = 1.0 - np.cos(xi) + (4.0 / F)
    return np.sqrt(t1 * t2)

def minimum_q(r, F, sin_phi):
    # closest distance of approach to the centre of the planet

    # If q > 1 there is no shielding and ηS = 1. If q  1, the particle
    # is stopped as soon as it reaches the surface of the planet so that
    # ηS = 0, unless it reaches the target point P before periapsis.

    term = F * (F + 2.0) * (sin_phi ** 2.0)
    q = r * (np.sqrt(1.0 + term) - 1.0) / F

    return q

def local_enhancement_factor(vi xi, F):
    sin_phi_short, sin_phi_long = sin_phi(vi, xi, F)

    # equation 6 from Jones & Poole 2007
    term1 = sin_phi / np.sin(xi)
    term2 = (F + 1.0) / F
    term3 = 1 / (2.0 * np.sin(xi) * sin_phi)
    term4 = 1.0 + np.cos(xi)
    term5a = np.cos(xi) - (2.0 * big_B(xi, F))
    term5b = np.cos(xi) + (2.0 * big_B(xi, F))
    term6 = F + 2
    term7 = 1.0 - (np.cos(xi)**2.0)
    term8 = 2.0 * F * big_B(xi, F)

    result1 = (term1 * term2) - term3 * (((term4 * term5a) / term6) + (term7/term8))
    result2 = (term1 * term2) - term3 * (((term4 * term5b) / term6) - (term7/term8))

    return result1, result2
