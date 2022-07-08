'-------------------------------'
'Author: Patrick Shober --------'
'Date: 07/07/2022 --------------'
'-------------------------------'
'---- Flux Enhancement Code ----'
'-------------------------------'

import os
import sys
import argparse
import numpy as np
import pandas as pd
import rebound
from astropy.io import fits

import matplotlib as mpl
import matplotlib.pyplot as plt

'--------------------------------------------------------------------'
'----------CHANGE FOR EACH "pokorny_output_to_usable.py" RUN---------'

lon_bounds = np.linspace(-180.0, 180.0, 101)
lat_bounds = np.linspace(-90.0, 90.0, 101)
vel_bounds = np.linspace(10.0, 72.0, 101)  # rough estimate

def middle4bins(array):
    return np.array([(array[i]+array[i+1]) / 2.0 for i in range(len(array)-1)])

lon_mids = middle4bins(lon_bounds)
lat_mids = middle4bins(lat_bounds)
vel_mids = middle4bins(vel_bounds)

distribution_array = np.load('results_histogram_run1.npy')

'--------------------------------------------------------------------'

# create a 3d histogram
n_bins = 200
side_distance = 4e8  # just bigger than 1 LD (~ 3.844e8 m)
hist_results = np.zeros((n_bins, n_bins, n_bins))
bounds = np.linspace(-side_distance, side_distance, int(n_bins+1))
distance_per_bin = (2.0 * side_distance) / n_bins

total_loops = 20

for n in range(total_loops):

    N_particles = int(5e4)
    n_outputs = int(1e3)

    print(f"Loop {n+1} \n")

    # set up simulation
    sim = rebound.Simulation()
    sim.units = ('s', 'm', 'kg')
    sim.integrator = 'ias15'
    sim.add("399") # Earth
    active_particles = sim.N
    sim.N_active = active_particles

    # define start x,y,z
    r_start = 3.9 * 3.844e8  # LD (m) (approx. 1 Hill-sphere for the Earth)

    def polar2cart(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    # position is uniform
    theta_pos = np.random.uniform(0.0, np.pi, N_particles)
    phi_pos = np.random.uniform(-np.pi, np.pi, N_particles)

    # velocities theta are based on pokorny output
    distro = np.sum(distribution_array, axis=1)  # sum over all longitudes
    concatenated_d = np.concatenate(distro)

    idx_array = np.reshape(np.arange(np.size(distro)), np.shape(distro))
    concatenated_idx = np.concatenate(idx_array)

    idx = np.random.choice(concatenated_idx, N_particles, p=concatenated_d/np.sum(concatenated_d))
    lat_idx = [np.where(idx_array==i)[0][0] for i in idx]
    vel_idx = [np.where(idx_array==i)[1][0] for i in idx]

    vels = vel_mids[vel_idx] * 1e3  # covert from km/s to m/s
    theta_vel = np.deg2rad(lat_mids[lat_idx]) - (np.pi/2.0) # convert to radians and shift to correct range for spherical coords
    phi_vel = np.random.uniform(-np.pi, np.pi, N_particles)

    # convert spherical to cartesian
    x, y, z = polar2cart(r_start, theta_pos, phi_pos)
    vx, vy, vz = polar2cart(vels, theta_vel, phi_vel)

    # add particles
    for i in range(N_particles):
        sim.add(m=0.0, x=x[i], y=y[i], z=z[i], vx=vx[i], vy=vy[i], vz=vz[i])

    results = np.zeros(shape=(n_outputs*(sim.N-1), 4)) * np.nan
    times = np.linspace(0.0, 2.0 * 24.0 * 60.0 * 60.0, n_outputs)

    for i, step in enumerate(times):
        percent = round(((i+1)/n_outputs*100),2)
        print(f"{percent} Integrated\nTimestep: {sim.dt}", flush=True, end="\033[F")

        sim.integrate(step)

        for j in range(1, sim.N):
            results[i*j,0] = j
            results[i*j,1] = sim.particles[j].x
            results[i*j,2] = sim.particles[j].y
            results[i*j,3] = sim.particles[j].z

    # 'save results in a fits file'
    # results_fits = fits.PrimaryHDU()
    # results_name = os.path.join(os.getcwd(), 'flux_results.fits')
    # results_fits.writeto(results_name)
    #
    # cols = [fits.Column(name='index', format='D', array=results[:,0]),
    #         fits.Column(name='x', format='D', array=results[:,1]),
    #         fits.Column(name='y', format='D', array=results[:,2]),
    #         fits.Column(name='z', format='D', array=results[:,3])]
    #
    # results_fits = fits.BinTableHDU.from_columns(cols)
    #
    # results_fits_list = fits.open(results_name, mode='append')
    # results_fits_list.append(results_fits)
    # results_fits_list.writeto(results_name, overwrite=True)
    # results_fits_list.close()

    '------------------------------------------- UPDATE --------------------------------------------'

    # divide distance_per_bin
    x_indices = ((results[:,1] + side_distance) / distance_per_bin).astype(int)
    y_indices = ((results[:,2] + side_distance) / distance_per_bin).astype(int)
    z_indices = ((results[:,3] + side_distance) / distance_per_bin).astype(int)

    # remove indices out of range
    x_indices[np.where(x_indices < 0)[0]] = 0.0
    x_indices[np.where(x_indices > n_bins-1)[0]] = 0.0
    y_indices[np.where(y_indices < 0)[0]] = 0.0
    y_indices[np.where(y_indices > n_bins-1)[0]] = 0.0
    z_indices[np.where(z_indices < 0)[0]] = 0.0
    z_indices[np.where(z_indices > n_bins-1)[0]] = 0.0

    hist_results[x_indices, y_indices, z_indices] += 1.0
    hist_results[0, 0, 0] = 0

    np.save('flux_histogram.npy', hist_results)


# create plots
middies = middle4bins(bounds)
axisBoi = np.meshgrid(middies,middies)
plt.hist2d(np.concatenate(axisBoi[0]), np.concatenate(axisBoi[1]), weights=np.concatenate(np.sum(hist_results[:,:,90:110], axis=2)), norm=mpl.colors.LogNorm(), bins=100)
plt.show()

# create csv
x_coords, y_coords, z_coords = np.meshgrid(middies,middies,middies)

x_coords = np.concatenate(np.concatenate(x_coords))
y_coords = np.concatenate(np.concatenate(y_coords))
z_coords = np.concatenate(np.concatenate(z_coords))

weights = np.concatenate(np.concatenate(hist_results))

hist_df = pd.DataFrame({'x': x_coords, 'y': y_coords, 'z': z_coords, 'count': weights})
hist_df.to_csv('flux_enhancement_results.csv')
