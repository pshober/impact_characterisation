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

import datetime
import shutil

# create save folder for raw results
date_str = datetime.datetime.now().strftime('%Y%m%d')
ResultsFolder = os.path.join(os.getcwd(), 'lunar_flux_results'+date_str)

if not os.path.isdir(ResultsFolder): # Create the directory if it doesn't exist
    os.mkdir(ResultsFolder)
else:
    # Removes all the subdirectories
    shutil.rmtree(ResultsFolder)
    os.mkdir(ResultsFolder)

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

if not os.path.isdir(os.path.join(ResultsFolder,'flux_histogram_new.npy')):
    hist_results = np.zeros((n_bins, n_bins, n_bins))  # new histogram
else:
    hist_results = np.load(os.path.join(ResultsFolder,'flux_histogram_new.npy'))  # load histogram to add more data to

bounds = np.linspace(-side_distance, side_distance, int(n_bins+1))
distance_per_bin = (2.0 * side_distance) / n_bins

'--------------------------------------------------------------------'

try:
    total_particles_simulated = np.load(os.path.join(ResultsFolder,'total_particles_simulated.npy'))
except FileNotFoundError:
    total_particles_simulated = 0

total_loops = 10000

# set up simulation
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = 'whfast'
sim.dt = 20.0

sim.add("399") # Earth
active_particles = sim.N
sim.N_active = active_particles
sim.move_to_com()

sim.save("solar_system.bin")

for n in range(total_loops):

    N_particles = int(6e4)
    n_outputs = int(8e2)

    print(f"Loop {n+1} \n")

    sim = rebound.Simulation("solar_system.bin")

    # define start x,y,z
    r_start = 3.9 * 3.844e8  # LD (m) (approx. 1 Hill-sphere for the Earth)

    def polar2cart(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    # position is uniform
    theta_pos = np.random.uniform(0.0, np.pi, N_particles)
    phi_pos = np.random.uniform(0, 2.0 * np.pi, N_particles)

    # velocities theta are based on pokorny output
    distro = np.sum(distribution_array, axis=1)  # sum over all longitudes
    concatenated_d = np.concatenate(distro)

    idx_array = np.reshape(np.arange(np.size(distro)), np.shape(distro))
    concatenated_idx = np.concatenate(idx_array)

    idx = np.random.choice(concatenated_idx, N_particles, p=concatenated_d/np.sum(concatenated_d))
    lat_idx = [np.where(idx_array==i)[0][0] for i in idx]
    vel_idx = [np.where(idx_array==i)[1][0] for i in idx]

    lat_idx = np.array(lat_idx)
    vel_idx = np.array(vel_idx)

    # REMOVE EDGE VELOCITIES
    lat_idx = lat_idx[vel_idx!=99]
    theta_pos = theta_pos[vel_idx!=99]
    phi_pos = phi_pos[vel_idx!=99]
    vel_idx = vel_idx[vel_idx!=99]

    # update
    N_particles = len(vel_idx)

    vels = vel_mids[vel_idx] * 1e3  # covert from km/s to m/s
    theta_vel = np.deg2rad(lat_mids[lat_idx]) + (np.pi/2.0) # convert to radians and shift to correct range for spherical coords
    phi_vel = np.random.uniform(0, 2.0 * np.pi, N_particles)

    # convert spherical to cartesian
    x, y, z = polar2cart(r_start, theta_pos, phi_pos)
    vx, vy, vz = polar2cart(vels, theta_vel, phi_vel)

    # add particles
    for i in range(N_particles):
        vr_dot = np.dot(np.array([x[i],y[i],z[i]]), np.array([vx[i],vy[i],vz[i]]))  # calculate velocity relative to Earth

        if vr_dot <= 0: # only simulate particles that approach the Earth (i.e. relative velocity <= 0)
            sim.add(m=0.0, x=x[i], y=y[i], z=z[i], vx=vx[i], vy=vy[i], vz=vz[i])

    # update again
    N_particles = sim.N - sim.N_active

    results = np.zeros(shape=(n_outputs*(sim.N-1), 4)) * np.nan
    times = np.linspace(0.0, 2.0*24.0*60.0*60.0, n_outputs)

    for i, step in enumerate(times):
        percent = round(((i+1)/n_outputs*100),2)
        print(f"{percent} Integrated\nTimestep: {sim.dt}", flush=True, end="\033[F")

        sim.integrate(step)

        for j in range(1, sim.N):
            results[i*j,0] = j
            results[i*j,1] = sim.particles[j].x
            results[i*j,2] = sim.particles[j].y
            results[i*j,3] = sim.particles[j].z

            # results[i*j,4] = sim.particles[j].vx
            # results[i*j,5] = sim.particles[j].vy
            # results[i*j,6] = sim.particles[j].vz

    # 'save results in a fits file'
    # results_fits = fits.PrimaryHDU()
    # results_name = os.path.join(ResultsFolder, 'velocity_flux_results_'+str(n)+'.fits')
    # results_fits.writeto(results_name)
    #
    # cols = [fits.Column(name='index', format='D', array=results[:,0]),
    #         fits.Column(name='x', format='D', array=results[:,1]),
    #         fits.Column(name='y', format='D', array=results[:,2]),
    #         fits.Column(name='z', format='D', array=results[:,3]),
    #         fits.Column(name='vx', format='D', array=results[:,4]),
    #         fits.Column(name='vy', format='D', array=results[:,5]),
    #         fits.Column(name='vz', format='D', array=results[:,6])]
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
    y_indices[np.where(x_indices < 0)[0]] = 0.0
    z_indices[np.where(x_indices < 0)[0]] = 0.0

    x_indices[np.where(x_indices > n_bins-1)[0]] = 0.0
    y_indices[np.where(x_indices > n_bins-1)[0]] = 0.0
    z_indices[np.where(x_indices > n_bins-1)[0]] = 0.0

    x_indices[np.where(y_indices < 0)[0]] = 0.0
    y_indices[np.where(y_indices < 0)[0]] = 0.0
    z_indices[np.where(y_indices < 0)[0]] = 0.0

    x_indices[np.where(y_indices > n_bins-1)[0]] = 0.0
    y_indices[np.where(y_indices > n_bins-1)[0]] = 0.0
    z_indices[np.where(y_indices > n_bins-1)[0]] = 0.0

    x_indices[np.where(z_indices < 0)[0]] = 0.0
    y_indices[np.where(z_indices < 0)[0]] = 0.0
    z_indices[np.where(z_indices < 0)[0]] = 0.0

    x_indices[np.where(z_indices > n_bins-1)[0]] = 0.0
    y_indices[np.where(z_indices > n_bins-1)[0]] = 0.0
    z_indices[np.where(z_indices > n_bins-1)[0]] = 0.0

    hist_results[x_indices, y_indices, z_indices] += 1.0
    hist_results[0, 0, 0] = 0

    np.save(os.path.join(ResultsFolder,'flux_histogram_new.npy'), hist_results)

    total_particles_simulated += sim.N-active_particles
    np.save(os.path.join(ResultsFolder,'total_particles_simulated'), total_particles_simulated)


# # create plots
# middies = middle4bins(bounds)
# axisBoi = np.meshgrid(middies,middies)
# plt.hist2d(np.concatenate(axisBoi[0]), np.concatenate(axisBoi[1]), weights=np.concatenate(np.sum(hist_results[:,:,90:110], axis=2)), norm=mpl.colors.LogNorm(), bins=100)
# plt.show()

# create csv
middies = middle4bins(bounds)
x_coords, y_coords, z_coords = np.meshgrid(middies,middies,middies)

x_coords = np.concatenate(np.concatenate(x_coords))
y_coords = np.concatenate(np.concatenate(y_coords))
z_coords = np.concatenate(np.concatenate(z_coords))

weights = np.concatenate(np.concatenate(hist_results))

hist_df = pd.DataFrame({'x': x_coords, 'y': y_coords, 'z': z_coords, 'count': weights})
hist_df.to_csv('flux_enhancement_results.csv')

# sum rotationally around z-axis
def cart2polar(x, y, z):
        r = np.sqrt((x**2.0)+(y**2.0)+(z**2.0))
        theta = np.arccos(z / r)

        # if x > 0:
        #     phi = np.arctan(y / x)
        # elif (x < 0) and (y >= 0):
        #     phi = np.arctan(y / x) + np.pi
        # elif (x < 0) and (y < 0):
        #     phi = np.arctan(y / x) - np.pi
        # elif (x == 0) and (y > 0):
        #     phi = np.pi / 2.0
        # elif (x == 0) and (y < 0):
        #     phi = -np.pi / 2.0
        # elif (x == 0) and (y == 0):
        #     phi = np.nan

        return r, theta, 0.0


r, theta, phi = cart2polar(x_coords, y_coords, z_coords)

# make all phi values the same
x_rotate, y_rotate, z_rotate = polar2cart(r, theta, phi)

# save csv where values are summed around z-axis
hist_df = pd.DataFrame({'x': x_rotate, 'y': y_rotate, 'z': z_rotate, 'count': weights})
hist_df.to_csv('flux_enhancement_rotated.csv')


# make side-view heat map with individual x & z 1D histograms
from matplotlib import cm as CM
from matplotlib import mlab as ML
from matplotlib.ticker import LogFormatter

gridsize=100

rotated_d = np.linalg.norm(np.array([x_rotate/3.844e8, z_rotate/3.844e8]), axis=0)
under_1 = rotated_d[rotated_d < 1.0]
C_values = weights / np.sqrt((x_coords**2.0)+(y_coords**2.0))
minimum_value = np.median(C_values[rotated_d < 1.0])

plt.subplot(111)
plt.hexbin(x_rotate/3.844e8, z_rotate/3.844e8, C=weights/np.median(weights), gridsize=gridsize, cmap=CM.jet, bins='log', reduce_C_function=np.mean)
plt.axis([0.0, 1.0, -1.0, 1.0])

plt.clim(1.0, 1.4)
formatter = LogFormatter(10, labelOnlyBase=False)
cb = plt.colorbar(format=formatter)
cb.set_label('mean value')
plt.show()
