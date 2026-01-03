#takes about <5 minutes to install
import fmm3dpy as fmm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

'''
Given 1e6 source charge electrons uniformly randomly distributed in a 1m x 1m area on the xy plane,
use FMM to find electric field at 1e6 uniformly randomly distributed target points in a 1m x 1m area on the xy plane
1m above the plane of electrons

- Time the time it takes to calculate electric field at sources, and time to calculate electric field at targets

- Plot electrons and electric field in 3d

'''

eps = 10**(-5) #machine epsilon (determines precision)
e_charge = -1.6e-19 #charge of an electron in coulombs
e_permitivity_vac = 8.854e-12 #electric permitivity of free space
k_e = 1 / 4 / np.pi / e_permitivity_vac #coulomb constant


#generate source locations
n = (int)(1e6) #1e6 sources
sources = np.random.uniform(0,1,(3,n)) #generate source x,y,z coords
sources[2] = np.zeros(n) #set all the z coords to 0


#setup charge values (all will have charge of 1 electron)
charges = np.full(n, e_charge) #array with length equal to number of sources, filled with e_charge at every index


#generate target locations
nt = (int)(1e6) #1e6 targets
targets = np.random.uniform(0,1,(3,nt)) #generate target x,y,z coords
targets[2] = np.ones(nt) #set all the z coords to 1


#find electric field at sources
start_time_s = time.perf_counter() #start timer

out_s = fmm.lfmm3d(eps=eps,sources=sources,charges=charges,pg=2) #performs FMM to solve for potential and gradient at sources
grad_s = out_s.grad #extract gradient at sources
e_field_s = grad_s * -k_e #scale gradient by coulomb constant to calculate electric field

end_time_s = time.perf_counter() #stop timer
time_elapsed_s = end_time_s - start_time_s

print("time to calculate e_field at sources:\n" + str(time_elapsed_s) + "\n")


#find electric field at targets
start_time_t = time.perf_counter() #start timer

out_t = fmm.lfmm3d(eps=eps,sources=sources,charges=charges,targets=targets,pgt=2) #performs FMM to solve for potential and gradient at targets
grad_t = out_t.gradtarg #extract gradient at targets
e_field_t = grad_t * -k_e #scale gradient by coulomb constant to calculate electric field

end_time_t = time.perf_counter() #stop timer
time_elapsed_t = end_time_t - start_time_t

print("time to calculate e_field at targets:\n" + str(time_elapsed_t) + "\n")


#find electric field at both sources and targets
start_time_st = time.perf_counter() #start timer

out_st = fmm.lfmm3d(eps=eps,sources=sources,charges=charges,targets=targets,pg=2,pgt=2) #performs FMM to solve for potential and gradient at both sources and targets

grad_st_s = out_st.grad
e_field_st_s  = grad_st_s * -k_e #sources e_field

grad_st_t = out_st.gradtarg
e_field_st_t  = grad_st_t * -k_e #targets e_field

end_time_st = time.perf_counter() #stop timer
time_elapsed_st = end_time_st - start_time_st

print("time to calculate e_field at sources and targets:\n" + str(time_elapsed_st) + "\n")


#calculate the magnitudes of electric field at targets
mag_e_field_t = np.linalg.norm(e_field_t, axis=0)
print("mag e field:\n" + str(mag_e_field_t) + "\n")

mag_e_field_st_t = np.linalg.norm(e_field_st_t, axis=0)
print("mag e field:\n" + str(mag_e_field_st_t) + "\n")

mag_e_field_s = np.linalg.norm(e_field_s, axis=0)
print("mag e field:\n" + str(mag_e_field_s) + "\n")

mag_e_field_st_s = np.linalg.norm(e_field_st_s, axis=0)
print("mag e field:\n" + str(mag_e_field_st_s) + "\n")


print(np.array_equal(e_field_t, e_field_st_t))
print(np.array_equal(mag_e_field_t, mag_e_field_st_t))
print(np.array_equal(np.round(e_field_s, decimals=5), np.round(e_field_st_s, decimals=5)))
print(np.array_equal(np.round(mag_e_field_s, decimals=5), np.round(mag_e_field_st_s, decimals=5)))



#plotting---------------------------------------------------------------------
skip = 1000 #plot every 1000 points so graphic doesnt take too long to load

#setup for plotting in 3d
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

#plot electrons
ax.scatter(sources[0][::skip], sources[1][::skip], sources[2][::skip], c='m')

x, y, z = targets #separate out components of target locations
u, v, w = e_field_t #separate out components of target e_field vector

#non-colorful plotting of e_field vectors
#ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

#map electric field magnitudes to different colors
norm = mpl.colors.Normalize(vmin=np.min(mag_e_field_t), vmax=np.max(mag_e_field_t)) #generate object that can normalize mag_e_field_t to a [0 to 1] range so that it can be colormapped
cmap = mpl.colormaps['viridis'] #get the colormap object from matplotlib
colors = cmap(norm(mag_e_field_t)) #generate 2d array of RGBA data based on e field magnitudes using normalization and color mapping objects

#plot colorful e_field vectors
quiver = ax.quiver(x[::skip], y[::skip], z[::skip], u[::skip], v[::skip], w[::skip], length=0.1, normalize=True, colors=colors[::skip])
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, label='Electric Field [V/m]')

ax.set_xlabel('X Axis [m]')
ax.set_ylabel('Y Axis [m]')
ax.set_zlabel('Z Axis [m]')

plt.show()


#plot electric field magnitudes as 2d scatter plot
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(targets[0][::skip], targets[1][::skip], c=mag_e_field_t[::skip], cmap='viridis')
plt.colorbar(scatter, label='Electric Field [V/m]')

ax.set_xlabel('X Pos [m]')
ax.set_ylabel('Y Pos [m]')

plt.show()
