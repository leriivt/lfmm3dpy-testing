#takes about 5 minutes to install
import fmm3dpy as fmm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

'''
Given 1e6 source charge electrons uniformly randomly distributed in a 1m x 1m area on the xy plane,
use FMM to find electric field at 1e6 uniformly randomly distributed target points in a 1m x 1m area on the xy plane
1m above the plane of electrons

Plot electrons and electric field in 3d

'''
skip = 1000

eps = 10**(-4)
e_charge = -1.6e-19 #charge of an electron in coulombs
e_permitivity_vac = 8.854e-12
k_e = 1 / 4 / np.pi / e_permitivity_vac #coulomb constant



#generate source locations
n = (int)(1e6) #1e6 sources
sources = np.random.uniform(0,1,(3,n)) #generate x,y,z coords
sources[2] = np.zeros(n) #set all the z coords to 0



#setup charge values (all will have charge of 1 electron)
charges = np.full(n, e_charge)
#charges = np.full(n, 1)
#print(charges)

#generate target locations
nt = n #1e6 targets
targets = np.random.uniform(0,1,(3,nt)) #generate x,y,z coords
#targets=np.array(sources)
targets[2] = np.ones(nt) #set all the z coords to 1
#print(targets)

#timer
start_time = time.perf_counter()

out = fmm.lfmm3d(eps=eps,sources=sources,charges=charges,targets=targets,pg=0,pgt=2)
grad = out.gradtarg
#grad_s = out.grad
#print(grad)
e_field = grad * -k_e #scale gradient by coulomb constant
#e_field_s = grad_s * -k_e 
end_time = time.perf_counter()
time_elapsed = end_time - start_time
print("time to calculate e_field at targets: ")
print(time_elapsed)
#print(e_field)

#calculate the magnitudes of electric field
mag_e_field = np.linalg.norm(e_field, axis=0)
#print("mag e field:\n")
#print(mag_e_field)
end_time = time.perf_counter()
time_elapsed = end_time - start_time
print("time to calculate e_field_mag at targets: ")
print(time_elapsed)


#map electric field magnitudes to different colors
norm = mpl.colors.Normalize(vmin=np.min(mag_e_field), vmax=np.max(mag_e_field))
cmap = mpl.colormaps['viridis']
colors = cmap(norm(mag_e_field))

#setup for plotting in 3d
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
#plot electrons
ax.scatter(sources[0][::skip], sources[1][::skip], sources[2][::skip], c='m')


# Make the grid
x, y, z = targets
u, v, w = e_field

#non-colorful plotting of e_field vectors
#ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

#plot colorful e_field vectors
quiver = ax.quiver(x[::skip], y[::skip], z[::skip], u[::skip], v[::skip], w[::skip], length=0.1, normalize=True, colors=colors[::skip])
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, label='Electric Field [V/m]')


ax.set_xlabel('X Axis [m]')
ax.set_ylabel('Y Axis [m]')
ax.set_zlabel('Z Axis [m]')

plt.show()


#plot electric field magnitudes as 2d scatter plot
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(targets[0][::skip], targets[1][::skip], c=mag_e_field[::skip], cmap='viridis')
plt.colorbar(scatter, label='Electric Field [V/m]')

ax.set_xlabel('X Pos [m]')
ax.set_ylabel('Y Pos [m]')

plt.show()





'''
#attempt with maxwell wrapper-----------------------------------------------------------
print("\nmaxwell wrapper---------------------------------------\n")
#zk = 1.1 + 1j*0
#zk=0.1+ 1j*0
zk=0.1+ 1j*0.1
charges = np.full(n, e_charge)
out = fmm.emfmm3d(eps=eps,zk=zk, sources=sources,e_charge=charges,targets=targets,ifE = 1, nd=1)
e_field = -k_e * out.E[0]

#calculate the magnitudes of electric field
mag_e_field = np.linalg.norm(abs(e_field), axis=0)
print("mag e field:\n")
print(mag_e_field)

#setup for plotting in 3d
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

#plot electrons
ax.scatter(sources[0], sources[1], sources[2], c='m')

x, y, z = targets
u, v, w = abs(e_field)
for index in range(len(u)):
    u[index] = np.sign(e_field[0][index]) * u[index]
for index in range(len(v)):
    v[index] = np.sign(e_field[1][index]) * v[index]
for index in range(len(w)):
    w[index] = np.sign(e_field[2][index]) * w[index]

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.show()
'''
