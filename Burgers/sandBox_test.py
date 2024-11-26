import tensorflow as tf
from tensorflow import keras
import sandBox
import sys
sys.path.insert(1, '../Utilities')
import utilities
import numpy as np
import matplotlib.pyplot as plt

###########################  Test setU  ##########################################
neighborhoodValue = 6

npts = 32

a = tf.constant([[0,0.,0.,1.,0.,0]], dtype=tf.float32)
coeffs = tf.tile(a, [npts,1])

u = tf.map_fn(lambda x : np.sin(2*np.pi*x+np.pi/2), elems = tf.linspace(0,1,npts),fn_output_signature=tf.float32)
u = u[tf.newaxis, ...]
u_output = sandBox.setU(u, coeffs, neighborhoodValue)

fig, ax = plt.subplots()
ax.plot(tf.linspace(0,1,npts),u[0],label=r"u start")
ax.plot(tf.linspace(0,1,npts),u_output[0],label=r"u setU")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u$")
ax.grid()
ax.legend()






###########################  Test pseudoDeriv  ##########################################

neighborhoodValue = 6

npts = 128
dx = 1/npts 

a = tf.constant([[0,0,-1.,1.,0,0]], dtype=tf.float32)
coeffs = tf.tile(a, [npts,1])

u = tf.map_fn(lambda x : np.sin(2*np.pi*x), elems = tf.linspace(0.,1.-dx,npts),fn_output_signature=tf.float32)
u = u[tf.newaxis, ...]

u_output = sandBox.pseudoDerivative(u, coeffs, neighborhoodValue, dx)

fig, ax = plt.subplots()
ax.plot(tf.linspace(0,1,npts),u[0],label=r"u start")
ax.plot(tf.linspace(0,1,npts),u_output[0]/(2*np.pi),label=r"u deriv")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u$")
ax.grid()
ax.legend()




###########################  euler step  ##########################################
neighborhoodValue = 6
npts = 128
dx = 1/npts

a = tf.constant([[0,0,-1.,1.,0,0]], dtype=tf.float32)
coeffs_deriv = tf.tile(a, [npts,1])
a = tf.constant([[0,0,1./2.,1./2.,0,0]], dtype=tf.float32)
coeffs_u = tf.tile(a, [npts,1])
coeffs = tf.stack([coeffs_u, coeffs_deriv])
coeffs = coeffs[tf.newaxis,...]

u = tf.map_fn(lambda x : utilities.initFuncSquare(x,0.1,0.4), elems = tf.linspace(0,1,npts),fn_output_signature=tf.float32)
u_input = u[tf.newaxis,...]
for t in range(5):
    u_output = sandBox.euler_step(u_input, coeffs, neighborhoodValue, dx, 0.00001)
    u_input = u_output

fig, ax = plt.subplots()
ax.plot(tf.linspace(0,1,npts),u,label=r"u start")
ax.plot(tf.linspace(0,1,npts),u_output[0],label=r"u step")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u$")
ax.grid()
ax.legend()


###########################  minmod step  ##########################################
neighborhoodValue = 6
npts = 128
dx = 1/npts

a = tf.constant([[0,0,-1.,1.,0,0]], dtype=tf.float32)
coeffs_deriv = tf.tile(a, [npts,1])
a = tf.constant([[0,0,0.,1.,0,0]], dtype=tf.float32)
coeffs_u = tf.tile(a, [npts,1])
coeffs = tf.stack([coeffs_u, coeffs_deriv])
coeffs = coeffs[tf.newaxis,...]

u = tf.map_fn(lambda x : utilities.initFuncSquare(x,0.1,0.4), elems = tf.linspace(0,1,npts),fn_output_signature=tf.float32)
u_input = u[tf.newaxis,...]
for t in range(150):
    u_output, u_t = sandBox.EulerMinMod_step(u_input, coeffs, neighborhoodValue, dx, 0.00001)
    u_input = u_output

fig, ax = plt.subplots()
ax.plot(tf.linspace(0,1,npts),u,label=r"u start")
ax.plot(tf.linspace(0,1,npts),u_output[0],label=r"u step")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u$")
ax.grid()







