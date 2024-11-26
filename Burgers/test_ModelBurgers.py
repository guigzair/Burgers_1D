from re import U
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
from core.Utilities import utilities
import pickle
import time
import argparse
import matplotlib.pyplot as plt
size = 15
params = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'cm',  # Computer Modern font
	'legend.fontsize':size,
    'axes.labelsize' : size+1,
	'axes.titlesize' : size +4,
    'xtick.labelsize' : size,
    'ytick.labelsize' : size
}
plt.rcParams.update(params)

import core.Burgers.BurgersScheme as BurgersScheme
import core.Burgers.sandBox_Burgers as sandBox_Burgers



# Parameters
neighboorhoodValue = 3
nptsVal = 1024
resolutionFactor = 4
nptsCoarse = nptsVal//resolutionFactor
dx = 1 / nptsCoarse
stepSize = 10000
gamma = 1.4


# Model
smModel = sandBox_Burgers.My_Model(nptsVal,
								resolutionFactor,
								neighboorhoodValue, 
								sandBox_Burgers.core_model,
								{"lambda_ent" : 0., 
      								"lambda_reg" : 0., 
					    			"lambda_TVD" : 0., 
									"activation" : "selu"}
						)
smModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
x = tf.random.uniform((100,nptsCoarse))
a  = smModel(x, training=False)

#Load weights
dir = './logs/Periodic/'
# dir = './logs/nonPeriodic/24032023_160529/'
smModel.load_weights(dir + 'weights_1d_100epochs.h5')


# # Predict score on test set
# with open('../../Data/Euler/dataset_Periodic_test_512.pkl','rb+') as f:
# 	data_test = pickle.load(f)

# data_test = utilities.Mean_coarsing(data_test, resolutionFactor)
# x_test, y_test = utilities.reshapeForTrainEuler(data_test, resolutionFactor, stepSize = 15)
# results = smModel.evaluate(x_test, y_test, batch_size=256, steps = x_test.shape[0]//256) 
# print("test loss",np.mean(results))




# Main distribution function
f = lambda x : utilities.complex_initial(x) / 2

# f = lambda x : tf.sin(2 * np.pi * x) 


fig_u, ax_u = plt.subplots()
ax_u.set_xlabel(r"$x$")
ax_u.set_ylabel(r"$u$")



# Fine reference
x_fine = tf.linspace(0.5*dx/resolutionFactor, 1. -0.5*dx/resolutionFactor, nptsVal)
x_coarse = tf.linspace(0.5*dx, 1. -0.5*dx, nptsVal//resolutionFactor)
u = tf.map_fn(f, x_fine)

u = u[tf.newaxis, ...]
stack = BurgersScheme.integrate_stack(u, stepSize * resolutionFactor, 1/nptsVal, 0.)
stack_fine = utilities.Mean_coarsing(stack , resolutionFactor)
# stack_fine = stack[...,::resolutionFactor]

ax_u.plot(x_coarse,stack_fine[-1,0],label=r"u fine")


# Coarse reference
x_coarse = tf.linspace(0.5*dx, 1. -0.5*dx, nptsVal//resolutionFactor)
u = tf.map_fn(f, x_coarse)

u = u[tf.newaxis, ...]
stack_coarse = BurgersScheme.integrate_stack(u, stepSize, dx, 0.)

ax_u.plot(x_coarse,stack_coarse[-1,0],label=r"u coarse")


## after N steps ML
x_coarse = tf.linspace(0.5*dx, 1. -0.5*dx, nptsVal//resolutionFactor)
u = tf.map_fn(f, x_coarse)
u = u[tf.newaxis, ...]

loss_ML = []
for t in range(stepSize):
	u = smModel(u, training=False)
	loss = keras.losses.mean_absolute_error(u[0], stack_fine[t*resolutionFactor,0])  
	loss_ML.append(loss)


# plot 

ax_u.plot(x_coarse,u[0],label=r"u ML")
ax_u.grid()
ax_u.legend()
fig_u.savefig('images/u.jpg', dpi = 500)


# Get error from coarse and ML compared to fine

loss_coarse = keras.losses.mean_absolute_error(stack_coarse[:,0], stack_fine[::resolutionFactor,0])

fig, ax = plt.subplots()
ax.plot(loss_coarse, label = r"Error coarse/fine")
ax.plot(loss_ML, label = r"Error ML/fine")
ax.grid()
ax.legend()
fig.savefig('images/erreur.jpg', dpi = 500)




#############################################################################################
############################## plot coeffcients #############################################
#############################################################################################

coeffs_deriv = smModel.get_coeffs(u)
coeffs_deriv = coeffs_deriv[0]

fig, ax = plt.subplots()
ax.plot(tf.reduce_mean(coeffs_deriv,0), label = r"coeffs deriv $\rho$")
ax.grid()
ax.legend()
fig.savefig('images/coeffs_deriv.jpg', dpi = 500)


N_point = 45
var = 0
fig, ax1 = plt.subplots()
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$u$")
ax1.plot(x_coarse,u[0], c="k", marker='o', markersize = 3, label="outer plot")

# draw line between coeff plot and figure
x1, y1 = [x_coarse[N_point], 0.2], [u[0,N_point], 0.5]
ax1.plot(x1, y1, color = 'k')
ax1.plot(x_coarse[N_point], u[0,N_point], marker="o", markersize=5, markerfacecolor="r")
left, bottom, width, height = [0.2, 0.4, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.stem(coeffs_deriv[var,N_point])
MIN, MAX = min(coeffs_deriv[var,N_point]), max(coeffs_deriv[var,N_point])
ax2.set_xlim([-0.5,2.5]) 
ax2.set_ylim([MIN-0.1,MAX + 0.1]) 
ax1.grid()
ax2.grid()
fig.savefig('images/coeffs_deriv_location.jpg', dpi = 500)

mean = tf.reduce_mean(coeffs_deriv[1]- np.array([0.,-1.,1.]), 0)
tf.reduce_sum(mean)


stencil_size = 3
stencil = np.arange(stencil_size) - (stencil_size) // 2
stencil = stencil 

# create the bias vector, with true coefficients
A = []
for i in range(stencil_size):
	A.append(stencil**i)
A = np.array(A)
b = []
for i in range(stencil_size):
	if i == 1:
		b.append(np.math.factorial(0))
	else : 
		b.append(0)
b = np.array(b)
alpha_p = np.linalg.solve(A,b)
print(alpha_p)