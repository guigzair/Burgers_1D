import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "Helvetica"
})
import sys
sys.path.insert(1, '../Utilities')
import utilities

nu = 0.


def derivative(T, dx):
	return (tf.roll(T, -1, -1) - tf.roll(T, 0, -1))/(dx)

def advance_time_step(u, dx, nu):
	CFL = 0.01

	# Derivative
	grad = derivative(u, dx)

	# minmod
	aa = tf.minimum(tf.sign(tf.roll(grad,0,-1)),0.0) * 2.0 + 1.0
	ri = tf.roll(grad,1,-1)/(tf.maximum(tf.abs(tf.roll(grad,0,-1)) ,1e-15) * aa)
	phi= tf.maximum(tf.minimum(ri,1.0),0.0)
	# phi= (ri**2 + ri)/(ri**2 + 1) # Van albada

	u_L = tf.roll(u, 0, axis = -1) + 1./2.*dx*phi*tf.roll(grad,0,-1)
	u_R =  tf.roll(u, -1, axis = -1) - 0.5 * dx * tf.roll(phi,-1,-1) * tf.roll(grad, -1, axis = -1)


	# flux and average
	fconv = 0.5 * (u_L**2/2 + u_R**2/2)
	# Add stabilizing diffusive terms
	fconv   -=  0.5 * tf.abs(u_L+u_R)/2 * (u_R - u_L)

	# finite volume
	fdiff = -nu * grad
	flux = fconv + fdiff
	div = flux - tf.roll(flux,1,-1)

	u = u - CFL*div

	return u
	

def integrate(u, numsteps, dx, nu):
	for _ in range(numsteps):
		u = advance_time_step(u, dx, nu)
	return u

def integrate_stack(u, numsteps, dx, nu):
	stack = [u]	
	for _ in range(numsteps - 1):
		u = advance_time_step(u, dx, nu)
		stack  = tf.concat([stack, [u]], 0)
	return stack


# #############   Essai ###########

# Npts = 128 
# dx = 1/Npts
# numSteps = 0
# x = tf.linspace(0.5*dx, 1. - 0.5*dx, Npts)
# f = lambda x : utilities.initFuncSquare(x,0.3,0.5)


# u_start = tf.map_fn(f, elems = x)
# # a = np.random.rand(2)
# # if a[0]<a[1]:
# # 	min = a[0]
# # 	max = a[1]
# # else :  
# # 	min = a[1]
# # 	max = a[0]
# # if max - min < 0.1 :
# # 	if max > 0.9 : 
# # 		min = min - 0.1
# # 	else : max = max  + 0.1 
# # if max - min > 0.8 :
# # 	max = max - 0.15
# # h = np.random.rand()
# # u =  tf.map_fn(lambda x : utilities.initFuncSquare(x,min,max), elems=x)
# # N = 20
# # A = np.random.uniform(low=-0.5, high=0.5, size=N)
# # phi = np.random.uniform(low=0, high=2*np.pi, size=N)
# # l = np.random.choice(4, N)+3
# # f= lambda x : utilities.random_sin(x, A, phi, l) + 1.
# # f = lambda x : np.sin(2*np.pi*x)
# f = lambda x : utilities.complex_initial(x) / 2

# # N = 20
# # A = np.random.uniform(low=-0.5, high=0.5, size=N)
# # phi = np.random.uniform(low=0, high=2*np.pi, size=N)
# # l = np.random.choice(4, N)+3
# # h = 2*np.random.rand()-1.
# # u =  tf.map_fn(lambda x : utilities.random_sin(x, A, phi, l), elems=x)
# # u_start = u * (1 - tf.map_fn(lambda x : utilities.initFuncSquare(x,0.15,0.5), elems=x)) + 0.5 * tf.map_fn(lambda x : utilities.initFuncSquare(x,0.15,0.5), elems=x)


# u_start = tf.map_fn(f, elems = x)
# u_start = u_start[tf.newaxis,...]

# # u_start = tf.stack([u_start,  tf.map_fn(f, elems = x)])

# u = integrate(u_start, numSteps, dx, nu)
# stack = integrate_stack(u_start, numSteps, dx, nu)

# fig, ax = plt.subplots()
# ax.plot(x, u_start[0], label = r"$u$ start")
# # ax.plot(x, u[0], label = r"$u$ end")
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$u$")
# ax.grid()
# ax.legend()
# fig.savefig('images/burgers_scheme.jpg', dpi = 500)

# import json
# SCORE = []
# for i in range(40):
# 	with open('./custom_training/trial_'+str(f"{i:02}")+'/trial.json','rb+') as f:
# 		data = json.load(f)
# 		SCORE.append(data['score'])