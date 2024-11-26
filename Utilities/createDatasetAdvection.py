from re import X
import tensorflow as tf
import sys
from core.Advection import AdvectionScheme
import numpy as np
import time
import pickle
import utilities
import sys




def make_train_data_Advection(NB_function, npts, steps):

	dx =  1. / npts
	x = tf.linspace(0.5*dx, 1. -0.5*dx, npts)

	f = lambda x : 0.698 - tf.sin(2 * np.pi * x)
	U = AdvectionScheme.integrate_reference_stack(f, steps, x, dx)
	U = [U]
	for i in range(NB_function):
		a = np.random.rand()
		if a < -0.2 : 
			N = 20
			A = np.random.uniform(low=-0.5, high=0.5, size=N)[:,np.newaxis]
			phi = np.random.uniform(low=0, high=2*np.pi, size=N)[:,np.newaxis]
			l = np.random.choice(6, N)[:,np.newaxis]
			h = 2*np.random.rand()-1.
			u =  tf.map_fn(lambda x : utilities.random_sin(x, A, phi, l), elems=x)
			f = lambda x : utilities.random_sin_Advection(x, A, phi, l) * (1 - utilities.initFuncSquare2(x,0.15,0.5)) + 0.5 * utilities.initFuncSquare2(x,0.15,0.5)
		else : 
			upper = np.random.randint(10,35)
			N = 20
			A = np.random.uniform(low=-0.5, high=0.5, size=N)[:,np.newaxis]
			phi = np.random.uniform(low=0, high=2*np.pi, size=N)[:,np.newaxis]
			l = np.random.choice(upper, N)[:,np.newaxis]
			f = lambda x : utilities.random_sin_Advection(x, A, phi, l)
		stack = AdvectionScheme.integrate_reference_stack(f, steps, x, dx)
		U = tf.concat([U, [stack]], 0)
	U = U[1:]
	return U

NB_function = 10
npts = 512
steps = 400

data = make_train_data_Advection(NB_function, npts, steps)

with open('../../Data/Advection/dataset_mixed_512_validation.pkl', "wb") as output_file:
	pickle.dump(data, output_file)    