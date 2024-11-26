from re import X
import tensorflow as tf
import sys
from core.Burgers import BurgersScheme
import numpy as np
import time
import pickle
import utilities
import sys
from core.Euler import EulerScheme2
from core.Euler import EulerScheme_reflexive



def make_train_data_Burgers(NB_function, npts, steps):
	nu = 0

	dx =  1. / npts
	x = tf.linspace(0.5*dx, 1. -0.5*dx, npts)

	u = 0.698 - tf.cast(tf.abs(x) > 0.5, tf.float32) * 0.698
	U = [u]
	for i in range(NB_function):
		print(i)
		a = np.random.rand()
		# if a< 0.5: 
		# 	a = np.random.rand(2)
		# 	if a[0]<a[1]:
		# 		min = a[0]
		# 		max = a[1]
		# 	else :  
		# 		min = a[1]
		# 		max = a[0]
		# 	if max - min < 0.1 :
		# 		if max > 0.9 : 
		# 			min = min - 0.1
		# 		else : max = max  + 0.1 
		# 	if max - min > 0.8 :
		# 		max = max - 0.15
		# 	h = 2 * np.random.rand()-1.
		# 	if h<0.1 and h > 0:
		# 		h = h + 0.1
		# 	elif h>-0.1 and h < 0:
		# 		h = h - 0.1
		# 	u =  tf.map_fn(lambda x : utilities.initFuncSquare(x,min,max), elems=x)
		# else:
		# 	N = 20
		# 	A = np.random.uniform(low=-0.5, high=0.5, size=N)
		# 	phi = np.random.uniform(low=0, high=2*np.pi, size=N)
		# 	l = np.random.choice(4, N)+3
		# 	h = 2*np.random.rand()-1.
		# 	u =  tf.map_fn(lambda x : utilities.random_sin(x, A, phi, l), elems=x)
		N = 20
		A = np.random.uniform(low=-0.5, high=0.5, size=N)
		phi = np.random.uniform(low=0, high=2*np.pi, size=N)
		l = np.random.choice(4, N)+3
		h = 2*np.random.rand()-1.
		u =  tf.map_fn(lambda x : utilities.random_sin(x, A, phi, l), elems=x)
		u = u * (1 - tf.map_fn(lambda x : utilities.initFuncSquare(x,0.15,0.5), elems=x)) + 0.5 * tf.map_fn(lambda x : utilities.initFuncSquare(x,0.15,0.5), elems=x)
		U = tf.concat([U, [u]], 0)
	U = U[1:]
	stack = BurgersScheme.integrate_stack(U, steps, dx, nu)
	return stack


	


###########################################################################################################################



# Burgers
NB_fonction = 5
npts = 256
ntimestep = 10000


# For Euler equations


data_tf = make_train_data_Burgers(NB_fonction, npts, ntimestep)
data_tf = tf.reshape(data_tf, (data_tf.shape[1], data_tf.shape[0], data_tf.shape[2], data_tf.shape[3]))
with open('../../Data/Burgers/dataset_mixed_256_periodic_001.pkl', "wb") as output_file:
	pickle.dump(data_tf, output_file)    