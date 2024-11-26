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


def get_random_parameters():
	a = np.random.rand(2)
	x0 = np.min(a)
	x1 = np.max(a)
	if x1 - x0 < 0.25 :
		if x1 > 0.8 : 
			x0 = x0 - 0.25
		else : x1 = x1  + 0.2
	if x1 - x0 > 0.8 :
		x1 = x1 - 0.2
	H = np.random.rand(2) 
	h0 = np.min(H)
	h1 = np.max(H)
	if h1 - h0 < 0.25 :
		h1 = h1  + 0.25
	return  x0, x1, h0, h1


def make_train_data_Euler(NB_function, npts, steps):
	gamma                  = 1.4 

	dx =  1. / npts
	x = tf.linspace(0.5*dx, 1. -0.5*dx, npts)

	rho = 0.445 + tf.cast(tf.abs(x) > 0.5, tf.float32) * 0.005
	u = 0.698 - tf.cast(tf.abs(x) > 0.5, tf.float32) * 0.698
	P = 3.528 - tf.cast(tf.abs(x) > 0.5, tf.float32) * 2.957
	Primitives = tf.stack([rho, u , P], 0)[tf.newaxis,...]
	for i in range(NB_function):
		print(i)
		a = np.random.rand()
		if a< 0.6: 
			# N = 20
			# A = np.random.uniform(low=0., high=0.5, size=N)
			# phi = np.random.uniform(low=0, high=2*np.pi, size=N)
			# l = np.random.choice(4, N)+3
			# f = lambda x : utilities.random_sin_Euler(x, A, phi, l)
			# rho = tf.map_fn(f, elems=x)
			# u =  tf.map_fn(f, elems=x)
			# P =  tf.map_fn(f, elems=x)
			a = np.random.rand(3)
			h = np.random.rand(3)
			sgn = np.sign(np.random.rand())
			rho = tf.map_fn(lambda x : tf.sin(2*np.pi*x + a[0]*np.pi)+1.2 + h[0], elems=x)
			u =  tf.map_fn(lambda x : tf.sin(2*np.pi*x + a[1]*np.pi) + 1. + h[1] , elems=x)
			P =  tf.map_fn(lambda x : tf.sin(2*np.pi*x + a[2]*np.pi) + 1. + h[2], elems=x)
		elif a< 0.8:
			x0 = 0.6*np.random.rand()+0.2
			a = np.random.rand()+0.5
			c =  np.random.rand()
			b = np.random.rand()*2+.5
			d = np.random.rand()
			f = np.random.rand() 
			rho = a + 0.2 + tf.cast(tf.abs(x) > x0, tf.float32) * (c-0.5)
			P = b +0.2- tf.cast(tf.abs(x) > x0, tf.float32) * (d-0.5)
			u = tf.cast(tf.abs(x) > x0, tf.float32) * (f)
		else: 			
			x0, x1, h, H = get_random_parameters()
			rho = tf.map_fn(fn= lambda x : utilities.initFuncSquare(x,x0=x0, x1 = x1, H=H + 0.1, h=h+ 0.1), elems=x)
			_, _, h, H = get_random_parameters()
			sgn = np.sign(np.random.rand())
			u = tf.map_fn(fn= lambda x : utilities.initFuncSquare(x,x0=x0, x1 = x1, H=H, h=h), elems=x)
			_, _, h, H = get_random_parameters()
			P = tf.map_fn(fn= lambda x : utilities.initFuncSquare(x,x0=x0, x1 = x1, H=H , h=h ), elems=x)
		Primitives = tf.concat([Primitives, tf.stack([rho, u , P], 0)[tf.newaxis,...]], 0)
	Primitives = Primitives[1:]
	stack = EulerScheme2.integrate(Primitives, steps,dx, gamma)
	return stack


def make_train_data_Euler_nonPeriodic(NB_function, npts, steps):
	gamma                  = 1.4 

	dx =  1. / npts
	x = tf.linspace(0.5*dx, 1. -0.5*dx, npts)

	rho = 0.445 + tf.cast(tf.abs(x) > 0.5, tf.float32) * 0.005
	u = 0.698 - tf.cast(tf.abs(x) > 0.5, tf.float32) * 0.698
	P = 3.528 - tf.cast(tf.abs(x) > 0.5, tf.float32) * 2.957
	Primitives = tf.stack([rho, u , P], 0)[tf.newaxis,...]
	for i in range(NB_function):
		print(i)
		a = np.random.rand()
		if a< 0.5:
			x0 = 0.25*np.random.rand()+0.5
			a = np.random.rand()+0.5
			c =  np.random.rand()
			b = np.random.rand()*2+.5
			d = np.random.rand()
			rho = a + tf.cast(tf.abs(x) > x0, tf.float32) * (c-0.5)
			P = b - tf.cast(tf.abs(x) > x0, tf.float32) * (d-0.5)
			u = 0.698 - tf.cast(tf.abs(x) > x0, tf.float32) * 0.698
		elif a < 0.7:
			x0 = 0.1*np.random.rand()+0.35
			x1 = 0.1*np.random.rand()+0.6
			H = np.random.rand(2) 
			h = np.min(H)
			H = np.max(H)
			if H - h < 0.25 :
				H = H  + 0.25
			rho = tf.map_fn(fn= lambda x : utilities.initFuncSinus(x,x0=x0, x1 = x1, H=H, h=h), elems=x)
			x0 = 0.1*np.random.rand()+0.35
			x1 = 0.1*np.random.rand()+0.6
			H = np.random.rand(2) 
			h = np.min(H)
			H = np.max(H)
			u = tf.map_fn(fn= lambda x : utilities.initFuncSinus(x,x0=x0, x1 = x1, H=H, h=0), elems=x)
			x0 = 0.1*np.random.rand()+0.35
			x1 = 0.1*np.random.rand()+0.6
			H = np.random.rand(2) 
			h = np.min(H)
			H = np.max(H)
			P = tf.map_fn(fn= lambda x : utilities.initFuncSinus(x,x0=x0, x1 = x1, H=H, h=h), elems=x)
		elif a < 1:
			a = np.random.rand(3)
			h = np.random.rand(3)
			rho = tf.map_fn(lambda x : tf.sin(2*np.pi*x + a[0]*np.pi)+1. + h[0], elems=x)
			u =  tf.map_fn(lambda x : tf.sin(2*np.pi*x + a[1]*np.pi)+1. + h[1], elems=x)
			P =  tf.map_fn(lambda x : tf.sin(2*np.pi*x + a[2]*np.pi)+1. + h[2], elems=x)
		else: 			
			x0, x1, h, H = get_random_parameters()
			rho = tf.map_fn(fn= lambda x : utilities.initFuncSquare(x,x0=x0, x1 = x1, H=H, h=h), elems=x)
			_, _, h, H = get_random_parameters()
			u = tf.map_fn(fn= lambda x : utilities.initFuncSquare(x,x0=x0, x1 = x1, H=H, h=0), elems=x)
			_, _, h, H = get_random_parameters()
			P = tf.map_fn(fn= lambda x : utilities.initFuncSquare(x,x0=x0, x1 = x1, H=H, h=h), elems=x)
		Primitives = tf.concat([Primitives, tf.stack([rho, u , P], 0)[tf.newaxis,...]], 0)
	Primitives = Primitives[1:]
	# W = EulerScheme2.getConserved(Primitives, gamma)
	stack = EulerScheme_reflexive.integrate(Primitives, steps,dx, gamma)
	return stack

	


###########################################################################################################################

# Euler
NB_fonction = 5
npts = 512
ntimestep = 6000


# Burgers
# NB_fonction = 5
# npts = 256
# ntimestep = 10000


# For Euler equations

data_tf = make_train_data_Euler(NB_fonction, npts, ntimestep)
# data_tf = make_train_data_Burgers(NB_fonction, npts, ntimestep)
# data_tf = tf.reshape(data_tf, (data_tf.shape[1], data_tf.shape[0], data_tf.shape[2], data_tf.shape[3]))
with open('../../Data/Euler/dataset_mixed_512_validation.pkl', "wb") as output_file:
	pickle.dump(data_tf, output_file)    