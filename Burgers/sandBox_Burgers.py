from re import U
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import numpy as np
import sys
from core.Utilities import utilities
import pickle
import time
import argparse
import matplotlib.pyplot as plt
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "Helvetica"
})

import BurgersScheme

# To debug 
# tf.config.experimental_run_functions_eagerly(True)

################################## Get arguments from shell #################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dir', dest = 'log_dir', default = "./logs")
parser.add_argument('--resFactor', dest ='resolutionFactor', type = int, default = 8)
parser.add_argument('--neighborhoodValue', dest ='neighborhoodValue', type = int, default = 5)
parser.add_argument('--npts', dest ='nptsVal', type = int, default = 512)
parser.add_argument('--stepSize', dest ='stepSize', type = int, default = 10)
parser.add_argument('--epochs', dest ='epochs', type = int, default = 1)
parser.add_argument('--batchSize', dest ='batchSize', type = int, default = 256)
parser.add_argument('--dataset', dest ='dataset', type = str, default = 'dataset_nonPeriodic_512.pkl')


args = parser.parse_args([])
print(args)

###################################"   Define metrics and functions    ################################################
loss_tracker = keras.metrics.Mean(name="loss")
val_tracker = keras.metrics.Mean(name="val_loss")
norm_u_tracker = keras.metrics.Mean(name="norm_u")
Loss_ent_tracker = keras.metrics.Mean(name="Loss_ent")
Loss_TVD_tracker = keras.metrics.Mean(name="Loss_ent")
norm_deriv_tracker = keras.metrics.Mean(name="norm_deriv")
mae_metric = keras.metrics.MeanSquaredError(name="mse")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def polynomial_accuracy_layer(x, stencil_size = 3,constraint_order = 1, derivative = 0):
	stencil = np.arange(stencil_size) - (stencil_size) // 2
	stencil = stencil

	# create the bias vector, with true coefficients
	A = []
	for i in range(stencil_size):
		A.append(stencil**i)
	A = np.array(A)
	b = []
	for i in range(stencil_size):
		if i == derivative:
			b.append(np.math.factorial(0))
		else : 
			b.append(0)
	b = np.array(b)
	alpha_p = np.linalg.solve(A,b)
	# alpha_p = np.array([0.,0.,-1.,1.,0.])
	# alpha_p = np.array([0.,-1.,1.])
	# Create the second vector with 
	# print(constraint_order)
	A_constraint = A[:constraint_order]
	# print(A_constraint)
	_, _, vh = np.linalg.svd(A_constraint)
	# print(vh[constraint_order:])
	# print(x[constraint_order:])
	alpha_null = tf.tensordot(x[...,:],vh[constraint_order:].astype(np.float32), axes=[-1, 0])
	alpha = alpha_p + alpha_null
	return alpha

x = np.array([[0.,0.]])
# result = polynomial_accuracy_layer(x.astype(np.float32),stencil_size = 3, constraint_order = 1, derivative = 1)


def setU(u, coeffs, neighborhoodValue, dx,  derivative_x = 0):
	# coeffs = coeffs[...,2:-2,:]
	pad = neighborhoodValue//2
	u = utilities.periodic_padding_flexible(u , axis = 1, padding_left = pad, padding_right = pad)
	# coeffs = tf.pad(coeffs, [[0,0], [0,0], [2,2], [0,0]], "CONSTANT")
	# u = utilities.nonPeriodic_padding_flexible(u, 2, neighborhoodValue//2)
	# u = tf.pad(u, [[0,0],[0,0],[pad,pad]],"SYMMETRIC")
	# u = utilities.reflexive_padding_flexible(u, 2, padding = 2)
	# u = utilities.pad_ML_reflexive(u, axis = -1,pad = pad * c_pad)
	u = u[...,tf.newaxis]
	u2 = tf.image.extract_patches(images=u[..., tf.newaxis],
						   sizes=[1, neighborhoodValue, 1 , 1],
						   strides=[1, 1, 1, 1],
						   rates=[1, 1, 1, 1],
						   padding='VALID')
	coeffs = polynomial_accuracy_layer(coeffs, 
										stencil_size = neighborhoodValue, 
										constraint_order = 1, 
										derivative = derivative_x)
	coeffs = coeffs[...,tf.newaxis, :]
	u = tf.reduce_sum(u2*coeffs, -1)
	u = u[...,0]
	return u / dx**derivative_x

# u = tf.random.uniform((10,64))
# coeffs = tf.zeros((10,64,2))
# setU(u, coeffs, 3, 1/64, derivative_x=0) 


def time_step(u, coeffs, neighborhoodValue, dx, gamma = 1.4, CFL = 0.01):
	u_old = u
	# Reconstruct variables
	# primitives = setU(primitives_old, coeffs[0], neighborhoodValue, dx)
	grad_x = setU(u_old, coeffs, neighborhoodValue, dx, derivative_x=1)


	# minmod
	aa = tf.minimum(tf.sign(tf.roll(grad_x,0,-1)),0.0) * 2.0 + 1.0
	ri = tf.roll(grad_x,1,-1)/(tf.maximum(tf.abs(tf.roll(grad_x,0,-1)) ,1e-15) * aa)
	# phi= tf.maximum(tf.minimum(ri,1.0),0.0) # minmod
	phi= (ri**2 + ri)/(ri**2 + 1) # Van albada

	u_L = tf.roll(u, 0, axis = -1) + 1./2.*dx*phi*tf.roll(grad_x,0,-1)
	u_R =  tf.roll(u, -1, axis = -1) - 0.5 * dx * tf.roll(phi,-1,-1) * tf.roll(grad_x, -1, axis = -1)

	# flux and average
	fconv = 0.5 * (u_L**2/2 + u_R**2/2)
	# Add stabilizing diffusive terms
	fconv   -=  0.5 * tf.abs(u_L+u_R)/2 * (u_R - u_L)

	# finite volume
	nu = 0.
	fdiff = -nu * grad_x
	flux = fconv + fdiff
	div = flux - tf.roll(flux,1,-1)

	u = u_old - CFL*div


	return u




def rescale_01(array, axis):
	arrayMax = tf.reduce_max(array, axis, keepdims=True)
	arrayMin = tf.reduce_min(array, axis, keepdims=True)
	return (array - arrayMin)/tf.maximum(arrayMax - arrayMin, 0.0001)

def rescale_to_range(inputs, min_value, max_value, axes):
  inputs_max = tf.reduce_max(inputs, axis=axes, keepdims=True)
  inputs_min = tf.reduce_min(inputs, axis=axes, keepdims=True)
  scale = (inputs_max - inputs_min) / (max_value - min_value)
  return (inputs - inputs_min) / tf.maximum(scale, 0.0001) + min_value

def Standardization(array, axis):
	arrayMean = tf.reduce_mean(array, axis, keepdims=True)
	arraySTD = tf.math.reduce_std(array, axis, keepdims=True)
	return (array - arrayMean)/tf.maximum(arraySTD, 0.0001)

def TVD(u, axis):
	# En entree une vecteur de taille [None, 3, N] et en sortie [None, 3]
	TV = tf.roll(u, -1, axis)-u
	TV = tf.reduce_sum(tf.abs(TV), axis)
	return TV

def Loss_TVD(Prim, axis):
	# En entree un vecteur de taille [None, 3, N, T] et en sortie [None, 3]
	# [None, 3, T]
	Loss = TVD(Prim, axis)
	# print(Loss.shape)
	Loss =  Loss - tf.roll(Loss, 1, -1)
	# print(Loss[...,1:].shape)
	Loss = tf.reduce_sum(tf.maximum(Loss[...,1:],0)**2, -1)
	
	return Loss


def Loss_entropy(Primitives, gamma = 1.4, axis = -1):
	u = Primitives[:,...]

	eta = u ** 2
	q = 2/3 * u**3
	# print(q.shape)

	Dx_q = q - tf.roll(q, 1, 1)
	# print(Dx_q.shape)
	Dt_eta = eta - tf.roll(eta, 1, -1)
	Loss = tf.maximum(Dt_eta[...,2:-2,1:] + 0.01 * Dx_q[...,2:-2,1:], 0)**2
	Loss = tf.reduce_sum(Loss, -1)
	Loss = tf.reduce_sum(Loss, -1)
	# print(Loss.shape)
	return Loss

a = tf.linspace(1,32,10)
a = tf.random.uniform((3,32,10))
Loss_entropy(a, axis = -2)


############################################################################################################################
############################################################################################################################
################################################   Model ###################################################################
############################################################################################################################
############################################################################################################################

class My_Model(tf.keras.Model):
	def __init__(self, numPoints,resolutionFactor, neighborhoodValue, model_func, params, stepSize = 10):
		#Input will be numPoints//resolutionFactor sized
		self.resolutionFactor = resolutionFactor
		self.neighborhoodValue = neighborhoodValue
		self.numPoints = numPoints
		self.dx = self.resolutionFactor / self.numPoints
		self.stepSize = stepSize
		self.stepSize_test = 15
		self.polynomial_accuracy = 1
		self.params = params
		super(My_Model, self).__init__()
		self.core_model = model_func(self.neighborhoodValue, polynomial_accuracy = self.polynomial_accuracy, activation_fct = self.params["activation"])
		self.epoch = 0

	def call(self, Prim, dx = 1/64):
		y_pred = rescale_to_range(Prim, -1., 1., -1)
		# y_pred = Standardization(Prim, -1)
		y_pred = y_pred[..., tf.newaxis]
		coeffs = self.core_model(y_pred)
		# print(tf.shape(coeffs))
		# coeffs = tf.split(coeffs, [self.neighborhoodValue - self.polynomial_accuracy,
		# 	     					self.neighborhoodValue - self.polynomial_accuracy], axis = -1)
		Prim_pred = time_step(Prim, coeffs, self.neighborhoodValue, dx, CFL = 0.01)
		return Prim_pred

	def get_coeffs(self, Prim):
		y_pred = rescale_01(Prim, -1)
		y_pred = y_pred[..., tf.newaxis]
		coeffs = self.core_model(y_pred)
		coeffs_U = polynomial_accuracy_layer(coeffs, 
									stencil_size = self.neighborhoodValue, 
									constraint_order = 1, 
									derivative = 1)
		return coeffs_U
	
	def get_limiter(self, Prim):
		y_pred = rescale_01(Prim, -1)
		y_pred = y_pred[..., tf.newaxis]
		coeffs = self.core_model(y_pred)

		grad_x = setU(Prim, coeffs, self.neighborhoodValue, self.dx, derivative_x=1)
		grad_x = tf.pad(grad_x, [[0,0],[0,0],[1,1]],"SYMMETRIC")
		# x direction
		aa = tf.minimum(tf.sign(tf.roll(grad_x,0,-1)),0.0) * 2.0 + 1.0
		ri = tf.roll(grad_x,1,-1)/(tf.maximum(tf.abs(tf.roll(grad_x,0,-1)) ,1e-6) * aa)
		# phi= tf.maximum(tf.minimum(ri,1.0),0.0) # minmod
		phi= (ri**2 + ri)/(ri**2 + 1) # Van albada
		a = phi[...,1:-1] * grad_x[...,1:-1]
		return a * self.dx

	
	def train_step(self, data):
		norm_u_tracker.reset_states()
		x, y = data
		loss = 0
		prim_pred = x
		with tf.GradientTape() as tape:
			# Initialization of the tensor
			PRIM = prim_pred[...,tf.newaxis]
			for t in range(self.stepSize-1):
				# Forward pass
				prim_pred = self(prim_pred, training=True) 
				PRIM = tf.concat([PRIM, prim_pred[...,tf.newaxis]], axis = -1)
			# Regularisation
			Loss_reg = tf.math.reduce_mean([self.core_model.layers[i].losses for i in [1,4,7]])
			#  + 0.0005 * loss_reg
			#TVD 
			loss_TVD = Loss_TVD(prim_pred, -2)
			# + 0.001 * loss_TVD
			# Loss entropique 
			Loss_ent = Loss_entropy(PRIM, gamma = 1.4, axis = -2)
			#  + 0.0001 *  Loss_ent

			# Computing loss
			a = tf.linspace(0.5,1.5,self.stepSize)
			a = a[tf.newaxis,...][tf.newaxis,...][tf.newaxis,...]
			loss = keras.losses.mean_absolute_error(y[:], PRIM[:])*10/self.stepSize + keras.losses.mean_squared_error(y[:], PRIM[:])*10/self.stepSize
			loss = tf.reduce_mean(tf.reduce_mean(loss, -1), -1) + \
									self.params["lambda_ent"] * Loss_ent + \
							  		self.params["lambda_reg"] * Loss_reg + \
									self.params["lambda_TVD"] * loss_TVD 
		# print("loss = " + str(loss))

		# Compute gradients
		trainable_vars = self.trainable_variables 
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Compute our own metrics
		loss_tracker.update_state(loss)
		mae_metric.update_state(y, PRIM[:,:,:])
		coeffs_u = self.get_coeffs(prim_pred)
		norm_u_tracker.update_state(tf.norm(coeffs_u,1)/(512*coeffs_u.shape[1]*coeffs_u.shape[2]))
		# norm_deriv_tracker.update_state(tf.norm(coeffs_deriv,1))
		Loss_ent_tracker.update_state(Loss_ent)
		Loss_TVD_tracker.update_state(Loss_reg)
		# Computing norm of the gradient
		gradient_norm = 0
		for grad in gradients:
			gradient_norm = gradient_norm + tf.norm(grad)
		return {"loss": loss_tracker.result(), 
				"mse": mae_metric.result()*10/self.stepSize, 
				"gradient norm": gradient_norm, 
				"norm u":norm_u_tracker.result(), 
				"Loss_TVD" :Loss_TVD_tracker.result(),  
				"Loss ent" :Loss_ent_tracker.result()}



	def test_step(self, data):
		x, y = data
		prim_pred = x
		loss = 0
		PRIM = prim_pred[...,tf.newaxis]
		for _ in range(self.stepSize_test-1):
			prim_pred = self(prim_pred, training=False) 
			PRIM = tf.concat([PRIM, prim_pred[...,tf.newaxis]], axis = -1)
		loss = keras.losses.mean_absolute_error(y[:], PRIM[:])*10/self.stepSize_test
		loss = tf.reduce_mean(tf.reduce_mean(loss, -1), -1)
		val_tracker.update_state(loss)
		return {"loss": val_tracker.result()}


def core_model(neighboorhoodvalue, polynomial_accuracy = 1, activation_fct = "selu"):
	pad = neighboorhoodvalue//2
	model = tf.keras.Sequential()
	model.add(keras.layers.Lambda(lambda x : utilities.periodic_padding_flexible(x , axis = 1, padding_left = pad, padding_right = pad)))
	# model.add(keras.layers.Lambda(lambda x : tf.pad(x, [[0,0],[pad,pad],[0,0]],"SYMMETRIC")))
	# model.add(keras.layers.Lambda(lambda x : utilities.pad_ML_reflexive(x, axis = 2,pad = pad)))
	model.add(keras.layers.Conv1D(32, neighboorhoodvalue, activation = activation_fct, kernel_regularizer=tf.keras.regularizers.L1(0.0001)))
	model.add(keras.layers.Lambda(lambda x : rescale_to_range(x, -1., 1., 2)))

	model.add(keras.layers.Lambda(lambda x : utilities.periodic_padding_flexible(x , axis = 1, padding_left = pad, padding_right = pad)))
	# model.add(keras.layers.Lambda(lambda x : tf.pad(x, [[0,0],[pad,pad],[0,0]],"SYMMETRIC")))
	# model.add(keras.layers.Lambda(lambda x : utilities.pad_ML_reflexive(x, axis = 2,pad = pad)))
	model.add(keras.layers.Conv1D(32, neighboorhoodvalue, activation = activation_fct, kernel_regularizer=tf.keras.regularizers.L1(0.0001)))
	model.add(keras.layers.Lambda(lambda x : rescale_to_range(x, -1., 1., 2)))

	model.add(keras.layers.Lambda(lambda x : utilities.periodic_padding_flexible(x , axis = 1, padding_left = pad, padding_right = pad)))
	# model.add(keras.layers.Lambda(lambda x : tf.pad(x, [[0,0],[pad,pad],[0,0]],"SYMMETRIC")))
	# model.add(keras.layers.Lambda(lambda x : utilities.pad_ML_reflexive(x, axis = 2,pad = pad)))
	model.add(keras.layers.Conv1D((neighboorhoodvalue - polynomial_accuracy), neighboorhoodvalue, kernel_regularizer=tf.keras.regularizers.L1(0.0001)))
	return model

model = core_model(3)
model.build((100,256,1))
model.summary()

# a = tf.random.uniform((10,3,64,1))
# b = rescale_to_range(a, -1., 1., 2)

# x = tf.random.uniform((100,3,64,1))
# model = core_model(3)
# y = model(x)
# tf.shape(y)

# smModel = My_Model(512,
# 					8,3, 
# 					core_model, 
# 					stepSize = 10)
# smModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm = 10., clipvalue = 10.),
# 				loss ="mae")
# x = tf.random.uniform((100,3,512//8))
# a  = smModel(x, training=True)


# train_dataset = tf.data.Dataset.from_tensor_slices((tf.random.uniform((100,3,64)), tf.random.uniform((100,3,64,10))))
# train_dataset = train_dataset.batch(15)
# smModel.train_step(next(iter(train_dataset)))

# stencil_size = 3
# stencil = np.arange(stencil_size) - (stencil_size) // 2
# stencil = stencil + 1

# A = []
# for i in range(stencil_size):
# 	A.append(stencil**i)
# A = np.array(A)
# b = []
# for i in range(stencil_size):
# 	if i == 1:
# 		b.append(np.math.factorial(0))
# 	else : 
# 		b.append(0)
# b = np.array(b)
# alpha_p = np.linalg.solve(A,b)