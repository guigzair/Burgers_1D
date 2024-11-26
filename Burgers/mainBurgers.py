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

import core.Burgers.sandBox_Burgers as sandBox_Burgers


####################################   Parameters for GPU    ####################################################

gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)




################################## Get arguments from shell #################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dir', dest = 'log_dir', default = "./logs/Periodic")
parser.add_argument('--resFactor', dest ='resolutionFactor', type = int, default = 2)
parser.add_argument('--neighborhoodValue', dest ='neighborhoodValue', type = int, default = 3)
parser.add_argument('--npts', dest ='nptsVal', type = int, default = 512)
parser.add_argument('--stepSize', dest ='stepSize', type = int, default = 2)
parser.add_argument('--epochs', dest ='epochs', type = int, default = 20)
parser.add_argument('--batchSize', dest ='batchSize', type = int, default = 256)
parser.add_argument('--dataset', dest ='dataset', type = str, default = 'dataset_mixed_256_periodic_001.pkl')
# for periodic = dataset_mixed_512_001
args = parser.parse_args([])
print(args)




###############################    Training and dataset operations   ##################################################

log_dir = args.log_dir
resolutionFactor = args.resolutionFactor
neighborhoodValue = args.neighborhoodValue
nptsVal = args.nptsVal
stepSize = args.stepSize
epochs = args.epochs
batchSize = args.batchSize
dataset = args.dataset

hp = kt.HyperParameters()
class HyperModel(kt.HyperModel):
	def build(self, hp):
		lambda_ent = hp.Float(
			"lambda_ent",
			min_value=0,
			max_value=0.001,
			step=0.0001)
		lambda_reg = hp.Float(
			"lambda_reg",
			min_value=0,
			max_value=0.0001,
			step=0.00001)
		lambda_TVD = hp.Float(
			"lambda_TVD",
			min_value=0,
			max_value=0.0001,
			step=0.00001)
		activation = hp.Choice(
    		"activation", values = ["relu", "selu"])
		params = {"lambda_ent" : lambda_ent,
	    			"lambda_reg" : lambda_reg,
					"lambda_TVD" : lambda_TVD,
					"activation" : activation}
		smModel = sandBox_Burgers.My_Model(nptsVal,
							resolutionFactor,
							neighborhoodValue, 
							sandBox_Burgers.core_model, 
							params,
							stepSize = stepSize)
		smModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm = 10., clipvalue = 10.), # type: ignore
						loss ="mae")
		return smModel


hypermodel = HyperModel()
model = hypermodel.build(hp)


def scheduler(epoch):
	if epoch < 5:
		return 0.0005
	elif epoch < 10:
		return 0.0001
	else:
		return 0.00005
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)



with open('../../Data/Burgers/'+dataset,'rb+') as f:
	data = pickle.load(f)

data = utilities.Mean_coarsing(data, resolutionFactor)
# data = data[...,::resolutionFactor]
x_train, y_train = utilities.reshapeForTrainBurgers(data, resolutionFactor, stepSize = stepSize)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.cache()
train_dataset = train_dataset.batch(batchSize)
# train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# Prepare the validation dataset
with open('../../Data/Burgers/dataset_mixed_256_periodic_001_validation.pkl','rb+') as f:
	data = pickle.load(f)

data = utilities.Mean_coarsing(data, resolutionFactor)
# data = data[...,::resolutionFactor]

x_val, y_val = utilities.reshapeForTrainBurgers(data, resolutionFactor, stepSize = 15)




start_time = time.time()
# Create a callback for tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
														min_delta = 5.e-07, # type: ignore
														patience=2, 
														verbose = 1, 
														mode = 'min')

file_writer = tf.summary.create_file_writer(log_dir) # type: ignore

smModel = sandBox_Burgers.My_Model(nptsVal,
					resolutionFactor,
					neighborhoodValue, 
					sandBox_Burgers.core_model, 
					{"lambda_ent" : 0.00, 
      					"lambda_reg" : 0.0000,
					    "lambda_TVD" : 0.0001, 
						"activation" : "selu"},
					stepSize = stepSize)
smModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm = 10., clipvalue = 10.), # type: ignore
				loss ="mae")

# For fine tuning
# x = tf.random.uniform((100,256))
# a  = smModel(x, training=False)
#Load weights
# dir = './logs/Periodic/'
# smModel.load_weights(dir + 'weights_good.h5')

history = smModel.fit(train_dataset,
						epochs = epochs,
						batch_size=batchSize,
						verbose=1,
						shuffle=True,
						callbacks=[scheduler_callback],
						validation_data=(x_val, y_val))


print("--- %s seconds ---" % (time.time() - start_time))
smModel.save_weights(log_dir+'/weights_1d_100epochs.h5')


# tuner = kt.BayesianOptimization(
#     objective="val_loss",
#     max_trials=40,
#     executions_per_trial=4,
# 	beta = 6., 
#     hypermodel=HyperModel(),
#     project_name="custom_training",
#     overwrite=True,
# )

# tuner.search(train_dataset, epochs = 3, validation_data=(x_val, y_val))