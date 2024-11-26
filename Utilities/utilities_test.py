import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import utilities

with open('./Data/Advection/dataset_square_256_01.pkl','rb+') as f:
    data = pickle.load(f)

# data_train = utilities.reshapeForTrain(data, 4, stepSize = 10)


train_input2, train_output2 = utilities.reshapeForTrain2(data,4,stepSize = 10)


y = tf.linspace(0,1, 400)
y = tf.reshape(y, (20,20))
utilities.Mean_coarsing(y, 4)

data_meancoarse = utilities.Mean_coarsing(data, 4)


y = tf.linspace(0,10,11)
y = tf.cast(y, dtype = tf.float32)
y = utilities.periodic_padding_flexible(y, 0, padding_left = 2, padding_right = 3)



##########################   Euler   ##################################
with open('../../Data/Euler/dataset_512.pkl','rb+') as f:
    dataEuler = pickle.load(f)

mean = utilities.Mean_coarsing(dataEuler, 4)

train_inputEuler, train_outputEuler = utilities.reshapeForTrainEuler(dataEuler,4,stepSize = 10)