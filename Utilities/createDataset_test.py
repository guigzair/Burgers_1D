import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
size = 14
params = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'cm',  # Computer Modern font
	'legend.fontsize':size,
    'axes.labelsize' : size,
    'xtick.labelsize' : size+1,
    'ytick.labelsize' : size+1
}
plt.rcParams.update(params)
with open('../../Data/Burgers/dataset_mixed_256_periodic_001.pkl','rb+') as f:
    data = pickle.load(f)



# For burgers
data = tf.transpose(data, [1,0,2])
elem = 0
while elem<50:
    x = tf.linspace(0,1,256)
    u = data[elem]
    fig, ax = plt.subplots(dpi = 500)
    ax.plot(x,data[elem, 0],label=r"Initial")
    ax.plot(x,data[elem, -1],label=r"Final")
    # C = tf.sqrt(1.4 *data[-1, elem, 2]/data[-1, elem, 0])
    # ax.plot(x,tf.abs(data[-1, elem, 1])+C,label=r"u+c")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u$")
    ax.grid()
    ax.legend()
    elem = elem + 1


# For Euler
# C = tf.sqrt(1.4 *data[:,:,2]/data[:,:,0])
# max = tf.reduce_max(data[:,:,1] + tf.sqrt(1.4 *data[:,:,2]/data[:,:,0]))
# var = 1
# elem = 1
# while elem<15:
#     x = tf.linspace(0,1,512)
#     u = data[elem]
#     fig, ax = plt.subplots(dpi = 500)
#     ax.plot(x,data[0, elem, var],label=r"$Initial$")
#     # ax.plot(x,data[0, elem, 1],label=r"$u$")
#     # ax.plot(x,data[0, elem, 2],label=r"$P$")
#     ax.plot(x,data[-1, elem, var],label=r"Final")
#     C = tf.sqrt(1.4 *data[:, elem, 2]/data[:, elem, 0])
#     CFL_max = tf.reduce_max((tf.abs(data[:, elem, 1])+C)*0.1, -1)
#     CFL_max = tf.reduce_max(CFL_max, -1)
#     print(CFL_max)
#     ax.set_xlabel(r"$x$")
#     ax.set_ylabel(r"$u$")
#     ax.grid()
#     ax.legend()
#     elem = elem + 1
    
    
# For Advection
# data = tf.transpose(data, [1,0,2])
# elem = 0
# while elem<15:
#     x = tf.linspace(0,1,512)
#     u = data[elem]
#     fig, ax = plt.subplots()
#     ax.plot(x,data[elem, 0],label=r"début")
#     ax.plot(x,data[elem, -1],label=r"fin")
#     ax.set_xlabel(r"$x$")
#     ax.set_ylabel(r"$u$")
#     ax.grid()
#     ax.legend()
#     elem = elem + 1

