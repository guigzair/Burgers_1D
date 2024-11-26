import tensorflow as tf
import numpy as np
import pickle

def periodic_padding_flexible(tensor, axis,padding_left=1, padding_right = 1):
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding_left,int):
        padding_left = (padding_left,)
    if isinstance(padding_right,int):
        padding_right = (padding_right,)
    ndim = len(tensor.shape)
    for ax,p,p_r in zip(axis,padding_left, padding_right):
        ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p_r) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right,middle,left], axis=ax)

    return tensor

def initFuncSquare(x,x0=0.1,x1=0.3, H = 1., h = 0.):
    if x>= x0 and x <= x1:
        return H
    return h

def initFuncSquare2(x,x0=0.1,x1=0.3, H = 1., h = 0.):
    return np.where(np.logical_and(x>= x0 , x<= x1 ), H, h)

def initFuncSinus(x,x0=0.1,x1=0.3, H = 1., h = 0.):
    if x>= x0 and x <= x1:
        return (1 - np.cos(2*np.pi*(x-x0)/(x1-x0)))*(H-h)/2 + h
    return h

def complex_initial(x):
    if x <= 3.5/8. and x > 3./8.:
        return 3.
    elif x > 3.5/8. and x <= 4./8.:
        return 1.
    elif x > 4./8. and x <=4.5/8:
        return 3.
    elif x > 4.5/8. and x <=5/8:
        return 2.
    else :
        return tf.sin(8* np.pi * x)	

def random_sin(x, A, phi, l):
    return np.sum(A * np.sin(2 * np.pi * l * x + phi))/3

def random_sin_Advection(x, A, phi, l):
    a = A * tf.sin(2 * np.pi * l * x + phi)
    return tf.reduce_sum(a, axis = 0)/3

def random_sin_Euler(x, A, phi, l):
    return np.sum(A * (np.sin(2 * np.pi * l * x + phi)+1.))/5

def Shock_entropy_Euler(x):
    if x<0.1:
        return 3.857143, 2.629369, 10.3333
    else:
        return 1. + 0.2 * np.sin(5*x), 0 , 1

def Mean_coarsing(data, resolutionFactor, axis = -1):
    shape = tf.shape(data).numpy()
    shape[-1] = shape[-1]//resolutionFactor
    shape = np.append(shape,resolutionFactor)
    return tf.reduce_mean(tf.reshape(data, shape),axis)

def reshapeForTrain(data, resolutionfactor, stepSize = 10):
    data_train = []
    # To take same time step between coarse and fine
    data = data[:,::resolutionfactor,:]
    for bCase in data:
        # Get right size for the stepSize and making sure the shape can be divided by stepSize
        bCase = bCase[:bCase.shape[0]//stepSize*stepSize,:]
        # Divide in tensor of size stepSize
        bCase = tf.split(bCase, bCase.shape[0]//stepSize)
        data_train.append(bCase)
    data_train = tf.convert_to_tensor(data_train)
    data_train = tf.reshape(data_train, (data.shape[0]*data.shape[1]//stepSize, stepSize, data.shape[2]))
    return data_train

def reshapeForTrainBurgers(data, resolutionFactor, stepSize = 10):
    data = tf.transpose(data, [1,0,2])
    train_output = tf.image.extract_patches(images=data[...,tf.newaxis],
                        sizes=[1, stepSize * resolutionFactor,1 , 1],
                        strides=[1, 1, 1, 1],
                        rates=[1, 1, 1, 1],
                        padding='VALID')
    print(train_output.shape)
    train_output = train_output[...,::resolutionFactor]
    train_output = tf.reshape(train_output, (train_output.shape[0]*train_output.shape[1], data.shape[-1], stepSize))
    train_input = train_output[:,:,0]
    train_input = tf.cast(train_input, tf.float32)
    train_output = tf.cast(train_output, tf.float32)
    return train_input, train_output


def reshapeForTrain2(data, resolutionFactor, stepSize = 10):
    train_output = tf.image.extract_patches(images=data[...,tf.newaxis],
                        sizes=[1, stepSize * resolutionFactor,1 , 1],
                        strides=[1, 1, 1, 1],
                        rates=[1, 1, 1, 1],
                        padding='VALID')
    train_output = train_output[:,:,:,::resolutionFactor]
    train_output = tf.reshape(train_output, (train_output.shape[0]*train_output.shape[1], data.shape[-1], stepSize))
    train_input = train_output[:,:,0]
    train_input = tf.cast(train_input, tf.float32)
    train_output = tf.cast(train_output, tf.float32)
    return train_input, train_output

# u = tf.random.uniform((100,5,64))
# u_input, u_ouput = reshapeForTrain2(u, 2, stepSize=2)

def reshapeForTrainEuler(data, resolutionFactor, stepSize = 10):
    dataEuler = tf.transpose(data, [1,0,2,3])
    rho = dataEuler[...,0,:]
    u = dataEuler[...,1,:]
    p = dataEuler[...,2,:]
    rho_input, rho_output = reshapeForTrain2(rho,resolutionFactor,stepSize = stepSize)
    u_input, u_output = reshapeForTrain2(u,resolutionFactor,stepSize = stepSize)
    p_input, p_output = reshapeForTrain2(p,resolutionFactor,stepSize = stepSize)

    input = tf.stack([rho_input, u_input, p_input], axis=1)
    output = tf.stack([rho_output, u_output, p_output], axis=1)
    return input, output
    



def get_time_derivative(data):
    dt = 0.5/512 # CFL*dx
    data_deriv = (tf.roll(data, -1, 1) - data)/dt
    data_deriv =data_deriv[:,:-1,:]
    return data_deriv


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


def nonPeriodic_padding_flexible(tensor, axis, paddings):
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(paddings,int):
        paddings = (paddings,)
    for ax,p in zip(axis,paddings):
        pad = np.zeros((len(tensor.shape),2)).astype(np.int)
        pad[ax] = pad[ax] + 1
        for _ in range(p):
            tensor = tf.pad(tensor,pad, "SYMMETRIC")
    return tensor

def constant_padding_flexible(tensor, axis, paddings, padding_left=1, padding_right = 1):
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(paddings,int):
        paddings = (paddings,)
    # On the left
    for ax,p in zip(axis,paddings):
        pad = np.zeros((len(tensor.shape),2))
        pad[ax,0] =  p
        tensor = tf.pad(tensor,pad, "CONSTANT", constant_values = padding_left)
    # On the right
    for ax,p in zip(axis,paddings):
        pad = np.zeros((len(tensor.shape),2))
        pad[ax,1] =  p
        tensor = tf.pad(tensor,pad, "CONSTANT", constant_values = padding_right)
    return tensor

tensor = tf.random.uniform((3,5))
# tf.pad(tensor, [[0,0], [0,0], [1,2], [0,0]], "CONSTANT", constant_values = 4.5)

# constant_padding_flexible(tensor, -1 ,2, padding_right=2, padding_left=1)

# nonPeriodic_padding_flexible(tensor, -1, 2)

def reflexive_padding_flexible(tensor, axis, padding = 1):
    NB_axis = tf.rank(tensor)
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        paddings = (padding,)
    pad_list = tf.zeros((NB_axis, 2)) 
    for ax in axis:
         pad_list[ax] = [padding,padding]

    Mask = tensor * 0 + 1.
    Mask = tf.pad(Mask, pad_list,"CONSTANT")
    Mask = tf.cast(Mask, tf.float32)
    tensor_pad = tf.pad(tensor, pad_list,"SYMMETRIC")
    u_pad = tensor_pad[:,1,...]
    pad_u = (1 - Mask)[:,1,...]*u_pad
    pad_u_list = pad_list[0] + pad_list[1:]
    u = tf.pad(tensor[:,1,...], pad_u_list,"CONSTANT") - pad_u
    tensor = tf.stack([tensor_pad[:,0,...], u , tensor_pad[:,2,...]], 1)
        
    return tensor



# tensor = tf.random.uniform((5,3,5))
# reflexive_padding_flexible(tensor, (2), padding = 2)

def reflexive(tensor, axis = -1, side = 0, pad = 1, gamma = 1.4):
    ref_state = tf.split(tensor, [pad, tensor.shape[axis]-pad][::side*2+1], axis = axis)
    ref_state = ref_state[side]
    
    rho_b = ref_state[:,0] 
    u_b = (-1) * ref_state[:,1] 
    p_b = ref_state[:,2] 
    Primitives_b = tf.stack([rho_b, u_b, p_b],axis = 1)
    Primitives_b = tf.reverse(Primitives_b, [axis])

    Primitives = tf.concat([Primitives_b, tensor][::side*2+1],axis = axis)
    return Primitives

def reflexive2(tensor, axis = -1, side = 0):
    ref_state = tf.split(tensor, [1,1, tensor.shape[axis]-2][::side*2+1], axis = axis)
    ref_state_2 = ref_state[1]
    ref_state_1 = ref_state[side]
    
    rho_b_1 = ref_state_1[:,0] 
    u_b_1 = (-1) * ref_state_1[:,1] 
    p_b_1 = ref_state_1[:,2] 

    rho_b_2 = ref_state_2[:,0] 
    u_b_2 = (-2) * ref_state_1[:,1] 
    p_b_2 = ref_state_2[:,2] 
    Primitives_b_1 = tf.stack([rho_b_1, u_b_1, p_b_1],axis = 1)
    Primitives_b_2 = tf.stack([rho_b_2, u_b_2, p_b_2],axis = 1)

    Primitives = tf.concat([Primitives_b_2,Primitives_b_1, tensor][::side*2+1],axis = axis)
    return Primitives

def pad_ML_reflexive(tensor, axis = -1, pad = 1):
    tensor = reflexive(tensor, axis = axis, side = 0, pad = pad)
    tensor = reflexive(tensor, axis = axis, side = -1, pad = pad)
    return tensor

t = tf.random.uniform((2,3,8,1))
pad_ML_reflexive(t,axis = 2, pad = 2)


