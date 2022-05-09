""" data_generation.py

This is based on equations from [1].

Author: Rasul Choupanzadeh
Date: 05/09/2022

[1] David. M. Pozar. Microwave Engineering, 4th edition. John Wiley and Sons, 2011.

"""


import numpy as np
import matplotlib.pyplot as plt
import time

Pi = np.pi
mu = 4*Pi * 10**-7

A = 1
B = 1

a = 1.07 * 10**-2
b = 0.43 * 10**-2

def kc(a, b, m, n):
    res = np.sqrt(   ((m*Pi)/a)**2+ ((n*Pi)/b)**2 )
    return(res)

z = 0
beta = 0                ## beta is not zero in general, but for our problem which deals with Ex and assumes z=0, the effect of beta will be removed anyway!

def Ex(x, y, m, n, w):
    res = ((1j * w * mu * n * Pi) / (b*kc(a, b, m, n)**2) ) * A * np.cos (m* Pi* x/a)* np.sin(n*Pi*y/b)* np.exp(-1j*beta*z)
    return(res)

m_max = 3
n_max = 3

#incr = 50                                               # Number of samples for x and y
x_range = np.linspace(0, a, incr)
y_range = np.linspace(0, b, incr)

mag_Ex = np.zeros((m_max, n_max, incr, incr))
phase_Ex = np.zeros((m_max, n_max, incr, incr))

def generate_training_data(mag_Ex, phase_Ex,w):
    for m in range(0, m_max):
        for n in range(0, n_max):  

            for i in range(0, incr):
                for j in range(0, incr):   

                    if  n == 0:               # Defines Abs(Ex)=Angle(Ex)=0 for all m values & n=0 (even including m=n=0)==> we will remove m=n=0 instance later.
                        res = 0
                    else:

                        if j == incr-1:                 # Enforcing B.C for Ex field (Ex is a tangential field component for upper and lower side of W.G) ==> Ex must be zero at y=0 & y=b.
                            res = 0
                        else:
                            res = Ex(x_range[i], y_range[j], m, n, w)

                    mag_Ex[m, n, i, j] = np.abs(res)
                    phase_Ex[m, n, i, j] = np.angle(res)
                                       

start = time.time()
freq = np.linspace(13e9,17e9,8000)

dataset = np.zeros((8*len(freq),2*incr*incr+3))
nt = 0
print('Please wait till 8000')

for f in freq:
    f0 = f
    w0=2*Pi*f0                      
    generate_training_data(mag_Ex, phase_Ex,w0)                         # Generates training data for frequency w0 ==> it is in the form of [incr by incr] square matrix for each m & n values.
    
    mag_reshaped = mag_Ex.reshape(m_max,n_max, 1, incr*incr)            # Reshapes the training data (for frequency w0) in the form of one-dimensional array for each m & n values
    phase_reshaped = phase_Ex.reshape(m_max,n_max, 1, incr*incr)
    
    X_mag = np.zeros((m_max*n_max,incr*incr))                           # Defines a zero matrix with q*p dimensions to store all training data, where q=number of instances for w0 frequency & p=number of features for each instance
    X_phase = np.zeros((m_max*n_max,incr*incr))
    Y = np.zeros((m_max*n_max,2))                                       # Defines a zero matrix with q*2 dimenstion to store labels (m & n values) of training data, where q=number of instances for w0 frequency
    Y1 = np.zeros((m_max*n_max,1))
    
    for m in range(0, m_max):  
        for n in range(0, n_max):
            X_mag[m*n_max+n,:] = mag_reshaped[m, n]
            X_phase[m*n_max+n,:] = phase_reshaped[m, n]
            Y[m*n_max+n,:] = m, n
            Y1[m*n_max+n,:] = m*n_max+n
            
    X = np.hstack((X_mag, X_phase))                                     # Horizontally stacks the X_mag & X_phase to create a training dataset (for w0 frequency) containing all features together (mag features and phase features).              
    dataset_w0 = np.hstack((X, Y))                                      # Horizontally stacks the features & labels to create a dataset_w0 (we will need this matrix to shuffle data later).     
    dataset_w0 = np.hstack((dataset_w0, Y1))  
    dataset_w0 = np.delete(dataset_w0, (0), axis=0)                     # Removes the instances of m=n=0
    dataset[nt*8:(nt+1)*8,:] = dataset_w0
    nt = nt+1
    print(nt)

end = time.time()
print("The execution time of data generation is :", end-start)

np.save('dataset.npy', dataset)
