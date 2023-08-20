# -*- coding: utf-8 -*-
"""
Created in August 2023
@author: LC
"""
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt



# =============================================================================
# Define state space matrices
# =============================================================================

# state transition matrix (computed elsewhere)
A = np.array([[ 9.87610634e-01,  7.42403599e-03,  9.88586175e-03, 5.39785161e-05],
              [ 7.42403599e-03,  9.92559992e-01,  5.39785161e-05, 9.92184743e-03],
              [-2.46336866e+00,  1.47478248e+00,  9.73143171e-01,1.32051843e-02],
              [ 1.47478248e+00, -1.48018034e+00,  1.32051843e-02,9.81946628e-01]])

B = np.array([[1.59724420e-07],
       [4.97597806e-05],
       [5.39785161e-05],
       [9.92184743e-03]])

#output matrix (computed elsewhere)
H = np.array([[ 150. , -150.  , 0.59064779, -1.07290968]])

J = np.array([[1.]])


#number of states
n_states = len(A)


t_end = 10
dt =   0.01


nsteps = int(t_end/dt)

time_array = np.arange(nsteps)*dt


A_f1 = 300
omega_f1 = 10



Ft = np.zeros((1,nsteps))
Ft += A_f1 * np.sin(omega_f1*time_array)

Ft[0,int(10/dt):] *= 0


z_vec = np.zeros([4,nsteps])
measurement_acc2 = np.zeros([len(H),nsteps])



y0 = np.array([0,0])
ydot0 = np.array([0,0])

z_vec[:,0] = np.concatenate((y0,ydot0))
measurement_acc2[:,0] = H@z_vec[:,0]



for k in range(0,nsteps-1):
    z_vec[:,k+1] = A@z_vec[:,k] + B@Ft[:,k]
    measurement_acc2[:,k+1] = H@z_vec[:,k+1] + J@Ft[:,k+1]


##add noise to measurement
n_meas = len(H)


d = np.zeros_like(measurement_acc2)
R_true = []

for i in range(n_meas):
    
    std_meas = np.std(measurement_acc2[i,:])
    noise = np.random.normal(scale=0.01*std_meas,size=measurement_acc2[i,:].shape)
    d[i,:] = measurement_acc2[i,:]+noise
    R_true.append(np.var(noise))



R = np.diag(R_true)






#%% DKF

def dkf(z_prev, Pz_prev,x_prev, Px_prev,
                  A,B,H,J,
                  Q,S,R,
                  d):

    #### Stage 1 - input estimation

    #input predition
    xpload = x_prev
    Ppload = Px_prev + S
    
    
    #Kalman Gain input
    Kload = Ppload @ J.T @ np.linalg.inv( J @ Ppload @ J.T + R)

    # input estimation
    Pload = Ppload - Kload @ J @ Ppload   
    xload = xpload+Kload@(d - H@z_prev - J@xpload)

    #### Stage 2 - state estimation
    
    #state prediction    
    xpdual = A @ z_prev + B@xload
    Ppdual = A @ Pz_prev @ A.T + Q
    
    #Kalman Filter state
    K = Ppdual @ (H.T) @np.linalg.inv(H@Ppdual@H.T + R)
    
    
    #state estimation
    xfdual = xpdual + K @ (d - H@xpdual - J@xload)
    Pfdual =  Ppdual - K @ H @ Ppdual

    return xfdual, Pfdual,xload, Pload


n_states = len(A)

#process noise and input process noise
Q = 1e-10*np.eye(n_states)
S = 1e5*np.eye(1)




#initialize estimated state and load vectors
z_dkf = np.zeros([n_states,nsteps])
p_dkf = np.zeros([1,nsteps])


# initialize covariance matrices
P = np.eye(n_states) * 0
Pp = np.eye(1)*0


for k in range(1,nsteps):
    z_dkf[:,k],P, p_dkf[:,k],Pp = dkf(z_dkf[:,k-1], P,
                                      p_dkf[:,k-1], Pp,
                                      A,B,H,J,
                                      Q,S,R,
                                      d[:,k])


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(13,7))

ax1.set_title("disp DOF 1")
ax2.set_title("disp DOF 2")

ax1.plot(time_array,z_vec[0,:],label="True")
ax1.plot(time_array,z_dkf[0,:],
         label="Dual Kalman Filter",color="orange")

ax2.plot(time_array,z_vec[1,:],label="True")
ax2.plot(time_array,z_dkf[1,:],
         label="Dual Kalman Filter")

for ax in (ax1,ax2):
    ax.legend()
    ax.grid()
ax2.set_xlabel("time")
plt.show()



#  %%load

    
fig, (ax1) = plt.subplots(1,1,figsize=(13,7))

ax1.set_title("Load")
ax1.plot(time_array,Ft[0,:],label="True")
ax1.plot(time_array,p_dkf[-1,:],
         label="Dual Kalman Filter",color="orange")

ax1.legend()
ax1.grid()
# ax.set_xlim([0,0.4])

# ax1.set_ylim([-0.7,0.7])
ax1.set_xlabel("time (s)")
plt.show()
