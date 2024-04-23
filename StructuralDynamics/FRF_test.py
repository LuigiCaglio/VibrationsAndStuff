# -*- coding: utf-8 -*-
"""
Vibrations and Stuff
Frequency response function - FRF
@author: LC
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt




M = np.array([[1, 0],
              [0, 1]])


K = np.array([[2500, -1500],
              [-1500, 1500]])


C = np.array([[4.64, -1.87],
              [-1.87, 3.39]])



# load influence/distribution matrix
Sp = np.zeros([2,1])
Sp[1,0] = 1



#%%compute modal parameters

#modes

omega_sq, phi = np.linalg.eig(np.linalg.inv(M)@K)
omega = omega_sq**0.5

sorted_indices = np.argsort(omega)
omega = omega[sorted_indices]
phi = phi[:, sorted_indices]

modal_mass = np.diag(phi.T@M@phi)
modal_damping = np.diag(phi.T@C@phi)
modal_stiff = np.diag(phi.T@K@phi)

damping_ratios =modal_damping/  (2*(modal_mass*modal_stiff)**0.5)


#%%compute state space matrices


n = len(M)
nLoads = Sp.shape[1]

# System matrix - continuous time
Ac = np.zeros((2*n,2*n))
Ac[:n,:n] = np.zeros((n,n))
Ac[:n,n:2*n] = np.identity(n)
Ac[n:2*n,:n] = - np.linalg.inv(M) @ K
Ac[n:2*n,n:2*n] = - np.linalg.inv(M) @ C


# Input matrix - continuous time 
Bc  = np.zeros((2*n,nLoads))
Bc[n:2*n,:] = np.linalg.inv(M) @ Sp


S_null =  np.identity(n)[[]]
Hcdisp  = np.vstack((
                np.hstack(( np.identity(n),                np.zeros_like( np.identity(n))  )),
                np.hstack((np.zeros_like(S_null), S_null                 )),
                np.hstack((-S_null@inv(M)@K,      -S_null@inv(M)@C       ))
                ))

Hc_acc  = np.vstack((
                np.hstack(( S_null,                np.zeros_like( S_null  ))),
                np.hstack((np.zeros_like(S_null), S_null                 )),
                np.hstack((-np.identity(n)@inv(M)@K,      -np.identity(n)@inv(M)@C       ))
                ))

#%% compute FRFs
omega_array = np.linspace(0.01,20*2*np.pi,1000)

def FRF_disp_MCK(M,C,K,Spj,omega_array):
    n = len(M)
    N = len(omega_array)
    Hj = np.zeros((n,N),complex)
    for ind,ω in enumerate(omega_array):
        denom= -ω**2 * M + 1j*ω*C + K 
        Hj[:,ind] = inv(denom)@Spj.flatten()
    return Hj

def FRF_accel_MCK(M,C,K,Spj,omega_array):
    n = len(M)
    N = len(omega_array)
    Hj = np.zeros((n,N),complex)
    for ind,ω in enumerate(omega_array):
        denom = -ω**2 * M + 1j*ω*C + K 
        Hj[:,ind] =-ω**2 *inv(denom)@Spj.flatten()
    return Hj



def FRF_disp_SDOF(mk,ζk,ωk,omega_array):
    ω = omega_array
    H_SDOF = 1/mk/(ωk**2-ω**2+2j*ω*ζk*ωk)
    return H_SDOF 

def FRF_disp_modes(m,ζ,ω,φ,Spj,omega_array):
    n_m = len(m)
    N = len(omega_array)
    ω_ar = omega_array
    Hj = np.zeros((n_m,N),complex)
    for k in range(n_m):
        mk,ζk,ωk,φk = m[k],ζ[k],ω[k],φ[:,k]
        FRF_modek = FRF_disp_SDOF(mk,ζk,ωk,ω_ar)
        FRF_modek = FRF_modek.reshape(1,-1)
        φkφkT = np.outer(φk,φk)
        Hj += FRF_modek * (φkφkT @ Spj)
    return Hj

def FRF_acc_modes(m,ζ,ω,φ,Spj,omega_array):
    n_m = len(m)
    N = len(omega_array)
    ω_ar = omega_array
    Hj = np.zeros((n_m,N),complex)
    for k in range(n_m):
        mk,ζk,ωk,φk = m[k],ζ[k],ω[k],φ[:,k]
        FRF_modek = FRF_disp_SDOF(mk,ζk,ωk,ω_ar)
        FRF_modek = -ω_ar**2* FRF_modek.reshape(1,-1)
        φkφkT = np.outer(φk,φk)
        Hj += FRF_modek * (φkφkT @ Spj)
    return Hj




def FRF_state_space(Ac,Bc,Hc,omega_array):
    H = np.zeros((len(Hc),Bc.shape[1],
                   len(omega_array)),
                   complex)
    I = np.eye(len(Ac))
    for ind,ω in enumerate(omega_array):
        temp_matr = 1j*ω*I- Ac 
        H[:,:,ind] = Hc@inv(temp_matr)@Bc
    return H

#MCK
Hdisp_MCK = FRF_disp_MCK(M,C,K,Sp,omega_array)
Hacc_MCK = FRF_accel_MCK(M,C,K,Sp,omega_array)

#modes
Hdisp_modes = FRF_disp_modes(modal_mass,damping_ratios,omega,phi,Sp,omega_array)
Hacc_modes = FRF_acc_modes(modal_mass,damping_ratios,omega,phi,Sp,omega_array)

#state space
H_disp_ss = FRF_state_space(Ac,Bc,Hcdisp,omega_array)
Hacc_ss = FRF_state_space(Ac,Bc,Hc_acc,omega_array)



#%% plot FRFs

def compare_3FRFs(H_MCK,H_modes,H_ss,fr_array,title):
 
    fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True, figsize=(10, 8))
    
    # Magnitude plot on the first subplot
    ax1.semilogy(fr_array, abs(H_modes),label="Modes", linewidth=10.0, alpha=0.3 )
    ax1.semilogy(fr_array, abs(H_MCK),"-", linewidth=3.0,label="MCK",color="orange")
    ax1.semilogy(fr_array, abs(H_ss),"-.",label="State space",color="green")
    ax1.set_title('Magnitude')
    
    
    # Phase plot on the second subplot
    ax2.plot(fr_array, np.angle(H_modes),label="Modes", linewidth=10.0, alpha=0.3 )
    ax2.plot(fr_array, np.angle(H_MCK),"-", linewidth=3.0,label="MCK")
    ax2.plot(fr_array, np.angle(H_ss),"-.",label="State space")
    ax2.set_title('Phase')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.legend()
    # ax2.set_ylabel('Phase (rad)')
    for ax in (ax1,ax2):
        ax.grid()
        
    
    fig.suptitle(title, fontsize=16)
    # Automatically adjust subplot params for a nice fit
    plt.tight_layout()
    
    # Show the entire plot with both subplots
    plt.show()

fr_array = omega_array/2/np.pi
compare_3FRFs(Hdisp_MCK[0],Hdisp_modes[0],H_disp_ss[0].flatten(),fr_array,title="FRF - displacement")
compare_3FRFs(Hacc_MCK[0],Hacc_modes[0],Hacc_ss[0].flatten(),fr_array,title="FRF - acceleration")
