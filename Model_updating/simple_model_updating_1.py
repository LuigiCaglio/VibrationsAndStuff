# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:30:21 2024

@author: LC

code for post https://vibrationsandstuff.wordpress.com/2024/03/03/simple-model-updating-via-global-optimization/

needs code for OpenSees model https://github.com/LuigiCaglio/VibrationsAndStuff/blob/main/OpenSees_models/ops_2story_1bay.py
"""

import numpy as np
import matplotlib.pyplot as plt
import opsvis
from ops_2story_1bay import define_model_2story

from scipy.optimize import differential_evolution,Bounds


true_params = [1.1,0.9,0.84,1.05]

freq,mode1,mode2 = define_model_2story(mass_1fact=true_params[0],
                                        mass_2fact=true_params[1],
                                        stiff1_fact=true_params[2],
                                        stiff2_fact=true_params[3],)

opsvis.plot_model()

opsvis.plot_mode_shape(1,)
opsvis.plot_mode_shape(2,)

print("periods :",1/freq)



#%% optimization for model updating - 2 masses and 2 stiffnesses

def MAC(phi_1,phi_2):
    MAC = np.dot(phi_1,phi_2)**2/ (phi_1.dot(phi_1)*phi_2.dot(phi_2))
    return MAC

def objective_function(params):
    
    freq_i,mode1_i,mode2_i = define_model_2story(mass_1fact=params[0],
                                               mass_2fact=params[1],
                                               stiff1_fact=params[2],
                                               stiff2_fact=params[3],)
    
    
    diff_freq_1 = freq_i[0]-freq[0]
    diff_freq_2 = freq_i[1]-freq[1]
    
    MAC1 = MAC(mode1_i,mode1)
    MAC2 = MAC(mode2_i,mode2)
    
    
    objective_function = (diff_freq_1/freq_i[0])**2 +\
                         (diff_freq_2/freq_i[1])**2 \
                         - MAC1 -MAC2
    
    return objective_function


bounds = Bounds([0.7, 0.7, 0.7, 0.7], 
                [1.3, 1.3, 1.3, 1.3])


result_optim = differential_evolution(objective_function, bounds, 
                                      maxiter=1000, popsize=20, tol=0.01)


print("true parameters: ", true_params)
print("estimated parameters: ", result_optim.x)

