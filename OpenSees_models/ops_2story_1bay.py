# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:30:21 2024

@author: LC
"""

import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import opsvis


def define_model_2story(mass_1fact=1,mass_2fact=1,
                         stiff1_fact=1,stiff2_fact=1,):

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    
    L = 5
    H = 3
    mass1 = 5e3 /L * mass_1fact
    mass2 = 5e3 /L * mass_2fact
    
    #HEB 500
    A_ele = 238.60 *0.01**2
    E_mod = 210e9
    Iz = 107180 *0.01**4
    
    mass_column = A_ele * 7850
    
    ops.node(10, *[0,0]); ops.fix(10, *[1,1,1,])
    ops.node(11, *[L,0]); ops.fix(11, *[1,1,1,])
    
    ops.node(20, *[0,H]) 
    ops.node(21, *[L,H])
    
    ops.node(30, *[0,2*H]) 
    ops.node(31, *[L,2*H])
    
    ops.geomTransf("Linear",123)
    
    #columns 1st floor
    ops.element('elasticBeamColumn', 1020, *[10,20], A_ele, stiff1_fact*E_mod, Iz, 123, '-mass', mass_column, '-cMass')
    ops.element('elasticBeamColumn', 1121, *[11,21], A_ele, stiff1_fact*E_mod, Iz, 123, '-mass', mass_column, '-cMass')
    
    #columns 2nd floor
    ops.element('elasticBeamColumn', 2030, *[20,30], A_ele, stiff2_fact*E_mod, Iz, 123, '-mass', mass_column, '-cMass')
    ops.element('elasticBeamColumn', 2131, *[21,31], A_ele, stiff2_fact*E_mod, Iz, 123, '-mass', mass_column, '-cMass')
    
    #beams
    ops.element('elasticBeamColumn', 2021, *[20,21], A_ele, E_mod, Iz, 123, '-mass', mass1, '-cMass')
    ops.element('elasticBeamColumn', 3031, *[30,31], A_ele, E_mod, Iz, 123, '-mass', mass2, '-cMass')
    

    
    eigs = ops.eigen(6)

    freq = np.array(eigs)**.5 / (2*np.pi)
    periods = 1/freq

    mode1 = []
    mode1.append(ops.nodeEigenvector(21, 1,1))
    mode1.append(ops.nodeEigenvector(31, 1,1))
    
    mode2 = []
    mode2.append(ops.nodeEigenvector(21, 2,1))
    mode2.append(ops.nodeEigenvector(31, 2,1))
    
    return freq,np.array(mode1),np.array(mode2)

if __name__ == '__main__':
    true_params = [1.1,0.9,0.84,1.05]
    print("true parameters: ", true_params)
    
    freq,mode1,mode2 = define_model_2story(mass_1fact=true_params[0],
                                            mass_2fact=true_params[1],
                                            stiff1_fact=true_params[2],
                                            stiff2_fact=true_params[3],)
    
    opsvis.plot_model()
    
    opsvis.plot_mode_shape(1,)
    opsvis.plot_mode_shape(2,)
    
    print("periods :",1/freq)
