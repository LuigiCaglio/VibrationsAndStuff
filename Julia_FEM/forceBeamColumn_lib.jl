"""
Created on Nov 25 2024

@author: Luigi Caglio
https://vibrationsandstuff.wordpress.com/finite-element-stuff/
Vibrations and Stuff
Force-based beam-column element in Julia
"""


mutable struct ForceBeamColumn2D <: BeamColumnElement
    tag::Int
    node1::Node
    node2::Node
    beamIntegration::BeamIntegration
    geomTransf::GeometricTransformation

    #response variables 
    Kb_trial::Matrix{Float64} # stiffness
    Kb_committed::Matrix{Float64} # stiffness
    Fb_trial::Vector{Float64} # forces
    Fb_committed::Vector{Float64}
    ub_trial::Vector{Float64} # displacements
    ub_committed::Vector{Float64}

    # basic to global transformation
    Tbg::Matrix{Float64}

    #unbalanced section forces
    Fs_unbalanced::Matrix{Float64}

    function ForceBeamColumn2D(tag::Int, node1::Node,node2::Node,
                            beamIntegration::BeamIntegration,
                            geomTransf::GeometricTransformation)
        # initialize stuff
        ub_trial,ub_committed,Fb_trial,Fb_committed = zeros(3),zeros(3),zeros(3),zeros(3)
        Tbg = Tbg!(geomTransf,node1,node2)
        Fs_unbalanced = zeros(beamIntegration.Np,2)
        
        L = norm(node2.crds-node1.crds) #  element length
        fb=zeros(3,3) #flexibility
        for i in 1:beamIntegration.Np #loop for computing initial flexibility
            x = 0.5*(beamIntegration.xIP[i] + 1.0)*L #xIP ∈[-1,1] while x ∈[0,L]
            Tsb = [1   0   0;  0   -(1 - x/L)   x/L]  
            section_i = beamIntegration.sections[i]
            Ks = return_stiffness(section_i)
            # Sum section terms
            fb += Tsb' * inv(Ks) * Tsb * (beamIntegration.wIP[i] * L/2)
        end
        Kb = inv(fb) #stiffness
        
        return new(tag,node1,node2,
        beamIntegration,geomTransf,
        Kb, Kb, 
        ub_trial,ub_committed,
        Fb_trial,Fb_committed,
        Tbg, Fs_unbalanced
        )
    end
end


function compute_Tsb(ele::ForceBeamColumn2D,x::Float64, L::Float64)
    #Tsb for the FBE is based on the force interpolation
    #x goes from 0 to L 
    Tsb = [1    0                0;
           0    -(1 - x/L)     x/L]   
    return Tsb
end
 



# from 1997 Neuenhofer and Filippou, Evaluation of Nonlinear Frame Finite-Element Models
function state_determination!(ele::ForceBeamColumn2D,
                                  Δug::Vector{Float64})
    
    #get locations and weights of integration points
    xIP = ele.beamIntegration.xIP
    wIP = ele.beamIntegration.wIP

    #compute basic displacement increment
    Tbg = Tbg!(ele.geomTransf,ele.node1,ele.node2) 
    Δub = Tbg * Δug
 
    #compute element length
    L = norm(ele.node2.crds-ele.node1.crds)

    kb_imin1 = ele.Kb_trial #previous stiffness
    ΔFb_i = kb_imin1*Δub    #basic deformation increment

    fb = zeros(3, 3) #initialize basic flexibility
    ub_residual_i = zeros(3) #initialize basic residual disp

    for j in 1:ele.beamIntegration.Np
        # Force interpolation evaluated at integration points
        x = 0.5*(xIP[j] + 1.0)*L #xIP ∈[-1,1] while x ∈[0,L]
        Tsb =  compute_Tsb(ele,x, L)
    
        section_j = ele.beamIntegration.sections[j]

        ΔFs_i = Tsb*ΔFb_i + ele.Fs_unbalanced[j,:] #section force incr   

        Fs_imin1 = copy(section_j.forces_trial) #previous section force
        
        # Section state determination (compute force and stiffness)
        Δus_i = section_j.ks\ΔFs_i #previous deformation incr
        Fs_i, ks = impose_def_incr!(section_j, Δus_i)
        fs = inv(ks) # section flexibility

        # Add contribution of section residual def and flexibility
        # to the basic residual disp and flexibility matrix.
        # The weights are multiplied by L/2 so they sum to L
        # (wIP sum to 2)
        fb += Tsb' * fs * Tsb *(wIP[j] * L/2)
        ub_residual_i += Tsb'*(fs*(Fs_imin1 + ΔFs_i - Fs_i))*(wIP[j] * L/2)
    end
 
    kb_i = inv(fb) #basic stiffness
    ele.Kb_trial = kb_i

    Fb_i = ele.Fb_trial + ΔFb_i -kb_i*ub_residual_i #basic force

    #loop over the sections to update the unbalanced force
    for j in 1:ele.beamIntegration.Np
        x = 0.5*(xIP[j] + 1.0)*L #xIP ∈[-1,1] while x ∈[0,L]
        Tsb =  compute_Tsb(ele,x, L)
        section_j = ele.beamIntegration.sections[j]

        #update unbalanced section forces for next Newton iteration
        ele.Fs_unbalanced[j,:] = Tsb*Fb_i - section_j.forces_trial
    end

    #update internal state of the element
    ele.ub_trial += Δub
    ele.Fb_trial = Fb_i

    #global quantities
    Fg = Tbg'*Fb_i
    Kg = Tbg'*kb_i*Tbg

    return Fg, Kg
end 



function commit!(ele::ForceBeamColumn2D)
    ele.Fb_committed = copy(ele.Fb_trial)
    ele.ub_committed = copy(ele.ub_trial)
    ele.Kb_committed = copy(ele.Kb_trial)
    for section in ele.beamIntegration.sections
        commit!(section)
    end
end

function revert_to_last_commit!(ele::ForceBeamColumn2D)
    ele.Fb_trial = copy(ele.Fb_committed)
    ele.ub_trial = copy(ele.ub_committed)
    ele.Kb_trial = copy(ele.Kb_committed)
    for section in ele.beamIntegration.sections
        revert_to_last_commit!(section)
    end
end
