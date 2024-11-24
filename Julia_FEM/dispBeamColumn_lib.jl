"""
Created on Fri Nov 24 2024

@author: Luigi Caglio
https://vibrationsandstuff.wordpress.com/finite-element-stuff/
Vibrations and Stuff
Displacement-based beam-column element in Julia
"""

abstract type Element end
abstract type BeamColumnElement <: Element end

mutable struct DispBeamColumn2D <: BeamColumnElement
    tag::Int
    node1::Node
    node2::Node
    beamIntegration::BeamIntegration
    geomTransf::GeometricTransformation

    #response variables 
    Kb::Matrix{Float64} # stiffness
    Fb_trial::Vector{Float64} # forces
    Fb_committed::Vector{Float64}
    ub_trial::Vector{Float64} # displacements
    ub_committed::Vector{Float64}

    # basic to global transformation
    Tbg::Matrix{Float64}


    function DispBeamColumn2D(tag::Int, node1::Node,node2::Node,
                            beamIntegration::BeamIntegration,
                            geomTransf::GeometricTransformation)
        Kb = zeros(3,3)
        Fb_trial,Fb_committed = zeros(3),zeros(3)
        ub_trial,ub_committed = zeros(3),zeros(3)
        Tbg = Tbg!(geomTransf,node1,node2)

        return new(tag,node1,node2,
        beamIntegration,geomTransf,
        Kb, 
        Fb_trial,Fb_committed,
        ub_trial,ub_committed,
        Tbg
        )
    end
end

function compute_Tsb(ele::DispBeamColumn2D,x::Float64, L::Float64)
    ##x goes from 0 to L
    Tsb = [1/L    0                0;
         0      -4/L + 6*x/L^2    -2/L + 6*x/L^2 ]   
    return Tsb
end

function state_determination!(ele::DispBeamColumn2D,Δug::Vector{Float64})

    #get locations and weights of integration points
    xIP = ele.beamIntegration.xIP
    wIP = ele.beamIntegration.wIP

    #compute basic displacement increment
    Tbg = Tbg!(ele.geomTransf,node1,node2)
    Δub = Tbg * Δug
 
    #compute element length
    L = norm(node2.crds-node1.crds)

    #initialize basic stiffness and force
    Kb = zeros(3,3)
    Fb = zeros(3)

    # The loop imposes the section deformations and performs
    # the integration via Gaussiann quadrature
    for i in 1:ele.beamIntegration.Np

        # Displacement interpolation evaluated at integration points
        x = 0.5*(xIP[i] + 1.0)*L #xIP ∈[-1,1] while x ∈[0,L]
        Tsb =  compute_Tsb(ele1,x, L)
        
        Δus = Tsb*Δub # Section deformation increment

        section_i = ele.beamIntegration.sections[i]

        # Impose section def to get section forces and stiffness
        Fs, Ks = impose_def_incr!(section_i, Δus)

        # Add contribution of section forces and stiffness
        # to the basic force vector and stiffness matrix.
        # The weights are multiplied by L/2 so they sum to L
        # (wIP sum to 2)
        Kb += Tsb' * Ks * Tsb * (wIP[i] * L/2)
        Fb += Tsb' * Fs   *     (wIP[i] * L/2)
    end 
    
    #global quantities
    Fg = Tbg' * Fb 
    Kg = Tbg' * Kb * Tbg   
    
    #save quantities
    ele.ub_trial += Δub
    ele.Fb_trial = Fb
    ele.Kb = Kb
    
    return Fg, Kg
end


function commit!(ele::DispBeamColumn2D)
    ele.Fb_committed = copy(ele.Fb_trial)
    ele.ub_committed = copy(ele.ub_trial) 
    for section in ele.beamIntegration.sections
        commit!(section)
    end
end

function revert_to_last_commit!(ele::DispBeamColumn2D)
    ele.Fb_trial = copy(ele.Fb_committed)
    ele.ub_trial = copy(ele.ub_committed)
    for section in ele.beamIntegration.sections
        revert_to_last_commit!(section)
    end
end

