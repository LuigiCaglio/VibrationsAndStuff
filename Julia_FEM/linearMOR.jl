"""
Created on Fri Nov 22 2024

@author: LC
https://vibrationsandstuff.wordpress.com/finite-element-stuff/
Vibrations and Stuff
Guyan reduction (a.k.a. static condensation) in Julia
Craig-Bampton Method (a.k.a. dynamic condensation) in Julia
"""

using LinearAlgebra


function GuyanReduction(M::Matrix{Float64},
                        C::Matrix{Float64},
                        K::Matrix{Float64})

    # Divide DOFs with and without mass
    DOFs_without_mass = findall(i -> M[i, i] == 0.0, 1:min(size(M)...))
    DOFs_with_mass = setdiff(1:min(size(M)...), DOFs_without_mass)

    # extract submatrices of K
    Kss = K[DOFs_without_mass,DOFs_without_mass]
    Ksm = K[DOFs_without_mass,DOFs_with_mass]

    # compute transformation_matrix
    T_G = zeros(size(M,1),length(DOFs_with_mass))
    T_G[DOFs_with_mass,:] .= I(length(DOFs_with_mass))
    T_G[DOFs_without_mass,:] .=  -Kss\Ksm
    
    # compute reduced matrices
    Mr, Cr, Kr = T_G'*M*T_G, T_G'*C*T_G, T_G'*K*T_G
    return Mr, Cr, Kr, T_G
end


function CraigBamptonReduction(M::Matrix{Float64},C::Matrix{Float64},K::Matrix{Float64},
                    DOFs_substructure::Vector{Int},n_modes_substructure::Int)
    
    # define retained DOFs of main structure
    DOFs_retained = setdiff(1:min(size(M)...), DOFs_substructure)

    #number of DOFs and submatrix of stiffness      
    nDOFs_full = size(M,1)
    nDOFs_retained = length(DOFs_retained)
    nDOFs_reduced = size(M,1)-length(DOFs_substructure)+n_modes_substructure
    Ksm = K[DOFs_substructure,DOFs_retained]

    # compute mode shapes ϕss of substructure
    Mss = M[DOFs_substructure,DOFs_substructure]
    Kss = K[DOFs_substructure,DOFs_substructure]
    ϕss = eigvecs(Kss,Mss)

    # compute transformation matrix (the dofs might be shuffled)
    T_CB = zeros(nDOFs_full,nDOFs_reduced)
    T_CB[DOFs_retained,1:nDOFs_retained] .= I(length(DOFs_retained))
    T_CB[DOFs_substructure,1:nDOFs_retained] .=  -Kss\Ksm
    T_CB[DOFs_substructure,nDOFs_retained+1:end] .= ϕss[:,1:n_modes_substructure]
    
    # compute reduced matrices
    Mr, Cr, Kr = T_CB'*M*T_CB, T_CB'*C*T_CB, T_CB'*K*T_CB
    return Mr, Cr, Kr, T_CB
end
