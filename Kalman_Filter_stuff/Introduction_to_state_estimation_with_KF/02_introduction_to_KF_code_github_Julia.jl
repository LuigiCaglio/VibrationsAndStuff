using LinearAlgebra
using PyCall
using Plots
using DelimitedFiles

cd("C:/Users/lucag/OneDrive - Danmarks Tekniske Universitet/Skrivebord/Luigi_personal/blog_struct_dyn/intro_to_KF_estimation_code")

function kalman_filter!(z_prev, P_prev, A, H, Q, R, d)
    #prediction step
    z_pred = A * z_prev
    P_pred = A * P_prev * A' + Q

    #Kalman Gain
    Kgain = P_pred * H' * inv(H * P_pred * H' + R)

    # estimation
    z_est = z_pred + Kgain * (d - H * z_pred)
    P_est = P_pred - Kgain * H * P_pred

    return z_est, P_est
end


# Define state space matrices

# state transition matrix (computed elsewhere)
A = [ 9.87610634e-01 7.42403599e-03 9.88586175e-03 5.39785161e-05
      7.42403599e-03 9.92559992e-01 5.39785161e-05 9.92184743e-03
     -2.46336866e+00 1.47478248e+00 9.73143171e-01 1.32051843e-02
      1.47478248e+00 -1.48018034e+00 1.32051843e-02 9.81946628e-01]

# output matrix (computed elsewhere)
H = [ 150. -150. 0.59064779 -1.07290968 ]

# measurements (computed elsewhere)
d = readdlm("measurement_KF_acc2.out")' 

# measurements noise variance (computed elsewhere)
R = [ 1.157 ] # noise variance (assumed known)

# number of time steps
N = size(d, 1)

# number of states
n_states = size(A, 1)

# Process noise covariance matrix
Q = 1e-15 * I(n_states) 

# initialize vector for estimation
z_kf = zeros(n_states, N)

# initial state and error covariance
z0 = zeros(n_states)
P0 = Matrix{Float64}(I, n_states, n_states) * 10  # 10 is relatively big compared to magnitude of displacements

P = copy(P0) 

# KF loop
for k = 2:N
    global P
    z_kf[:, k], P = kalman_filter!(z_kf[:, k-1], P, A, H, Q, R, d[k, :])
end




# compare estimation with true response
# Loading the array with the true responses (displacements and velocities)
pickle = pyimport("pickle")

z_true = open("z_true.pkl", "r") do f
    pickle.load(f)
end

dt = 0.01
t_end = dt * N
time_array = range(0, step=dt, length=N)

# Plot for disp DOF 1
plot1 = plot(time_array, [z_true[1, :], z_kf[1, :]], label = ["True" "Kalman Filter"])
title!("disp DOF 1")
xlabel!("time")
display(plot1)

# Plot for vel DOF 1
plot2 = plot(time_array, [z_true[3, :], z_kf[3, :]], label = ["True" "Kalman Filter"])
title!("vel DOF 1")
xlabel!("time")
display(plot2)