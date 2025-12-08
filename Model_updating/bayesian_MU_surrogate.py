# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:30:21 2024

@author: LC
"""

import numpy as np
import matplotlib.pyplot as plt
import opsvis
from ops_2story_1bay import define_model_2story
import corner
from scipy.stats import truncnorm




true_params = [1.1,0.9,0.84,1.05]


# true_params = [1.05,0.94,0.96,1.02]

freq,mode1,mode2 = define_model_2story(mass_1fact=true_params[0],
                                        mass_2fact=true_params[1],
                                        stiff1_fact=true_params[2],
                                        stiff2_fact=true_params[3],)

opsvis.plot_model()

opsvis.plot_mode_shape(1,)
opsvis.plot_mode_shape(2,)

print("periods :",1/freq)



#%% define probability distributions

def MAC(phi_1,phi_2):
    MAC = np.dot(phi_1,phi_2)**2/ (phi_1.dot(phi_1)*phi_2.dot(phi_2))
    return MAC

def gaussian_distribution(x,mu, sigma):
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y



def truncated_gaussian_pdf(x, mu, sigma, a, b):
    x = np.array(x).reshape(-1)
    # Calculate the truncated normal distribution bounds relative to the mean and std
    a, b = (a - mu) / sigma, (b - mu) / sigma
    
    tn_dist = truncnorm(a, b, loc=mu, scale=sigma)
    
    # Calculate the PDF for the given x
    pdf = tn_dist.pdf(x)
    
    return pdf

def truncated_uniform_pdf(x, mu, sigma, a, b):
    x = np.array(x).reshape(-1)
    
    constant_value = 1/(b-a)
    pdf = np.where((x >= a) & (x <= b), constant_value, 0)
    
    
    return pdf



plt.rcParams.update({'font.size': 16})  # Adjust the 16 to your desired font size

def plot_distribution(distribution_function,params,title="", limits_plot = [0.85,1.15]):
    # Generate data for the Gaussian distribution
    mu,sigma = params[0],params[1]
    x = np.linspace(mu*limits_plot[0], mu*limits_plot[1], 1000)
    y = distribution_function(x,*params)
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'μ={mu:.2f}, σ={sigma:.2f}')
    plt.title(title)
    plt.xlabel('freq (Hz)')
    plt.ylabel('Probability Density')
    # plt.legend()
    plt.grid(True)
    plt.show()
    
    

#define standard deviations frequency models
sigma_f_i_normalized = 0.01

sigma_f1 = sigma_f_i_normalized *freq[0]
sigma_f2 = sigma_f_i_normalized *freq[1]



sigma_prior = 0.5
prior_upper_limit = 1.3
prior_lower_limit = 0.7



plot_distribution(gaussian_distribution,[freq[0], sigma_f1],"Frequency 1",[0.95,1.05])
plot_distribution(gaussian_distribution,[freq[1], sigma_f2],"Frequency 2",[0.95,1.05])


plot_distribution(truncated_uniform_pdf,[1,sigma_prior ,
                                         prior_lower_limit,prior_upper_limit],
                                        "Parameter prior - $P(m_1)$ and $P(m_2) $",[0.6,1.4])


#%%
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 6))

limits_plot = [0.85,1.15]
mu,sigma = freq[0], sigma_f1
x = np.linspace(mu*limits_plot[0], mu*limits_plot[1], 1000)
y = gaussian_distribution(x,*[freq[0], sigma_f1])
# Plot the distribution

ax1.plot(x, y, label=f'μ={mu:.2f}\n σ={sigma:.2f}')
ax1.set_title("Frequency 1")
ax1.set_xlabel('freq (Hz)')
ax1.set_ylabel('Probability Density')

limits_plot = [0.90,1.10]
mu,sigma = freq[1], sigma_f2
x = np.linspace(mu*limits_plot[0], mu*limits_plot[1], 1000)
y = gaussian_distribution(x,*[freq[1], sigma_f2])
# Plot the distribution

ax2.plot(x, y, label=f'μ={mu:.2f}\n σ={sigma:.2f}')
ax2.set_title("Frequency 2")
ax2.set_xlabel('freq (Hz)')

ax1.grid(True)
ax2.grid(True)
ax1.legend(loc='upper right',fontsize=12)
ax2.legend(loc='upper right',fontsize=12)
plt.show()

    
    

# %% define likelihood function - prior - unnormalized posterior

def likelihood_function(params):
    
    freq_i,mode1_i,mode2_i = define_model_2story(mass_1fact=params[0],
                                               mass_2fact=params[1],
                                               # stiff1_fact=params[2],
                                               # stiff2_fact=params[3]),
                                               stiff1_fact=true_params[2],
                                               stiff2_fact=true_params[3],)
    
    likelihood_f_1 = gaussian_distribution(freq_i[0],freq[0], sigma_f1)
    likelihood_f_2 = gaussian_distribution(freq_i[1],freq[1], sigma_f2)
    
    MAC1 = MAC(mode1_i,mode1)
    MAC2 = MAC(mode2_i,mode2)
        
    
    likelihood_all = likelihood_f_1*likelihood_f_2
    
    return likelihood_all, [freq_i,[MAC1,MAC2]]



def prior(params):
    
    # prior_m1 = truncated_gaussian_pdf(params[0], mu=1, sigma=sigma_prior, a=prior_lower_limit, b=prior_upper_limit)[0]
    # prior_m2 = truncated_gaussian_pdf(params[1], mu=1, sigma=sigma_prior, a=prior_lower_limit, b=prior_upper_limit)[0]
    # prior_k1 = truncated_gaussian_pdf(params[2], mu=1, sigma=sigma_prior, a=prior_lower_limit, b=prior_upper_limit)[0]
    # prior_k2 = truncated_gaussian_pdf(params[3], mu=1, sigma=sigma_prior, a=prior_lower_limit, b=prior_upper_limit)[0]
    prior_m1 = truncated_uniform_pdf(params[0], mu=1, sigma=sigma_prior, a=prior_lower_limit, b=prior_upper_limit)[0]
    prior_m2 = truncated_uniform_pdf(params[1], mu=1, sigma=sigma_prior, a=prior_lower_limit, b=prior_upper_limit)[0]
    # prior_k1 = truncated_uniform_pdf(params[2], mu=1, sigma=sigma_prior, a=prior_lower_limit, b=prior_upper_limit)[0]
    # prior_k2 = truncated_uniform_pdf(params[3], mu=1, sigma=sigma_prior, a=prior_lower_limit, b=prior_upper_limit)[0]
  
    return prior_m1*prior_m2





def unnormalized_neglogposterior(params):
    likelihood, freq_i = likelihood_function(params)
    return -np.log(prior(params)+1e-320) - np.log(likelihood),  freq_i


#%%


from scipy.stats import qmc
from smt.surrogate_models import RBF  ### radial basis function

n_samples_LH= 20
n_parameters = 2

prior_upper_limit = 1.3
prior_lower_limit = 0.7

sampler_LH = qmc.LatinHypercube(d=n_parameters)
samples_LH = sampler_LH.random(n=n_samples_LH)
l_bounds = [prior_lower_limit]*n_parameters
u_bounds = [prior_upper_limit]*n_parameters
samples_LH_scaled = qmc.scale(samples_LH, l_bounds, u_bounds)

plt.figure()
plt.plot(samples_LH_scaled[:,0],samples_LH_scaled[:,1],"o")
plt.xlabel("m1")
plt.xlabel("m2")
plt.show()


### we train three surrogates to get the posterior
training_points = samples_LH_scaled
training_negloglikelihood = np.zeros(len(training_points))



for ind,point_i in enumerate(training_points):
        # if ind<50: continue
        training_negloglikelihood[ind] = -np.log(likelihood_function(point_i)[0])
        
        print(f"{ind+1} of {n_samples_LH}")  


#%%
sm = RBF(print_global=False)##surrogate model
sm.set_training_values(training_points, training_negloglikelihood)
sm.train()

def surrogate_unnorm_neglogposterior(params):
    negloglike = sm.predict_values(params[:2].reshape(1,2))[0,0]
    neglogposterior = -np.log(prior(params)+1e-320) + negloglike
    return neglogposterior
    
    
#%% ##test how accurate the surrogate model is (not necessary)
import sys
import os
from contextlib import redirect_stdout, redirect_stderr


x_contour_plot = np.linspace(prior_lower_limit,prior_upper_limit, 100)
y_contour_plot = np.linspace(prior_lower_limit,prior_upper_limit, 100)

X_cp, Y_cp = np.meshgrid(x_contour_plot, y_contour_plot)

posterior_true_cp = np.zeros((100, 100))
posterior_sm_cp = np.zeros((100, 100))


# with open(os.devnull, 'w') as fnull: ##to not show the messages on the console
#     with redirect_stdout(fnull), redirect_stderr(fnull):
for i in range(len(x_contour_plot)):
    for j in range(len(y_contour_plot)):
        point_ij = np.array([x_contour_plot[i],y_contour_plot[j]] )
        
        posterior_sm_cp[i, j] = surrogate_unnorm_neglogposterior(point_ij)
        posterior_true_cp[i, j] = unnormalized_neglogposterior(point_ij)[0]

        

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Creates a figure and two subplots

# Contour plot on the first subplot
contour1 = ax1.contourf(X_cp, Y_cp, posterior_true_cp, 20, cmap='Spectral_r')
fig.colorbar(contour1, ax=ax1)
ax1.set_title('True posterior')
ax1.set_xlabel('m 1')
ax1.set_ylabel('m 2')

# Contour plot on the second subplot (identical to the first)
contour2 = ax2.contourf(X_cp, Y_cp, posterior_sm_cp, 20, cmap='Spectral_r')
fig.colorbar(contour2, ax=ax2)
ax2.set_title('Surrogate model')
ax2.set_xlabel('m 1')
ax2.set_ylabel('m 2')

plt.show()


#%% Bayesian inference - Markov Chain Monte Carlo



std_step = 0.1

def sampling_algorigm(theta_prev):
    proposal = np.random.normal(theta_prev, std_step)
    
    return proposal
    # return np.clip(proposal,prior_lower_limit,prior_upper_limit)
    



def metropolis_sampling(theta_prev,nll_prev):
    theta_proposal = sampling_algorigm(theta_prev)
    
    # likelihood_proposal,freq_i = unnormalized_posterior(theta_proposal)
    likelihood_proposal = surrogate_unnorm_neglogposterior(theta_proposal)
    
    u = np.random.rand()
    
    # alpha_i = likelihood_proposal/nll_prev
    alpha_i = np.exp(nll_prev-likelihood_proposal)
    
    
    if alpha_i > u:
        is_new_proposal_accepted = 1
        return theta_proposal, likelihood_proposal, is_new_proposal_accepted,None
    else:
        is_new_proposal_accepted = 0
        return theta_prev, nll_prev,is_new_proposal_accepted, None

initial_theta = np.random.uniform(prior_lower_limit, prior_upper_limit,4)

max_iter = 250_000


nll_array = np.zeros(max_iter+1)
theta_iter = np.zeros([len(true_params),max_iter+1])

freq_list = []

# nll = unnormalized_posterior(initial_theta)


theta_iter[:,0] = initial_theta
nll_array[0] =  surrogate_unnorm_neglogposterior(initial_theta)
freq_list.append(None)

iteration = 1

list_accepted_proposal = [1]
n_accepted_proposals = 1




#%% start or continue iterations


while iteration < max_iter:
    
    theta_new,nll_new, accepted,freq_i = \
        metropolis_sampling(theta_iter[:,iteration-1],
                            nll_array[iteration-1])
    theta_iter[:,iteration] = theta_new
    nll_array[iteration] = nll_new
    if freq_i is None: 
        freq_list.append(freq_list[-1])
    else:
        freq_list.append(freq_i)
    
    
    n_accepted_proposals += accepted
    list_accepted_proposal.append(accepted)
    iteration +=1
    
    if iteration%1000 == 0: print(iteration)
    
    if iteration%10000 == 0:
        print("\n\n acceptance ratio ",n_accepted_proposals/iteration,
              "\n\n")




#%% plot inference

burnin = 1000

acceptance_vs_iter = np.array(list_accepted_proposal[burnin:])

    


fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(20,10))
fig.suptitle("m1", fontsize=16)
ax1.hist(theta_iter[0,burnin:iteration],bins=100)
ax1.set_title("histogram of samples")

ax2.plot(theta_iter[0,burnin:iteration])
# ax2.set_title("histogram of samples")

ax4.plot(np.cumsum(acceptance_vs_iter)/(np.arange(len(acceptance_vs_iter))+1))
plt.plot()


fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(20,10))
fig.suptitle("m2", fontsize=16)
ax1.hist(theta_iter[1,burnin:iteration],bins=100)
ax1.set_title("histogram of samples")

ax2.plot(theta_iter[1,burnin:iteration])
# ax2.set_title("histogram of samples")

ax4.plot(np.cumsum(acceptance_vs_iter)/(np.arange(len(acceptance_vs_iter))+1))
plt.plot()





figure = corner.corner(theta_iter[:2,burnin:iteration].T, 
                        labels=["$m_1$", "$m_2$",], 
                        show_titles=True)

plt.show()



# # true_params = [1.1,0.9,0.84,1.05]
print(true_params)



