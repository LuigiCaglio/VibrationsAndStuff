# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:30:21 2024

@author: LC
"""

import numpy as np
import matplotlib.pyplot as plt
import opsvis
from ops_2story_1bay import define_model_2story
from math import erf
import corner
from scipy.stats import truncnorm



# np.random.seed(123)

#define true mean and standard deviation of population normal distribution
mean_stiffness_distribution_population = 1.0
std_stiffness_distribution_population = 0.20

number_of_structures_in_population = 5


true_stiffness_params = np.random.normal(loc=mean_stiffness_distribution_population, 
                                         scale=std_stiffness_distribution_population, 
                                         size=number_of_structures_in_population)


true_params = [mean_stiffness_distribution_population,std_stiffness_distribution_population] + list(true_stiffness_params)
true_params = np.array(true_params)


#let's vary the mass so each structure is different inherently - we will consider this to be known
true_mass_params = np.random.normal(loc=1.0, 
                                        scale=0.5, 
                                         size=number_of_structures_in_population)



#save first frequency for each structure in the population
first_frequencies_population = [] 

for i in range(number_of_structures_in_population):
 
    freq_i,_,_ = define_model_2story(       mass_1fact=true_mass_params[i],# known
                                            mass_2fact=true_mass_params[i],# known
                                            
                                            stiff1_fact=true_stiffness_params[i],# to be updated
                                            stiff2_fact=true_stiffness_params[i],)# to be updated
    
    first_frequencies_population.append(freq_i[0]) 





first_frequencies_population = np.array(first_frequencies_population)
print("periods :",1/first_frequencies_population)

cov_freq = first_frequencies_population.std()/first_frequencies_population.mean()




opsvis.plot_model()

opsvis.plot_mode_shape(1,)
opsvis.plot_mode_shape(2,)

#%% define probability distributions

def gaussian_distribution(x,mu, sigma):
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y



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


def plot_distribution2(distribution_function,params,label_=""):
    # Generate data for the Gaussian distribution
    mu,sigma = params[0],params[1]
    x = np.linspace(np.min(first_frequencies_population)*0.7, np.max(first_frequencies_population)*1.3, 1000)
    y = distribution_function(x,*params)
    # Plot the distribution
    plt.plot(x, y, label=label_)
    
    

#define standard deviations frequency models - i.e., noise/error in the estimation of the frequencies
std_frequency_measurement_normalized = 0.01 ##i.e., 0.01 = 1% of measurement uncertainty



#values of (unnormalized) standard deviations for the whole population
std_frequency_measurement_population = std_frequency_measurement_normalized * first_frequencies_population
 



plt.figure()


for i in range(number_of_structures_in_population):
    
    plot_distribution2(gaussian_distribution,[first_frequencies_population[i], std_frequency_measurement_population[i]],
                       label_=f"individual {i}")
    
plt.title("Frequency 1 population")
plt.xlabel('freq (Hz)')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
 
    

# %% define likelihood functions - one for each individual of the population


# I am defining a dictionary and each entry with key "i" is the likelihood function of the i-th individual

dictionary_likelihood_functions = {}
    
for i in range(number_of_structures_in_population):
   
   def make_likelihood_function(structure_idx):  # Factory function
       def likelihood_function_i(params_i):
           stiffness_parameter = params_i
           
           freq_i, _, _ = define_model_2story(
               mass_1fact=true_mass_params[structure_idx],  # Use structure_idx
               mass_2fact=true_mass_params[structure_idx],
               stiff1_fact=stiffness_parameter,
               stiff2_fact=stiffness_parameter
           )
           
           first_frequency_i = freq_i[0]
           
           likelihood_freq_1 = gaussian_distribution(
               first_frequency_i,
               first_frequencies_population[structure_idx],  # Use structure_idx
               std_frequency_measurement_population[structure_idx]
           )
           
           return likelihood_freq_1
       
       return likelihood_function_i
   
   dictionary_likelihood_functions[i] = make_likelihood_function(i)   
    

# dictionary_likelihood_functions[0](1)
#%%
# ##likelihood of the whole population
# def likelihood_function()

# hyper_prior = p(Ψ)
def hyper_prior(hyper_params):
    return 1.0 ##assume a flat prior


# conditional_prior = p(θ_i|Ψ)
def conditional_prior_i(hyper_params,params_i): # probability of theta_i given a value of the hyperparams
    # hyper_params are the parameters of the population
    # params_i are the parameters of the single structure
    
    stiffness_param_i = params_i
    conditional_prior_i = gaussian_distribution(x = stiffness_param_i,
                                                mu = hyper_params[0], 
                                                sigma = hyper_params[1])
    return conditional_prior_i
    



def unnormalized_posterior(all_params): # considering all the parameters
    
    hyper_params = all_params[:2]
    params_population = all_params[2:]
    
    
    hyper_prior_ = hyper_prior(hyper_params) #p(Ψ)
    
    total_likelihood_times_cond_prior = 1.0 # initialize product
    
    for i in range(number_of_structures_in_population):
        params_i = params_population[i]
        
        data_likelihood_i = dictionary_likelihood_functions[i](params_i)
        cond_prior_i = conditional_prior_i(hyper_params,params_i)
        # print(i)
        # print(data_likelihood_i)
        # print(cond_prior_i)
        
        total_likelihood_times_cond_prior *= data_likelihood_i*cond_prior_i

    return hyper_prior_ * total_likelihood_times_cond_prior 



def unnormalized_neg_log_posterior(all_params): # considering all the parameters
    ##we do it in the logspace so it is more stable
    
    all_params_unnormalized = all_params*true_params
    hyper_params = all_params_unnormalized[:2]
    params_population = all_params_unnormalized[2:]
    
    
    neg_log_hyper_prior_ = -np.log(hyper_prior(hyper_params)) #-log(p(Ψ))
    
    
    
    total_negative_log_likelihood_plus_neglogcond_prior = 0. # initialize product
    
    for i in range(number_of_structures_in_population):
        params_i = params_population[i]
        
        data_likelihood_i = dictionary_likelihood_functions[i](params_i)
        cond_prior_i = conditional_prior_i(hyper_params,params_i)
        
        neg_log_likelihood_i = -np.log(data_likelihood_i)
        neg_log_cond_prior_i = -np.log(cond_prior_i)
        
        # print(i)
        # print(data_likelihood_i)
        # print(cond_prior_i)
        
        total_negative_log_likelihood_plus_neglogcond_prior += neg_log_likelihood_i + neg_log_cond_prior_i

    return neg_log_hyper_prior_ + total_negative_log_likelihood_plus_neglogcond_prior 

#%% Bayesian inference - Markov Chain Monte Carlo



std_step = 0.02

diag_cov = [0.1,0.01,std_step,std_step]

prior_upper_limit = 2
prior_lower_limit = 0.4


n_params = 2+number_of_structures_in_population
initial_theta = np.ones_like(true_params)*2


max_iter = 50_000


nll_array = np.zeros(max_iter+1)
theta_iter = np.zeros([len(true_params),max_iter+1])

freq_list = []

nll = unnormalized_posterior(initial_theta*true_params)



def sampling_algorithm(theta_prev):
    proposal = np.random.normal(theta_prev, diag_cov)
    
    # return proposal
    return np.clip(proposal,prior_lower_limit,prior_upper_limit)
    



def metropolis_sampling(theta_prev,nll_prev):
    theta_proposal = sampling_algorithm(theta_prev)
    
    # likelihood_proposal,freq_i = unnormalized_posterior(theta_proposal)
    likelihood_proposal = unnormalized_neg_log_posterior(theta_proposal)
    
    u = np.random.rand()
    
    # alpha_i = likelihood_proposal/nll_prev
    alpha_i = np.exp(nll_prev-likelihood_proposal)
    
    
    if alpha_i > u:
        is_new_proposal_accepted = 1
        return theta_proposal, likelihood_proposal, is_new_proposal_accepted,None
    else:
        is_new_proposal_accepted = 0
        return theta_prev, nll_prev,is_new_proposal_accepted, None




def RAM_sampling(theta_prev,nll_prev,S_prev,i):
    #Robust Adaptive Metropolis (RAM) - Bayesian Filtering and Smoothing (2023) - algorithm 16.6
    n = len(theta_prev)
    ri = np.random.randn(n)
    
    theta_proposal = theta_prev + S_prev@ri
    np.clip(theta_proposal,prior_lower_limit,prior_upper_limit)
    
    # likelihood_proposal,freq_i = unnormalized_posterior(theta_proposal)
    likelihood_proposal = unnormalized_neg_log_posterior(theta_proposal)
    
    u = np.random.rand()
    
    # alpha_i = likelihood_proposal/nll_prev
    alpha_i = np.exp(nll_prev-likelihood_proposal)
    alpha_i = min(alpha_i,1)
    
    I = np.eye(n)
    gamma_exponent = 0.51 #(between 0.5 (not included) and 1 (included))
    alpha_star_bar = 0.234 #♦ see book
    eta_i = i**-gamma_exponent
    SiSiT = S_prev@(I + eta_i*(alpha_i-alpha_star_bar)*np.outer(ri,ri)/np.dot(ri,ri))@S_prev.T
    Si = np.linalg.cholesky(SiSiT)
    
    if alpha_i > u:
        is_new_proposal_accepted = 1
        return theta_proposal, likelihood_proposal, is_new_proposal_accepted,None,Si
    else:
        is_new_proposal_accepted = 0
        return theta_prev, nll_prev,is_new_proposal_accepted, None,Si


# initial_theta = np.random.uniform(prior_lower_limit, prior_upper_limit,4)

initial_theta = np.ones_like(true_params)*1.0

max_iter = 50_000


nll_array = np.zeros(max_iter+1)
theta_iter = np.zeros([len(true_params),max_iter+1])

freq_list = [] 


theta_iter[:,0] = initial_theta
nll_array[0] =  unnormalized_neg_log_posterior(initial_theta)
freq_list.append(None)

iteration = 1

list_accepted_proposal = [1]
n_accepted_proposals = 1




#%% start or continue iterations

initial_std_step = 0.0005
S_prev = np.linalg.cholesky(np.eye(len(true_params))*initial_std_step**2)

while iteration < max_iter:
    
    # theta_new,nll_new, accepted,freq_i = \
    #     metropolis_sampling(theta_iter[:,iteration-1],
    #                         nll_array[iteration-1])
    theta_new,nll_new, accepted,freq_i,S_prev = \
        RAM_sampling(theta_iter[:,iteration-1],
                            nll_array[iteration-1],
                            S_prev,iteration)
    
    
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
    
    if iteration%5000 == 0:
        print("\n\n acceptance ratio ",n_accepted_proposals/iteration,
              "\n\n")




#%% plot inference

burnin = 1000
true_params

acceptance_vs_iter = np.array(list_accepted_proposal[burnin:])

    
import matplotlib.pyplot as plt

burnin = 1000
n_params = theta_iter.shape[0]
acceptance_vs_iter = np.array(list_accepted_proposal[burnin:])

# Parameter names for titles
param_names = ['Population mean (μ)', 'Population std (σ)'] + [f'Structure {i+1} stiffness' for i in range(n_params - 2)]

# Create figure with subplots for each parameter
fig, axes = plt.subplots(n_params, 4, figsize=(20, 4*n_params))
fig.suptitle('MCMC Diagnostics for All Parameters', fontsize=18, y=0.995)

for i in range(n_params):
    # Unnormalize samples for this parameter
    samples = theta_iter[i, burnin:iteration] * true_params[i]
    
    # Column 1: Histogram
    axes[i, 0].hist(samples, bins=50, alpha=0.7, edgecolor='black')
    axes[i, 0].axvline(true_params[i], color='red', linestyle='--', linewidth=2, label='True value')
    axes[i, 0].set_ylabel('Frequency', fontsize=10)
    axes[i, 0].set_title(f'{param_names[i]} - Histogram', fontsize=11)
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
    # Column 2: Trace plot
    axes[i, 1].plot(samples, linewidth=0.5, alpha=0.7)
    axes[i, 1].axhline(true_params[i], color='red', linestyle='--', linewidth=2, label='True value')
    axes[i, 1].set_ylabel('Parameter value', fontsize=10)
    axes[i, 1].set_xlabel('Iteration', fontsize=10)
    axes[i, 1].set_title(f'{param_names[i]} - Trace', fontsize=11)
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)
    
    # Column 3: Autocorrelation
    from matplotlib import pyplot as plt
    max_lag = min(500, len(samples) // 2)
    acf = np.correlate(samples - samples.mean(), samples - samples.mean(), mode='full')
    acf = acf[len(acf)//2:]
    acf = acf[:max_lag] / acf[0]
    axes[i, 2].plot(acf, linewidth=1)
    axes[i, 2].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[i, 2].axhline(0.05, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[i, 2].axhline(-0.05, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[i, 2].set_ylabel('ACF', fontsize=10)
    axes[i, 2].set_xlabel('Lag', fontsize=10)
    axes[i, 2].set_title(f'{param_names[i]} - Autocorrelation', fontsize=11)
    axes[i, 2].grid(True, alpha=0.3)
    axes[i, 2].set_ylim([-0.2, 1.0])
    
    # Column 4: Running mean
    running_mean = np.cumsum(samples) / (np.arange(len(samples)) + 1)
    axes[i, 3].plot(running_mean, linewidth=1)
    axes[i, 3].axhline(true_params[i], color='red', linestyle='--', linewidth=2, label='True value')
    axes[i, 3].set_ylabel('Running mean', fontsize=10)
    axes[i, 3].set_xlabel('Iteration', fontsize=10)
    axes[i, 3].set_title(f'{param_names[i]} - Running Mean', fontsize=11)
    axes[i, 3].legend()
    axes[i, 3].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig("MCMC_diagnostics_all_params.pdf")
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("MCMC Summary Statistics (unnormalized)")
print("="*70)
for i in range(n_params):
    samples = theta_iter[i, burnin:iteration] * true_params[i]
    print(f"\n{param_names[i]}:")
    print(f"  True value:      {true_params[i]:.4f}")
    print(f"  Posterior mean:  {samples.mean():.4f}")
    print(f"  Posterior std:   {samples.std():.4f}")
    print(f"  95% CI:          [{np.percentile(samples, 2.5):.4f}, {np.percentile(samples, 97.5):.4f}]")

# Overall acceptance rate
print("\n" + "="*70)
print(f"Overall acceptance rate: {acceptance_vs_iter.mean():.2%}")
print("="*70)
#%%



figure = corner.corner(theta_iter[:,burnin:iteration].T, 
                        show_titles=True)

plt.show()


print(true_params)

#%%

import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

# Unnormalize the samples
burnin = 1000
theta_samples = theta_iter[:, burnin:iteration].T * true_params  # Shape: (n_samples, n_params)

# Extract population and individual parameters
mu_population = theta_samples[:, 0]  # Mean of population
sigma_population = theta_samples[:, 1]  # Std of population
individual_stiffness = theta_samples[:, 2:]  # Individual stiffness for each structure

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Define stiffness range based on ACTUAL data range
k_min = individual_stiffness.min() - 0.1
k_max = individual_stiffness.max() + 0.1
k_range = np.linspace(k_min, k_max, 1000)

# Plot individual posteriors for each structure
n_structures = individual_stiffness.shape[1]
colors = plt.cm.tab10(np.arange(n_structures))

for i in range(n_structures):
    kde_i = gaussian_kde(individual_stiffness[:, i])
    density_i = kde_i(k_range)
    ax.plot(k_range, density_i, color=colors[i], label=f'Structure {i+1}', linewidth=2)
    
    # Add vertical line for true parameter of this structure
    true_k_i = true_params[i + 2]  # true_params[2:] are individual stiffness
    ax.axvline(true_k_i, color=colors[i], linestyle=':', linewidth=1.5, alpha=0.7)

# Plot population distribution
population_densities = np.zeros_like(k_range)
n_samples_to_average = min(500, len(mu_population))
sample_indices = np.random.choice(len(mu_population), n_samples_to_average, replace=False)

for idx in sample_indices:
    population_densities += norm.pdf(k_range, mu_population[idx], sigma_population[idx])

population_densities /= n_samples_to_average

ax.plot(k_range, population_densities, 'k--', linewidth=2.5, label='Population distribution')

# Add vertical line for true population mean
ax.axvline(true_params[0], color='black', linestyle='-.', linewidth=2, alpha=0.7, label='True population mean')

# Formatting
ax.set_xlabel('Stiffness coefficient - k', fontsize=12)
ax.set_ylabel('Probability density', fontsize=12)
ax.set_title('Individual posteriors and population distribution', fontsize=14)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"True params: {true_params}")
print(f"\nTrue population mean: {true_params[0]:.3f}")
print(f"True population std: {true_params[1]:.3f}")
print(f"Population mean (posterior): {np.mean(mu_population):.3f} ± {np.std(mu_population):.3f}")
print(f"Population std (posterior): {np.mean(sigma_population):.3f} ± {np.std(sigma_population):.3f}")
print(f"\nIndividual stiffness posteriors vs true values:")
for i in range(n_structures):
    true_val = true_params[i + 2]
    post_mean = np.mean(individual_stiffness[:, i])
    post_std = np.std(individual_stiffness[:, i])
    print(f"  Structure {i+1}: {post_mean:.3f} ± {post_std:.3f} (true: {true_val:.3f})")
