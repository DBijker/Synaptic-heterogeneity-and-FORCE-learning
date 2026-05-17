import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.stats import multivariate_normal, poisson, norm, expon, chi2, multinomial
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os
from numba import njit, prange

# fundamental functions:

# @njit(parallel=True, fastmath=True)
# def matvec_parallel(W, v):
#     n = W.shape[0]
#     out = np.zeros(n, dtype=v.dtype)
#     for i in prange(n):
#         s = 0.0
#         row = W[i]
#         for j in range(n):
#             s += row[j] * v[j]
#         out[i] = s
#     return out

@njit(fastmath=True)
def rate_network_numba(X, W, I, tau):
    return (-X + W @ np.tanh(X) + I)/tau

#runge-kutta is not necessary because we use a sufficient amount of time steps for the euler method, so the result of runge_kutta and euler should be close
#to each other, see Answers2exercises_CN_Research_Guide_A exercises task 2 question 4. 
#def runge_kutta_rate_network(network, W, f_activation):
def euler_rate_network(network, W, f_activation, N, t, x0, I, tau):
    
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    h = t[1] - t[0]
    for i in range(1, len(t)):
        x[i] = x[i-1] + h*network(f_activation, x[i-1], W, I[i-1], tau)#this function will not work now, because the rate_network does not have f_activation as input anymore
    return x    

# @njit(fastmath=True)
# def euler_rate_network_numba_full(W, tau, dt, t_steps, x0, I):
#     N = W.shape[1]
#     x = np.zeros((t_steps, N))
#     X = x0.copy()
#     for i in range(t_steps):
#         X = X + dt * rate_network_numba(X, W, I, tau)
#         x[i, :] = X
#     return x

# @njit(fastmath=True)
# def euler_rate_network_numba_final(W, tau, dt, t_steps, x0, I):
#     X = x0.copy()
#     for _ in range(t_steps):
#         X = X + dt * rate_network_numba(X, W, I, tau)
#     return X

# module.py - New dedicated full trajectory function, DOES WORK 20-10-2025
@njit(fastmath=True)
def euler_rate_network_numba_full_trajectory(W, tau, dt, t_steps, x0, I):
    N = len(x0)
    x = np.zeros((t_steps, N), dtype=x0.dtype)
    x[0, :] = x0 # Explicitly set the first row

    X = np.copy(x0)

    for i in range(1, t_steps):
        # select input (time varying or constant)
        if I.ndim == 2:
            Ii = I[i-1, :]
        else: 
            Ii = I
        
        X = X + dt * rate_network_numba(X, W, Ii, tau)

        x[i, :] = X # Explicitly assign the 1D array X to the 2D slice x[i, :]

    return x # x is always 2D

# @njit(fastmath=True)
# def euler_rate_network_numba(W, tau, dt, t_steps, x0, I, return_full = True):
#     """
#     Euler integration of the rate network (optimized with Numba)

#     Parameters:
#         W           : ndarray (N X N), Weight matrix
#         tau         : float, Time constant
#         dt          : float, Integration step size
#         t_steps     : int, Number of time steps to simulate
#         x0          : ndarray (N,), Initial state
#         I           : ndarray (t_steps x N) or (N,), Input array (constant or time dependent)
#         return_full : bool,
#                     : if True, return full trajectory (t_steps x N).
#                     : if False, return only the last state (N,)

#     Returns:
#         x           : ndarray, full trajectory or last state depending on 'return_full'.
#     """
#     N = len(x0)
#     if return_full:
#         x = np.zeros((t_steps, N), dtype=x0.dtype)
#         x[0] = x0

#     else:
#         x = np.copy(x0)

#     X = np.copy(x0)

#     for i in range(1, t_steps):
#         #select input (time varying or constant)
#         if I.ndim == 2:
#             Ii = I[i-1]
#         else: 
#             Ii = I
        
#         X = X + dt * rate_network_numba(X, W, Ii, tau)

#         if return_full:
#             x[i,:] = X

#     return x if return_full else X
    


def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) 

def weight_matrix_fast(N, MSH, gain, rng_weights):
    #identical values as the weight_matrix one, but ~1.2 faster
    p = 1
    mus = np.linspace(-MSH, MSH, N)

    # Generate full matrix of normal random numbers in one call
    W = rng_weights.normal(
        loc=mus[:, None],             # broadcast mus over columns
        scale=gain/np.sqrt(p*N),
        size=(N, N)
    )

    # Remove self-connections
    np.fill_diagonal(W, 0)
    return W


def weight_matrix(N, MSH, gain, rng_weights):

    "creating 1 heterogeneous matrix, with mus linspace(-bound, bound, N)"
    
    
    W = np.zeros((N,N))
    #the means of the distributions of the incoming weights of the neurons in the weight matrix, to make it an synaptic heterogeneous weight matrix
    mus = np.linspace(-MSH, MSH, N)
    
    #creating heterogeneous weight matrix, using a normal distribution

    #there misses something similar to the precomputed_weights as in the code for the heterogeneous weight matrices, there I made sure
    #that every state of matrix will have the same homogeneous base matrix with means for incoming weights of 0 for every neuron.

#check if corrected_weights addition is correct!!!
    # #calculating precomputed_weights to be able to use the same synaptic "homogeneous" weight matrix
    # precomputed_weights = rng.normal(loc = 0, scale = 1/np.sqrt(0.1*N), size = (N, N)) #check situations with 1*sigma (i.e. gain), 10*sigma (gain) and 0.1*sigma (gain)
    # # print(f"precomputed_weights before= {precomputed_weights}")
    # mus_precomputed_weights = precomputed_weights.mean(1)
    # # print(f"mus_precomputed_weights = {mus_precomputed_weights}")
    # corrected_weights = precomputed_weights - mus_precomputed_weights[:, np.newaxis]#[:,np.newaxis] to make mus_precomputed from (N,) to a (N,1)

    p = 1 # the probability or ratio of connections between neurons, in a fully connected network it should be p = 1

    for i in range(N):
        ################changing gain to 10, 1 or 0.1
        #W[i, :] = corrected_weights[i,:] + mus[i]  #check if this is correct!!!
        W[i, :] = rng_weights.normal(loc = mus[i], scale = gain/np.sqrt(p*N), size = N) #normal distribution
        
        W[i,i] = 0 #no self connection

    #introduce 90% sparsity mask
    #mask = rng_weights.choice([0,1], size=(N,N), p=[0.9, 0.1]) #90% zeros, 10% ones
    #W *= mask

    return W


#statistical functions:
def participation_ratio(X):
    """
    Compute the participation ratio using eigenvalues.    
    Parameters:
    X (numpy.ndarray): Input matrix (time points x neurons)    
    Returns:
    float: The participation ratio
    """
    # Center the data by subtracting the mean of each neuron
    X_centered = X - np.mean(X, axis=0)   
    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered.T) #the covariance step makes the matrix nxn so that the trace or the eigenvalues can be calculated.    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov_matrix)   
    # Compute participation ratio
    ParticipationRatio = (np.sum(eigenvalues)**2) / (np.sum(eigenvalues**2))    
    
    return ParticipationRatio

def lyapunov_exponent_and_activity(running_time, reset_steps, tau, perturbation, N, W, dt, rng_initial):

    last_3000_time = len(running_time) - 3000

    I = np.zeros((len(running_time), N))
    
    distances = []

    X0 = rng_initial.uniform(-1.0, 1.0, size=(1,N)) #rng_initial see farhad slack
    X_perturbed = X0 + perturbation/np.sqrt(N)
    dist_mag = np.linalg.norm(X_perturbed - X0)

    distances.append(dist_mag)
    
    X1 = np.zeros((len(running_time), X0.shape[1]))#X0.shape[1] = N
    X2 = np.zeros((len(running_time), X0.shape[1]))
    X1[0,:] = X0
    X2[0,:] = X_perturbed

    
    activity_X1 =[]
    activity_X2 = []
    
    

    activity_X1.append(X1[0, 0:1000:10])
    activity_X2.append(X2[0, 0:1000:10])





    for t in range(1, len(running_time)):
    
        X1[t] = X1[t-1] + dt*rate_network(tanh, X1[t-1], W, I[t-1], tau)
        X2[t] = X2[t-1] + dt*rate_network(tanh, X2[t-1], W, I[t-1], tau)

        #if t % 100 == 0: #every 100th activity of X1 and X2 gets saved
        if t >= last_3000_time:
            activity_X1.append(X1[t, 0:1000:10])
            activity_X2.append(X2[t, 0:1000:10])
        
        #compute distance
        dist_vec = X2[t] - X1[t]
        dist_mag = np.linalg.norm(dist_vec)
        distances.append(dist_mag)
        
        if t % reset_steps == 0:
            unit_vec = dist_vec / dist_mag
            X2[t] = X1[t] + perturbation*unit_vec

    activity_X1 = np.array(activity_X1)
    activity_X2 = np.array(activity_X2)
    
    distances = np.array(distances)
    # time_array = running_time[1:] #one less than full time array

    log_distances = np.log(distances)
    
    
    
    time_chunks = []
    lyapunov_exponents = []
    
    for i in range(0, len(log_distances), reset_steps):
        #ensure chunk has full length
        if i + reset_steps > len(log_distances):
            break
    
        #get time and log-distance chunk
        t_chunk = running_time[i:i + reset_steps]
        d_chunk = log_distances[i:i + reset_steps]
    
        #linear fit to log(distance) vs time
        slope, intercept, r_value, p_value, std_err = linregress(t_chunk, d_chunk)
    
        #store local lyapunov exponent
        lyapunov_exponents.append(slope)

    largest_lyapunov_exponent = np.mean(lyapunov_exponents)
    

    return largest_lyapunov_exponent, distances, activity_X1, activity_X2

@njit(fastmath=True)

def lyapunov_exponent_numba(running_time, reset_steps, tau, perturbation, N, W, dt, X0):

    """
    Numba accelarated Lyapuov exponent computation
    Arguments:
        running_time : 1D array of timesteps
        reset_steps  : time steps between resetting the distance between both X1 and X2 back to the original perturbation
        tau          : time constant
        perturbation : slight perturbation between X1 and X2 to check for divergence over time or convergence
        N            : number of neurons
        W            : weight matrix (N x N)
        dt           : integration timestep
        X0           : initial state vector (1D, length N)
    Returns:
        Largest Lyapunov Exponent (float)
    """

    steps = len(running_time)
    X1 = np.copy(X0)
    X2 = np.copy(X0 + perturbation / np.sqrt(N))

    distances = np.empty(steps, dtype=X0.dtype)
    distances[0] = np.linalg.norm(X2 - X1)

    # Euler integration loop
    I = np.zeros(N) #constant input (zero)
    for t in range(1, steps):
        X1 = X1 + dt * rate_network_numba(X1, W, I, tau)
        X2 = X2 + dt * rate_network_numba(X2, W, I, tau)

        dist_vec = X2 - X1
        dist_mag = np.linalg.norm(dist_vec)
        distances[t] = dist_mag

        if t% reset_steps == 0:
            unit_vec = dist_vec / dist_mag
            X2 = X1 + perturbation * unit_vec
    
    #compute the log_distances
    log_distances = np.log(distances)

    #Estimate local slopes (linear regression, numba-safe)
    # does exectly the same thing as linregress(...)[0], but is much faster and does not use memory and does not calculate extra stuff you do not need
    # linregress is also not supported by Numba
    lyaps = []
    for i in range(0, steps - reset_steps, reset_steps):
        t_chunk = running_time[i:i+reset_steps]
        d_chunk = log_distances[i:i+reset_steps]

        x_mean = t_chunk.mean()
        y_mean = d_chunk.mean()

        #slope of linear regression: m = sum((x_i - x_mean)*(y_i - y_mean))/(sum((x_i - x_mean)^2))
        num = ((t_chunk - x_mean) * (d_chunk - y_mean)).sum() #numerator: dot product between mean centered t_chunk and mean centered d_chunk
        den = ((t_chunk - x_mean) ** 2).sum() #denominator: dotproduct of t_chunk with itself

        slope = num/den #gives the least squares slope
        lyaps.append(slope)

    return np.mean(np.array(lyaps))


def lyapunov_exponent(running_time, reset_steps, tau, perturbation, N, W, dt, rng_initial):

    """
    Lyapunov exponent computation OLD SLOW VERSION   
    Arguments:
        running_time : 1D array of timesteps
        reset_steps  : time steps between resetting the distance between both X1 and X2 back to the original perturbation
        tau          : time constant
        perturbation : slight perturbation between X1 and X2 to check for divergence over time or convergence
        N            : number of neurons
        W            : weight matrix (N x N)
        dt           : integration timestep
        X0           : initial state vector (1D, length N)
    Returns:
        Largest Lyapunov Exponent (float)
    """

    I = np.zeros((len(running_time), N))
    
    X0 = rng_initial.uniform(-1.0, 1.0, size=(1,N)) #rng_initial see farhad slack
    X_perturbed = X0 + perturbation/np.sqrt(N)
    dist_mag = np.linalg.norm(X_perturbed - X0)
    
    X1 = np.zeros((len(running_time), X0.shape[1]))#X0.shape[1] = N
    X2 = np.zeros((len(running_time), X0.shape[1]))
    X1[0,:] = X0
    X2[0,:] = X_perturbed

    
    distances = []

    distances.append(dist_mag)


    for t in range(1, len(running_time)):
    
        X1[t] = X1[t-1] + dt*rate_network(tanh, X1[t-1], W, I[t-1], tau)
        X2[t] = X2[t-1] + dt*rate_network(tanh, X2[t-1], W, I[t-1], tau)
        
        #compute distance
        dist_vec = X2[t] - X1[t]
        dist_mag = np.linalg.norm(dist_vec)
        distances.append(dist_mag)
        
        if t % reset_steps == 0:
            unit_vec = dist_vec / dist_mag
            X2[t] = X1[t] + perturbation*unit_vec


    distances = np.array(distances)
    log_distances = np.log(distances)
    
    #parameters
    
    
    time_chunks = []
    lyapunov_exponents = []
    
    for i in range(0, len(log_distances), reset_steps):
        #ensure chunk has full length
        if i + reset_steps > len(log_distances):
            break
    
        #get time and log-distance chunk
        t_chunk = running_time[i:i + reset_steps]
        d_chunk = log_distances[i:i + reset_steps]
    
        #linear fit to log(distance) vs time
        slope, intercept, r_value, p_value, std_err = linregress(t_chunk, d_chunk)
    
        #store local lyapunov exponent
        lyapunov_exponents.append(slope)

    largest_lyapunov_exponent = np.mean(lyapunov_exponents)
    

    return largest_lyapunov_exponent

def activities(running_time, reset_steps, tau, perturbation, N, W, dt, rng_initial):
    
    last_3000_time = len(running_time) - 3000

    I = np.zeros((len(running_time), N))
    
    X0 = rng_initial.uniform(-1.0, 1.0, size=(1,N)) #rng_initial see farhad slack
    X_perturbed = X0 + perturbation/np.sqrt(N)
    
    X1 = np.zeros((len(running_time), X0.shape[1]))#X0.shape[1] = N
    X2 = np.zeros((len(running_time), X0.shape[1]))
    X1[0,:] = X0
    X2[0,:] = X1[0,:] + perturbation/np.sqrt(N)
    

    activity_X1 =[]
    activity_X2 = []

    activity_X1.append(X1[0, 0:1000:10])
    activity_X2.append(X2[0, 0:1000:10])

    for t in range(1, len(running_time)):
    
        X1[t] = X1[t-1] + dt*rate_network(tanh, X1[t-1], W, I[t-1], tau)
        X2[t] = X2[t-1] + dt*rate_network(tanh, X2[t-1], W, I[t-1], tau)

        #if t % 100 == 0: #every 100th activity of X1 and X2 gets saved
        if t >= last_3000_time:
            activity_X1.append(X1[t, 0:1000:10])
            activity_X2.append(X2[t, 0:1000:10])
        
        if t % reset_steps == 0:
            X2[t] = X1[t] + perturbation

    activity_X1 = np.array(activity_X1)
    activity_X2 = np.array(activity_X2)
    

    return  activity_X1, activity_X2
