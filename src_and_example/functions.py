#!/usr/bin/env python3
# coding: utf-8
# all functions for the calculation of the network state, the functional and its gradient, the network initialization.
#only works for control matrix=identity matrix and no limit/bounds for the control


import sys

import scipy.sparse as sp
import numpy as np
from numpy import Inf
from numba import jit
import pickle


def set_parameters(goal,bifurcation,parentDir):
    """Set all needed node and network parameters

        Parameters
        ----------
        goal : string, can be 'sync' or 'switch'
            the control task
        bifurcation : string, can be 'low' or 'high'
            bifurcation lne close to which parameters are chosen
        filedir : string
            directory of file in which the code and the folder data lie

        Returns
        -------
        parameters: dictionary
            dictionary with all needed node and network parameters
    """
    #single node parameters
    parameters={
        'alpha':3.0,
        'beta':4.0,
        'gamma':3./2,
        'delta':0,
        'epsilon':0.5,
        'tau':20.0
    }
    #position in state space
    if goal=='sync':
        parameters['sigma']=0.025
        if bifurcation=='low':
            parameters['mu']=0.7
        elif bifurcation=='high':
            parameters['mu']=1.3
    if goal=='switch':
        if bifurcation=='low':
            parameters['mu']=0.378
            parameters['sigma']=0.21
        elif bifurcation=='high':
            parameters['mu']=1.22
            parameters['sigma']=0.26

    #adjacency matrix
    A= np.load(parentDir+"/src_and_example/in_data/SC_m12av_voxNorm_th20.npy")
    parameters['A'] = A
    parameters['N'] = A.shape[0] #number of nodes

    return parameters

def set_random_initial_conds_for_nodes(bifurcation,noise_strength,tsteps,d,dt,parameters,realizations,noise):
    """Set initial conditions for all nodes randomly.

        Parameters
        ----------
        bifurcation : string, can be 'low' or 'high'
            bifurcation lne close to which parameters are chosen
        noise_strength : float
            strenght f the Gaussian normalized white noise
        tsteps : int
            number of time integration steps
        d : int
            dimension of an uncoupled node (d=2 for FHN)
        dt : time integration stepsize
            strenght f the Gaussian normalized white noise
        parameters : dictionary
            includes all node and netwrok parameters
        realizations : int
            number of noise realizations
        noise : array of shape (realizations,tsteps,N) if noise_strength>0, or 0 else
            noise realizations

        Returns
        -------
        node_ic: array of size (d,N)
            initial conditions for all nodes
    """
    #unpack parameters
    alpha=parameters['alpha']
    beta=parameters['beta']
    gamma=parameters['gamma']
    delta=parameters['delta']
    epsilon=parameters['epsilon']
    tau=parameters['tau']
    sigma=parameters['sigma']
    mu=parameters['mu']
    A=parameters['A']
    N=parameters['N']
    nocontrol=np.zeros((tsteps+4000,N))

    #set random initials on circle with set amplitudes
    if bifurcation=='low':
        ini_y=np.random.uniform(size= N)/2+0.1
        ini_z=np.random.uniform(size= N)/5+.5
    elif bifurcation=='high':
        ini_y=np.random.uniform(size= N)
        ini_z=np.random.uniform(size= N)*0.3+1.0
    
    ini_transient=np.dstack((ini_y,ini_z)).transpose()
    ini_transient=ini_transient.reshape(2,N)

    #transient, use last value of transient as initial condition
    if noise_strength==0.0:
        node_ic=runge_kutta_FHN_network(ini_transient, nocontrol,  tsteps+4000 , d , dt , N ,alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)[-1]
    else:
        node_ic=runge_kutta_FHN_network_noisy(ini_transient, nocontrol,  tsteps+4000 , d , dt , N ,alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,noise.reshape(realizations*tsteps,N))[-1]
    return node_ic

def initialize_noise(noise_strength,dt,realizations,tsteps,N):
    """Initialize additive Gaussian white noise. Returns array of noise realizations with shape (realizations,tsteps,N).
    """
    sqrt_dt=dt**(1/2)
    noise =  (noise_strength/sqrt_dt)*np.random.normal(size=(realizations,tsteps,N))
    return noise

def check_parameters(goal,bifurcation,switchfromto,noise_strength,I_p,I_e,I_s,control):
    """Checks that all aparameters are chosen compatible with code.
    """


    if goal!='switch' and goal!='sync':
        raise NameError("parameter goal should be 'switch' or 'sync'")
    if goal=='switch':
        if switchfromto!='lowtohigh' and switchfromto!='hightolow':
            raise NameError("parameter switchfromto should be 'lowtohigh' or 'hightolow'")
    if bifurcation!='low' and bifurcation!='high':
        raise NameError("parameter bifurcation should be 'low' or 'high'")
    if noise_strength<0:
        raise NameError("parameter noise_strength should be larger equal 0")
    if noise_strength>0 and goal=='switch':
        raise NameError("the scenario with noisy dynamics and the control goal to switch states is not coded")
    if I_p<0:
        raise NameError("parameter I_p should be larger equal 0")
    if I_e<0:
        raise NameError("parameter I_e should be larger equal 0")
    if I_s<0:
        raise NameError("parameter I_s should be larger equal 0")
    if I_s>1e-10:
        controlsum = np.sum(control**2, axis=0)**(1.0 / 2.0)
        if any(cs<1e-8 for cs in controlsum):
            raise NameError("initial control input should have entries larger 0")

def read_node_inis_from_file(bifurcation,switchfromto,fileDir,tsteps,d,dt,parameters):
    """Load file with predefines initial conditions for all nodes

        Parameters
        ----------
        bifurcation : string, can be 'low' or 'high'
            bifurcation lne close to which parameters are chosen
        switchfromto : string, can be 'lowtohigh' or 'hightolow'
            low to high means that the low state is the initial state and the high state the target state and vice versa
        filedir : string
            direction of file in which the code and the folder data lie
        tstaps : int
            number of time integration steps
        d : int
            dimension of an uncoupled node (d=2 for FHN)
        dt : time integration stepsize
            strenght f the Gaussian normalized white noise
        parameters : dictionary
            includes all node and netwrok parameters

        Returns
        -------
        node_ic: array of shape (d,N)
            initial conditions for all nodes
        target_state: array of shape (tsteps,N)
            the state we want to achieve with the control
    """
    #unpack parameters
    alpha=parameters['alpha']
    beta=parameters['beta']
    gamma=parameters['gamma']
    delta=parameters['delta']
    epsilon=parameters['epsilon']
    tau=parameters['tau']
    sigma=parameters['sigma']
    mu=parameters['mu']
    A=parameters['A']
    N=parameters['N']
    control=np.zeros((tsteps,N))

    with open(fileDir+'/in_data/node_inis_'+bifurcation+'_bifurcation', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    ic_state_low=content['state_low']
    ic_state_high=content['state_high']
    if switchfromto=='lowtohigh':
        node_ic=ic_state_low
        target_state=runge_kutta_FHN_network(ic_state_high, control,  tsteps , d , dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)[:,0,:]
    elif switchfromto=='hightolow':
        node_ic=ic_state_high
        target_state=runge_kutta_FHN_network(ic_state_low, control ,  tsteps , d , dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)[:,0,:]
    return node_ic,target_state        

def plot_runge_kutta(f_func,control,**args): 
    """Runge kutta of 4th order.

        Parameters
        ----------
        control : array of shape (tsteps,N)
            control input to network
        kwargs
            dictionary of all other parameters to be passed to function ODE_FHN_network() or ODE_FHN_network_noisy()

        Returns
        -------
        x : array of shape (tsteps,d,N)
            numerical solution of the network ODE with FHN nodes with initial conditions ini.

    """
    #1. load args
    tsteps = args['tsteps'] #timesteps, int
    dt = args['dt'] #stepsize of timesteps, float
    N = args['N'] #number of nodes, int
    d = args['d'] #dimension of oscillator dynamnode_ics, int
    alpha = args['alpha'] #parameter of FHN oscillator, float
    beta = args['beta'] #parameter of FHN oscillator, float
    gamma = args['gamma'] #parameter of FHN oscillator, float
    delta = args['delta'] #parameter of FHN oscillator, float
    epsilon = args['epsilon'] #parameter of FHN oscillator, float
    tau = args['tau'] #parameter of FHN oscillator, float
    mu = args['mu'] #parameter of FHN oscillator, float
    sigma = args['sigma'] #parameter of FHN oscillator, float
    A = args['A'] #adjacency matrix, array shape(N,N)
    node_ic= args['node_ic'] #initial conditions of the network dynamnode_ics, array shape(d,N)
    noise=args['noise']
    x=np.zeros((tsteps,d,N))
    x[0]=node_ic 
    if f_func==ODE_FHN_network_noisy:
        for ts in np.arange(0,tsteps-1):
            k1 = dt* f_func(x[ts],ts,control,tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,noise[ts])
            k2 = dt* f_func(x[ts] + k1/2 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,noise[ts])
            k3 = dt* f_func(x[ts] + k2/2 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,noise[ts])
            k4 = dt* f_func(x[ts] + k3 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu, sigma, A,noise[ts])
            x[ts+1]= x[ts] + (1/6)* (k1 + 2*k2 + 2*k3 + k4)

    elif f_func==ODE_FHN_network:
        for ts in np.arange(0,tsteps-1):
            k1 = dt* f_func(x[ts],ts,control,tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)
            k2 = dt* f_func(x[ts] + k1/2 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)
            k3 = dt* f_func(x[ts] + k2/2 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)
            k4 = dt* f_func(x[ts] + k3 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu, sigma, A)
            x[ts+1]= x[ts] + (1/6)* (k1 + 2*k2 + 2*k3 + k4)
    else:
        raise NameError('f_func should be ODE_FHN_network_noisy or ODE_FHN_network')
    return x

@jit(nopython=True)
def covariance_matrix(x , tsteps , N):
    """Covariance matrix

        Parameters
        ----------
        x : array_like
            2D, Contains multiple variables and observations. Each column of m represents a variable, and each row a single observation of all those variables.
        tsteps : int
            number of rows of x
        N : int
            number of columns of x

        Returns
        -------
        covariance_mat: array of shape (N,N)
            covariance matrix
    """
    means = (1/tsteps) * np.sum(x,axis=0)
    xcc = (x-means)
    covariance_mat= np.dot(xcc.transpose(), xcc)

    return covariance_mat


@jit(nopython=True)
def cross_correlation_matrix(x , tsteps , N):
    """Normalized pairwise cross-correlation matrix
       (1/tsteps) * Sum_{t=0}^{tsteps-1} (x_i(t)-x_mean)*(x_j(t)-x_mean)/(std(x_i)*std(x_j)

        Parameters
        ----------
        x : array_like
            2D, Contains multiple variables and observations. Each column of m represents a variable, and each row a single observation of all those variables.
        tsteps : int
            number of rows of x
        N : int
            number of columns of x

        Returns
        -------
        cross_correlation_mat: array of shape (N,N)
            normalized pairwise cross-correlation matrix
    """
    means = (1/tsteps) * np.sum(x,axis=0)
    stds =  ((x-means)**2).sum(axis=0)**(1/2) 
    stdsgrid= np.ones((N,1))*stds
    stdsgrid= stdsgrid.transpose()*stdsgrid
    xcc = (x-means)
    cross_correlation_mat= np.dot(xcc.transpose(), xcc)/stdsgrid

    return cross_correlation_mat

@jit(nopython=True)
def f_switch(control,tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,  target_output ,node_ic,realizations,noise):
    """Returns the value of the cost functional for the control given. f= precision + energy + sparsity
    f= (1/2) sum_{t=0}^{tsteps-1} [ I_p(t)*(x(t)-x_target(t))**2 ] + (1/2) sum_{t=0}^{tsteps-1} [ I_e*(control**2)] + I_s * sum_{t=0}^{tsteps-1} [ (control**2) ]**(1/2)
    To be used if the control goal is to switch between states!

        Parameters
        ----------
        control : array of shape (tsteps*N,)
            the control signal
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        N : int
            number of nodes
        I_p : array of shape (tsteps,N) 
            weight of precision term of cost functional
        I_e : float
            weight of energy term of cost functional
        I_s : float
            weight of sparsity term of cost functional
        alpha , beta, gamma, delta, epsilon, tau, mu ,sigma : floats
            parameters of the FHN-oscillator
        A : array shape (N,N)
            adjacency matrix
        target_output : array of shape (tsteps,N) 
            the oscillator's activity of the desired state (only the 1st dimension of the FHN oscillators is considered)
        node_ic : array shape (d,N)
            initial conditions of the controlled state
        realizations : int
            number of noise realizations
        noise : array of shape (realizations,tsteps,N) if noise_strength>0, or 0 else
            noise realizations

        Returns
        -------
        (f1+f2+f3) : float
            evaluation of the cost functional for the given control
    """

    control=control.reshape(tsteps,N)
    state=runge_kutta_FHN_network(node_ic, control, tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)
    state_diff_2=(state[:,0,:]-  target_output)**2
    f1= 1/2  * (I_p*state_diff_2).sum() 
    f2= I_e/2 * (control**2).sum() 
    if I_s>1e-10:
        controlsum = np.sum(control**2, axis=0)**(1.0 / 2.0)#
        f3 = I_s  *np.sum(controlsum)
    else:
        f3=0
    return (f1+f2+f3) 


@jit(nopython=True)
def f_sync(control,tsteps , d , dt , N ,   I_p ,   I_e ,   I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,  target_output ,node_ic,realizations,noise):
    """Returns the value of the cost functional for the control given. f= precision + energy + sparsity
    f= (1/4) sum_{i=0}^{N-1} sum_{j=0}^{N-1} I_p* [ cross-corr_matrix_ij(x)-target_cross_corr ]**2 + (1/2) sum_{t=0}^tsteps [ I_e*(control**2)] + I_s * sum_{t=0}^tsteps [ (control**2) ]**(1/2)
    To be used  if the control goal is to synchronize the network dynamics!

        Parameters
        ----------
        control : array of shape (tsteps*N,)
            the control signal
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        N : int
            number of nodes
        I_p : float 
            weight of precision term of cost functional
        I_e : float
            weight of energy term of cost functional
        I_s : float
            weight of sparsity term of cost functional
        alpha , beta, gamma, delta, epsilon, tau, mu ,sigma : floats
            parameters of the FHN-oscillator
        A : array shape (N,N)
            adjacency matrix
        target_output : array of shape (tsteps,N) 
            the oscillator's activity of the desired state (only the 1st dimension of the FHN oscillators is considered)
        node_ic : array shape (d,N)
            initial conditions of the controlled state
        realizations : int
            number of noise realizations
        noise : array of shape (realizations,tsteps,N) if noise_strength>0, or 0 else
            noise realizations

        Returns
        -------
        (f1+f2+f3) : float
            evaluation of the cost functional for the given control
    """
    control=control.reshape(tsteps,N)
    state=runge_kutta_FHN_network(node_ic, control, tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)[:,0,:]
    cc = cross_correlation_matrix(state , tsteps , N)
    f1 =   I_p * (1/4)* np.sum((cc -   target_output)**2)
    f2=   I_e/2 * (control**2).sum() 
    if   I_s>1e-10:
        controlsum= np.sum(control**2, axis=0)**(1.0 / 2.0)#
        f3 =   I_s  *np.sum(controlsum)
    else:
        f3=0
    return (f1+f2+f3) 


@jit(nopython=True)
def f_sync_noisy(control,tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,  target_output ,node_ic,realizations,noise):
    """Returns the value of the cost functional for the control given. f= <precision> + energy + sparsity
    For the precision term the mean over all noise realizations (denoted by <>) is used.
    f= <(1/4) sum_{i=0}^{N-1} sum_{j=0}^{N-1} I_p* [ cross-corr_matrix_ij(x)-target_cross_corr ]**2> + (1/2) sum_{t=0}^tsteps [ I_e*(control**2)] + I_s * sum_{t=0}^tsteps [ (control**2) ]**(1/2)
    To be used  if the control goal is to synchronize the network dynamics and the noise strength is larger 0!

        Parameters
        ----------
        control : array of shape (tsteps*N,)
            the control signal
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        N : int
            number of nodes
        I_p : float 
            weight of precision term of cost functional
        I_e : float
            weight of energy term of cost functional
        I_s : float
            weight of sparsity term of cost functional
        alpha , beta, gamma, delta, epsilon, tau, mu ,sigma : floats
            parameters of the FHN-oscillator
        A : array shape (N,N)
            adjacency matrix
        target_output : array of shape (tsteps,N) 
            the oscillator's activity of the desired state (only the 1st dimension of the FHN oscillators is considered)
        node_ic : array shape (d,N)
            initial conditions of the controlled state
        realizations : int
            number of noise realizations
        noise : array of shape (realizations,tsteps,N) if noise_strength>0, or 0 else
            noise realizations

        Returns
        -------
        (f1+f2+f3) : float
            evaluation of the cost functional for the given control
    """
    control=control.reshape(tsteps,N)
    #compute R realizations of noisy state x(u) and mean(f1)
    f1=0
    for realization in np.arange(realizations):
        state=runge_kutta_FHN_network_noisy(node_ic, control, tsteps , d , dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,noise[realization])[:,0,:]
        cc = cross_correlation_matrix(state , tsteps , N)
        f1 += np.sum((cc - target_output)**2)
    f1=I_p * (1/4)* f1/realizations
    f2=I_e/2 * (control**2).sum() 
    if I_s>1e-10:
        controlsum= np.sum(control**2, axis=0)**(1.0 / 2.0)#
        f3 = I_s *np.sum(controlsum)
    else:
        f3=0
    return (f1+f2+f3) 

@jit(nopython=True)
def nabla_x_f_switch(state, tsteps,N,  target_output,  I_p):
    """Multidimensional derivative of the cost functional given in function f_switch() with respect to the oscillators states.
    nabla_x f_switch [t,k]= del/(del x_{k,t}) f_sync= I_p[t,k]* (state-target_output)[t,k]
    To be used if the control goal is to switch between states!

        Parameters
        ----------
        state : array of shape (tsteps,N)
            the first dimension of the FHN dynamcis
        tsteps : int
            number of timesteps
        N : int
            number of nodes
        target_output : array of shape (tsteps,N) 
            the oscillator's activity of the desired state (only the 1st dimension of the FHN oscillators is considered)
        I_p : array of shape (tsteps,N)
            weight of precision term of cost functional

        Returns
        -------
        out : array of shape (tsteps,N)
            gradient of the cost functional with respect to the state evaluated at the given state
    """
    out=I_p* (state-target_output)
    return out

@jit(nopython=True)
#derivaive of f1= sum((crosscor - crosscor_target)**2)
#with loops
def nabla_x_f_sync(state, tsteps,N,  target_output,  I_p):
    """Multidimensional derivative of the cost functional given in function f_sync() with respect to the oscillators states.
    nabla_x f_sync[t,k]= del/(del x_{k,t}) f_sync=  sum_{i=0}^{N-1} (cross_cor_matrix[k,i]-  target_output[k,i]) * ( (x[ts,i]-mean(x[i])) - (1/std(x[k])**2) *(x[ts,k]-mean(x[k])) *covaiance_matrix[k,i]) /( std(x[i])*stds(x[k]) )
    To be used if the control goal is to synchronize the network dynamics!

        Parameters
        ----------
        state : array of shape (tsteps,N)
            the first dimension of the FHN dynamcis
        tsteps : int
            number of timesteps
        N : int
            number of nodes
        target_output : array of shape (N,N) 
            the target cross-correlation
        I_p : float
            weight of precision term of cost functional

        Returns
        -------
        out : array of shape (tsteps,N)
            gradient of the cost functional with respect to the state evaluated at the given state
    """
    
    means = (1/tsteps) * np.sum(state,axis=0)
    out=np.zeros((tsteps,N))
    xm= state-means
    cc=cross_correlation_matrix(state , tsteps , N)
    stds =  (((state-means)**2).sum(axis=0))**(1/2) 
    cov_mat=covariance_matrix(state,tsteps,N)
    for ts in np.arange(tsteps):
        for k in np.arange(N):
            for i in np.arange(N):
                out[ts,k] += (cc[k,i]-  target_output[k,i]) * ( xm[ts,i] - (1/stds[k]**2) *xm[ts,k] *cov_mat[k,i]) /( stds[i]*stds[k] )
    out=out*I_p
    return out


@jit(nopython=True)
def nabla_u_f_sparsity(control,adjoint,tsteps,dt,N,I_e,I_s):
    """Returns array with shape (tsteps,N): the derivative of the gradient of the sparsity term of the cost functional with respect to the control
    if integrate_0^T[ (control(t)**2) dt]**(1/2)>small: fprime_sparsity(t) = I_s * control(t)/integrate_0^T[ (control(t)**2) dt]**(1/2) 
    else:                                               fprime_sparsity(t) = -( adjoint(t) -I_e*control(t)) ( so that fprime= fprime_precision + fprime_energy + fprime_sparsity = 0 )

        Parameters
        ----------
        control : array of shape (tsteps,N)
            the control signal
        adjoint : array of shape (tsteps,N)
            the first dimension of the adjoint state for all times 0<=t<tsteps
        tsteps : int
            number of timesteps
        dt : float
            stepsize of integration
        N : int
            number of nodes
        I_e : float
            weight of energy term of cost functional
        I_s : float
            weight of sparsity term of cost functional

        Returns
        -------
        out : array of shape (tsteps,N)
            gradient of the sparsity term of the cost functional with respect to the control, evaluated at the given control
    """
    small=1e-10 #value where the gradient is cut off, since otherwise errors might arise due to division between small numbers)
    out=np.zeros((tsteps,N))
    controlsum= np.sum(control**2, axis=0)**(1.0 / 2.0)#vecnorm(control,2,0)

    for i in np.arange(N):
        for j in np.arange(tsteps):
            if controlsum[i]>small:
                out[j,i]=I_s*control[j,i]/controlsum[i]
            else:
                out[j,i]= -adjoint[j,i] - I_e* control[j,i]
    return out



@jit(nopython=True)
#nabla_u of the functional
def nabla_u_f(control,adjoint,tsteps,dt,N,I_e,I_s):
    """Returns array with shape (tsteps,N): the derivative of the gradient with respect to the control
    fprime(t)=  I_p(t)*(x(t)-x_desired(t)) +  I_e*control(t) + I_s * control(t)/integrate_0^T[ (control(t)**2) dt]**(1/2)

        Parameters
        ----------
        control : array of shape (tsteps,N)
            the control signal
        adjoint : array of shape (tsteps,N)
            the first dimension of the adjoint state
        tsteps : int
            number of timesteps
        dt : float
            stepsize of integration
        N : int
            number of nodes
        I_e : float
            weight of energy term of cost functional
        I_s : float
            weight of sparsity term of cost functional

        Returns
        -------
        out : array of shape (tsteps,N)
            gradient of the cost functional given in functions f_switch() and f_sync() with respect to the control, evaluated at the given control u.
    """
    nabla_u_energy = I_e* control
    if I_s>1e-10:
        nabla_u_sparsity = nabla_u_f_sparsity(control,adjoint,tsteps,dt,N,I_e,I_s)
        return nabla_u_energy + nabla_u_sparsity
    else:
        return nabla_u_energy


@jit(nopython=True)
def fprime_switch(control,tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,  target_output ,node_ic,realizations,noise):
    """Gradient of the cost functional given in function f_switch() avaluated at the given control u.
    fprime_k(t)= del/(del x_k) (t) f(t)= nabla_x_f_switch[t,k] + nabla_u_f[t,k] for 0<=t<tsteps and 0<=k<N
    To be used if the control goal is to switch between states!

        Parameters
        ----------
        control : array of shape (tsteps*N,)
            the control signal
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        N : int
            number of nodes
        I_p : array of shape (tsteps,N) 
            weight of precision term of cost functional
        I_e : float
            weight of energy term of cost functional
        I_s : float
            weight of sparsity term of cost functional
        alpha , beta, gamma, delta, epsilon, tau, mu ,sigma : floats
            parameters of the FHN-oscillator
        A : array shape (N,N)
            adjacency matrix
        target_output : array of shape (tsteps,N) 
            the oscillator's activity of the desired state (only the 1st dimension of the FHN oscillators is considered)
        node_ic : array shape (d,N)
            initial conditions of the controlled state
        realizations : int
            number of noise realizations
        noise : array of shape (realizations,tsteps,N) if noise_strength>0, or 0 else
            noise realizations

        Returns
        -------
        out : array of shape (tsteps,N)
            gradient of the cost functional given in functions f_switch(), evaluated at the given control u.
    """

    control=control.reshape(tsteps,N)
    state=runge_kutta_FHN_network(node_ic, control , tsteps , d ,dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)[:,0,:]
    nabla_x=nabla_x_f_switch(state, tsteps,N,  target_output,  I_p)
    adjoint=AS(tsteps , d , dt , N ,alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,state, nabla_x)
    out=(nabla_u_f(control,adjoint,tsteps,dt,N,I_e,I_s) + adjoint )
    out=out.reshape(tsteps*N)
    return  out


#Gradient of the functional
@jit(nopython=True)
def fprime_sync(control,tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,  target_output ,node_ic,realizations,noise):
    """Gradient of the cost functional given in function f_sync() evaluated at the given control u.
    fprime_k(t)= del/(del x_k) (t) f(t)= nabla_x_f_sync[t,k] + nabla_u_f[t,k] for 0<=t<tsteps and 0<=k<N
    To be used if the control goal is to synchronize the network dynamics!

        Parameters
        ----------
        control : array of shape (tsteps*N,)
            the control signal
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        N : int
            number of nodes
        I_p : array of shape (tsteps,N) 
            weight of precision term of cost functional
        I_e : float
            weight of energy term of cost functional
        I_s : float
            weight of sparsity term of cost functional
        alpha , beta, gamma, delta, epsilon, tau, mu ,sigma : floats
            parameters of the FHN-oscillator
        A : array shape (N,N)
            adjacency matrix
        target_output : array of shape (tsteps,N) 
            the oscillator's activity of the desired state (only the 1st dimension of the FHN oscillators is considered)
        node_ic : array shape (d,N)
            initial conditions of the controlled state
        realizations : int
            number of noise realizations
        noise : array of shape (realizations,tsteps,N) if noise_strength>0, or 0 else
            noise realizations

        Returns
        -------
        out : array of shape (tsteps,N)
            gradient of the cost functional given in functions f_sync(), evaluated at the given control u.
    """

    control=control.reshape(tsteps,N)
    state=runge_kutta_FHN_network(node_ic, control , tsteps , d ,dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)[:,0,:]
    nabla_x=nabla_x_f_sync(state, tsteps,N,  target_output,  I_p)
    adjoint=AS(tsteps , d , dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,state, nabla_x)
    out=(nabla_u_f(control,adjoint,tsteps,dt,N,I_e,I_s) + adjoint )
    out=out.reshape(tsteps*N)
    return  out

@jit(nopython=True)
def fprime_sync_noisy(control,tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,  target_output ,node_ic,realizations,noise):
    """Gradient of the cost functional given in function f_sync() evaluated at the given control u.
    fprime_k(t)= del/(del x_k) (t) f(t)= <nabla_x_f_sync[t,k]> + nabla_u_f[t,k] for 0<=t<tsteps and 0<=k<N and the mean over noise realizations <>.
    To be used if the control goal is to synchronize the noisy network dynamics!

        Parameters
        ----------
        control : array of shape (tsteps*N,)
            the control signal
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        N : int
            number of nodes
        I_p : array of shape (tsteps,N) 
            weight of precision term of cost functional
        I_e : float
            weight of energy term of cost functional
        I_s : float
            weight of sparsity term of cost functional
        alpha , beta, gamma, delta, epsilon, tau, mu ,sigma : floats
            parameters of the FHN-oscillator
        A : array shape (N,N)
            adjacency matrix
        target_output : array of shape (tsteps,N) 
            the oscillator's activity of the desired state (only the 1st dimension of the FHN oscillators is considered)
        node_ic : array shape (d,N)
            initial conditions of the controlled state
        realizations : int
            number of noise realizations
        noise : array of shape (realizations,tsteps,N) if noise_strength>0, or 0 else
            noise realizations

        Returns
        -------
        out : array of shape (tsteps,N)
            gradient of the cost functional given in functions f_sync_noisy(), evaluated at the given control u.
    """


    control=control.reshape(tsteps,N)
    adjoint=np.zeros((tsteps,N))

    #compute R realizations of noisy state x(u) and mean(f1)
    for realization in np.arange(realizations):
        state=runge_kutta_FHN_network_noisy(node_ic, control , tsteps , d ,dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, noise[realization])[:,0,:]
        nabla_x=nabla_x_f_sync(state, tsteps,N,  target_output,  I_p)
        adjoint+=AS(tsteps , d , dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,state, nabla_x)
    adjoint=adjoint/realizations 
    out=(nabla_u_f(control,adjoint,tsteps,dt,N,I_e,I_s) + adjoint )
    out=out.reshape(tsteps*N)
    return  out


@jit(nopython=True)
def runge_kutta_FHN_network(ini, control, tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A): 
    """Runge kutta of 4th order.

        Parameters
        ----------
        ini : array of shape (d,N)
            boundary conditions at t=0
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        all other parameters are only passed on to function ODE_FHN_network()

        Returns
        -------
        x : array of shape (tsteps,d,N)
            numerical solution of the network ODE with FHN nodes with initial conditions ini.

    """
    x=np.zeros((tsteps,d,N))
    x[0]=ini    
    for ts in np.arange(0,tsteps-1):
        k1 = dt* ODE_FHN_network(x[ts],ts,control,tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)
        k2 = dt* ODE_FHN_network(x[ts] + k1/2 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)
        k3 = dt* ODE_FHN_network(x[ts] + k2/2 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A)
        k4 = dt* ODE_FHN_network(x[ts] + k3 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu, sigma, A)
        x[ts+1]= x[ts] + (1/6)* (k1 + 2*k2 + 2*k3 + k4)
    return x

@jit(nopython=True)
def runge_kutta_AS(ini, tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, state,nabla_x): 
    """Runge kutta of 4th order.

        Parameters
        ----------
        ini : array of shape (d,N)
            boundary conditions at t=tsteps-1
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        all other parameters are only passed on to function ODE_AS()

        Returns
        -------
        x : array of shape (tsteps,d,N)
            numerical solution of the adjoint state with boundary conditions ini.

    """
    x=np.zeros((tsteps,d,N))
    x[0]=ini    
    for ts in np.arange(0,tsteps-1):
        k1 = dt* ode_AS(x[ts],ts, tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, state,nabla_x)
        k2 = dt* ode_AS(x[ts] + k1/2 , ts , tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, state,nabla_x)
        k3 = dt* ode_AS(x[ts] + k2/2 , ts , tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, state,nabla_x)
        k4 = dt* ode_AS(x[ts] + k3 , ts , tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, state,nabla_x)
        x[ts+1]= x[ts] + (1/6)* (k1 + 2*k2 + 2*k3 + k4)
    return x

@jit(nopython=True)
def runge_kutta_FHN_network_noisy(ini, control, tsteps , d , dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, noise):
    """Runge kutta of 4th order. Has additional parameter for noise.

        Parameters
        ----------
        f_func : the ODE function to be solved, can be FHN() (FitzHugh-Nagumo) or adjoint() (adjoint state ODE).
        ini : boundary conditions at t=0
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        all other parameters are only passed on to function ODE_FHN_network_noisy()
        noise : array of shape (steps,N)
            noise realization

        Returns
        -------
        x : array of shape (tsteps,d,N)
            numerical solution of the given ODE f_func with initial conditions ini and noise realization noise

    """
    x=np.zeros((tsteps,d,N))
    x[0]=ini    
    for ts in np.arange(0,tsteps-1):
        k1 = dt* ODE_FHN_network_noisy(x[ts],ts,control,tsteps , d , dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,noise[ts])
        k2 = dt* ODE_FHN_network_noisy(x[ts] + k1/2 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,noise[ts])
        k3 = dt* ODE_FHN_network_noisy(x[ts] + k2/2 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, noise[ts])
        k4 = dt* ODE_FHN_network_noisy(x[ts] + k3 , ts ,control,tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu, sigma, A, noise[ts])
        x[ts+1]= x[ts] + (1/6)* (k1 + 2*k2 + 2*k3 + k4)
    return x


@jit(nopython=True)
def R(x, alpha , beta, gamma):
    """Calculates term of the first dimension of the ODE of FitzHugh Nagumo, see function ODE_FHN_network() and ODE_FHN_network_noisy(): R= -alpha* x1**3 + beta *x1**2 - gamma* x1 

        Parameters
        -------
        x : array of shape (N,)
            first dimension of state at a single timepoint
        alpha, beta, gamma : floats
            parameter of sigle FutzHugh-Nagumo nodes

        Returns
        -------
        R : array of shape (N,)
             -alpha* x1**3 + beta *x1**2 - gamma* x1 

    """
    return -alpha* x**3 + beta *x**2 - gamma* x 


@jit(nopython=True)
def Rder(x, alpha , beta, gamma):
    """Calculates the derivative with respect x1 of the first dimension of the function R() 

        Parameters
        -------
        x : array of shape (N,)
            first dimension of state at a single timepoint
        alpha, beta, gamma : floats
            parameter of sigle FutzHugh-Nagumo nodes

        Returns
        -------
        R : array of shape (N,)
           d/dx1 ( -alpha* x1**3 + beta *x1**2 - gamma* x1 )

    """
    return -3*alpha* x**2 + 2*beta*x - gamma

@jit(nopython=True)
def ODE_FHN_network(x,ts,control, tsteps , d, dt , N, alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A):
    """ODE of network dynamics with FitzHugh Nagumo oscillators.
    dx1/dt = -alpha x1^3 + beta x1^2 - gamma x1 - x2 + I_{ext} + coupling_term +control
    dx2/dt = 1/tau (x1 + delta  - epsilon x2)
    To be used if noise strength=0!

        Parameters
        -------
        x : array of shape (d,N)
            oscillator states at a single timepoint ts-1
        ts : float
            integration timepoint
        control : array of shape (tsteps,N)
            the control signal
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        N : int
            number of nodes
        alpha , beta, gamma, delta, epsilon, tau, mu ,sigma : floats
            parameters of the FHN-oscillator
        A : array shape (N,N)
            adjacency matrix

        Returns
        -------
        out : array of shape (d,N)
             evaluation of network dynamics at timepoint ts
    """

    out=np.zeros(x.shape)
    control=control.reshape(tsteps,N)
    Coupling = np.dot(A,x[0])
    out[0]= R(x[0], alpha , beta, gamma) - x[1] + mu + control[ts] + sigma * Coupling
    out[1]= (1/tau)* (x[0] + delta  - epsilon* x[1])
    return out  
 
@jit(nopython=True)
def ODE_FHN_network_noisy(x,ts,control, tsteps , d, dt , N ,alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, noise_ts):
    """ODE of network dynamics with FitzHugh Nagumo oscillators.
    dx1/dt = -alpha x1^3 + beta x1^2 - gamma x1 - x2 + I_{ext} + coupling_term +control+noise
    dx2/dt = 1/tau (x1 + delta  - epsilon x2)
    To be used if noise strength>0!

        Parameters
        -------
        x : array of shape (d,N)
            oscillator states at a single timepoint ts-1
        ts : float
            integration timepoint
        control : array of shape (tsteps,N)
            the control signal
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        N : int
            number of nodes
        alpha , beta, gamma, delta, epsilon, tau, mu ,sigma : floats
            parameters of the FHN-oscillator
        A : array shape (N,N)
            adjacency matrix
        noise : array of shape (tsteps,N)

        Returns
        -------
        out : array of shape (d,N)
             evaluation of noisy network dynamics at timepoint ts
    """


    out=np.zeros(x.shape)
    control=control.reshape(tsteps,N)
    Coupling = np.dot(A,x[0])
    out[0]= R(x[0], alpha , beta, gamma) - x[1] + mu + control[ts] + sigma * Coupling +noise_ts
    out[1]= (1/tau)* (x[0] + delta  - epsilon* x[1])
    return out  
  
@jit(nopython=True)
def ode_AS(phi,ts, tsteps , d, dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, state,nabla_x):
    """ODE of adjoint state (AS) phi. Calculated backwards in time.
    d phi1/dt = (-3*alpha* x**2 + 2*beta*x - gamma )*phi1+ phi2/tau+nabla_x_f(ts) + dot(A,phi1)
    d xphi2/dt = -phi1 - (epsilon/tau) * phi2

        Parameters
        -------
        phi : array of shape (d,N)
            adjoint states at a single timepoint ts-1
        ts : float
            integration timepoint
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        N : int
            number of nodes
        alpha , beta, gamma, delta, epsilon, tau, mu ,sigma : floats
            parameters of the FHN-oscillator
        A : array shape (N,N)
            adjacency matrix
        state : array of shape (tsteps,N)
            first dimension x1 of network state
        nabla_x : array of shape (tsteps,N)
            multidimesional derivative of the cost functional with respect to the state as given in equations nabla_x_f_switch and nabla_x_f_switch

        Returns
        -------
        out : array of shape (d,N)
             evaluation of adjoint state dynamics at timepoint ts
    """ 
    tss=-ts-1
    out=np.zeros(phi.shape)
    Coupling = np.dot(A,phi[0])
    out[0]= Rder(state[tss], alpha , beta, gamma) *phi[0] + phi[1]/np.float(tau) + nabla_x[tss]  + sigma * Coupling 
    out[1]= -phi[0] - (np.float(epsilon)/np.float(tau))*phi[1]
    return out 

@jit(nopython=True)
def AS(tsteps , d, dt , N , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, state,nabla_x):
    """Adjoint state. Calculated backwards in time

        Parameters
        ----------
        tsteps : int
            number of timesteps
        d : int
            dimension of local dynamics (e.g. for FHN-oscillator d=2)
        dt : float
            stepsize of integration
        N : int
            number of nodes
        alpha , beta, gamma, delta, epsilon, tau, mu ,sigma : floats
            parameters of the FHN-oscillator
        A : array shape (N,N)
            adjacency matrix
        state : array of shape (tsteps,N)
            first dimension x1 of network state
        nabla_x : array of shape (tsteps,N)
            multidimesional derivative of the cost functional with respect to the state as given in equations nabla_x_f_switch and nabla_x_f_switch

        Returns
        -------
        adjoint : array shape (tsteps,N)
            first dimension of the adoint state phi1
    """   
    ini_AS=np.zeros((d,N))
    adjoint_flipped=runge_kutta_AS( ini_AS, tsteps , d , dt , N ,alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, state,nabla_x)[:,0,:]
    adjoint=np.zeros((tsteps,N))
    for i in np.arange(tsteps):
        adjoint[i]=adjoint_flipped[-i-1]    
    return adjoint


