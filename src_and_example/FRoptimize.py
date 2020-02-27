#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("/home/tereza/control01/tereza")

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
from numpy import Inf

def vecnorm(x, ord=2):
    '''Returns the norm of x'''
    if ord == Inf:
        return np.amax(np.abs(x))
    elif ord == -Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x)**ord, axis=0)**(1.0 / ord)
    
    
        

def bisection(f, uk , fprime, fk , gk , pk  ,tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,target_output,node_ic , realizations,noise  ,max_iter=10000 , max_alpha = 5000 , c1=1e-10 ):
    """
    Find stepsize with simple bisection.

    Parameters
    ----------
        f : function
            cost functional to be minimized
        uk : array shape (tsteps*N,)
            control
        fprime : function
            function returning the gradient of the cost functional
        fk : float
            vaue of the cost functional
        gk : array shape (tsteps*N,)
            gradient of the cost functional fk
        pk : array shape (tsteps*N,)
            descent direction (from polak ribiere method)
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

    Returns
    -------
    ukp : array shape (tsteps*N,)
        new control
    fkp : float
        new cost functional
    gkp : array shape (tsteps*N,)
        new gradient of cost functional
    gval_alpha[0] : float
        stepsize
    """

    #1. initialize
    gval = [None]
    gval_alpha = [None]
    minimal_alphastep=1e-17
    
    #2. test to see whether direction pk from Polak Ribiere method is actually a descet direction. If not use -gk, the negative gradient of the cost functional, instead.
    fkp_test = f(uk + 1e-13 * pk, tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,target_output,node_ic,realizations,noise)
    if fkp_test>fk or np.isnan(fkp_test):
     #   print('Replaced PR descent direction with negative gradient at x0[10]',uk[10])
        pk = -gk

    #3. function calculating the cost functional for the control uk + alpha2*pk.
    def phi(alpha2):
        return f(uk + alpha2*pk, tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, target_output,node_ic,realizations,noise)
    
    def loop_for_stepsize(pk,alpha_ini):
    #4. initialize
        warning=3
        alpha1=alpha_ini #start with a large stepsize
        iteration=0
    
        #5. loop for finding the stepsize
        while warning==3:
            alpha1=alpha1/2
           # print(alpha1)
            if iteration > max_iter or alpha1 < minimal_alphastep:
                print('Bisection: minimal alphastep or max iteration reached:',alpha1)
                ukp=uk.copy()
                break
    
            if np.max(np.abs(uk + alpha1 * pk))<50: #set this condition, since for larger control values the network dynamics might diverge
                fkp = phi(alpha1)
               # print(alpha1,fk,fkp)
                if (fkp <= fk ): #
                    warning=0    
                #    print('found!')
                    ukp = uk + alpha1*pk 
                    break

            iteration += 1
        return warning , ukp , fkp , alpha1 

    alpha_ini=100
    warning , ukp , fkp , alpha1  = loop_for_stepsize(pk,alpha_ini)

    #6.   
    if warning==3:
    #no stepsize found  . functional might be too flat
    #retry with different stepsizes
        alpha_ini=15
        warning , ukp , fkp , alpha1  = loop_for_stepsize(pk,alpha_ini)  

    if warning==3:
        gval_alpha[0]=None
        ukp=uk.copy()
        fkp=fk
        gkp=gk.copy()

    else:
    #stepsize found.
        gval_alpha[0]=alpha1
        gkp=fprime(ukp, tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, target_output,node_ic,realizations,noise   ) 
        
    return ukp , fkp , gkp , gval_alpha[0] 
  

def direction(k , gk, gkp , pk):
    """
    Calculate the descent direction with Polak-Ribiere method. Every hundredth iteration set to zero/reset.

    Parameters
    ----------
    k : int
        iteration
    gk : array shape (tsteps*N,)
        old gradient of the cost functional (from previous iteration step)
    gkp : array shape (tsteps*N,)
        new gradient of the cost functional
    pk : array shape (tsteps*N,)
        old descent direction (from previous iteration step)

    Returns
    -------
    pkp:array shape (tsteps*N,)
        new descent direction
    """
    if k%100==0:
        betak=0
    else:
        gkdiff=gkp-gk
        #Polak-Ribiere method:
        betak = max(0, np.dot(gkp, (gkdiff)) / np.dot(gk, gk) )
        #print(betak)
        #Hestenes-Stiefel method:
      #  betak= max(0, np.dot(gkp, gkdiff) / np.dot(pk, gkdiff) )
    pkp = -gkp + betak*pk
    return pkp

def FR_algorithm(f,x0,fprime,max_k=250,gtol=1e-4,**args):
    """
    Fletcher and reeves algorithm, conjugate gradient method to calvulate optimal solution of control.

    Parameters
    ----------
    f : function
        cost functional to be minimized
    x0 : array shape (tsteps*N,)
        initial values of control
    fprime : function
        function returning the gradient of the cost functional
    max_k : int
        maximal iteration number, set to 500 if no value is given
    gtol : float
        critnode_ical value, when the gradient is smaller than gtol*(1+functional) th control is optimal
    args :  dnode_ictionary
        all arguments needed for the calculation of f and fprime

    Returns
    -------
    uk : array shape (tsteps*N)
        resulting control
    fk : float
        resulting cost functional
    warnflag : int can be 0,1 or 3
        status warflag=0--> solved, uk is the optimal control.
               warnflag=1-> further iterations are needed.
               warflag=3--> error. the derivative does not match the cost functional
    k : int
        number of iterations
    gnorm : float
        Inf-norm of gradient at last iteration step
    pk : array shape (tsteps*N,)
        descent direction at last iteration step
    gk : array shape (tsteps*N,)
        gradient at last iteration step
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
    I_p = args['I_p'] #weight of the precision term of the cost functional, float
    I_e = args['I_e'] #weight of the energy term of the cost functional, float
    I_s = args['I_s'] #weight of the sparsity term of the cost functional, float
    target_output = args['target_output'] #desired/target state, array shape(tsteps,N)
    node_ic= args['node_ic'] #initial conditions of the network dynamnode_ics, array shape(d,N),
    realizations=args['realizations']
    noise=args['noise']


    #2. Iitialization
    uk=x0
    fk=f(uk,tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,target_output,node_ic,realizations,noise)
    gk=fprime(uk, tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,target_output,node_ic ,realizations,noise)
    pk=-gk.copy()
    warnflag=0
    k=1
    gnorm=vecnorm(gk,Inf)


    not_ready=True

    #3f. terminate if the solution is already optimal.
    if gnorm<=gtol*(1+fk):
        not_ready=False

    #3. Loop to find optimal solution. Wranflag=0: optimal solution is found, break. Warnflag=1: solution is not optimal yet. Warnflag=3: Negative derivative was o descent direction, Error, break. 
    while not_ready:
        #if k%1==0:
        #    print('iteration',k)
        #3a. When maximal iteration step number max_k is reached break, in order to save values and restart.
        if k>=max_k:
            warnflag=1
            print('Maximal iteration of FR algorithm reached.')
            break


        #3b. use bisection to calculate the stepsize alphak and the resulting new control ukp, the new cost functional fkp, its gradient gkp.
        ukp , fkp , gkp , alphak  = bisection(f , uk , fprime , fk , gk , pk , tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, target_output,node_ic,realizations,noise)


        #3c. if alphak=None no stepsize was found for which f(uk)>f(uk+alphak*gk) holds. Something went wrong, the negative derivative was no descent direction. Error.
        if alphak==None:
            warnflag=3
            print('Problem with the linesearch.')
            break

        #3d. calculate the new descent direction.
        pkp = direction(k , gk, gkp , pk)

        #needed for 3f.
        gnorm=vecnorm(gkp,Inf)
        uknorm=vecnorm(uk-ukp,Inf)
        #print(gnorm,fkp)

        #3e. redifine
        uk=ukp.copy()
        fk=fkp#.copy()
        gk=gkp.copy()
        pk=pkp.copy()


        k+=1

        #3f. terminate the algorithm if the gradient becomes zero or the control does not change any more (because of box constrains on control).
        if gnorm<=gtol*(1+fkp) or uknorm<=1e-20:
            print("gnorm",gnorm,"uknorm",uknorm)
            break


    #4. return 
    return uk,fk,warnflag,gnorm,k
    
