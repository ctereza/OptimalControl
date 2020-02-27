# OptimalControl
The project contains the codes for calculating otpimal controls for a network of FitzHugh-Nagumo oscillators coupled with structural connectivity data. Control tasks can be either to synchronize the network dynamics or to lead the system to a target state.

## Dependencies
* python 3
* jupyter notebook

## Content
This project includes:
* A pdf file  'draft_paper.pdf' with the needed mathematical backround, applied methods, and figures of example results.
* A pdf file 'minimization_problem.pdf' with a more precise mathematical derivation of the equations given in 'draft_paper.pdf'.
* A folder 'src_and_examples' including the optimization code with all needed functions and an example:
    * FR_optimize.py: includes the functions for the gradient descent method (optimization loop, find descent direction, choose stepsize). The gradient descent algorithm is described in section IIIB of the pdf-file 'draft_paper.pdf'.
    * functions.py: includes the functions for the initialization of the code and the calculation of the cost functional and its gradient. The mathematical definitions of the cost functionals and their gradients are given in section IIIA of the pdf-file 'draft_paper.pdf'.
    * example.ipynb
* A folder for reproducing all figures included in the draft_paper.pdf containing:
   * notebook files for each figure (the figures are enumerated as in the draft_paper.pdf)
   * a folder called data included all the data needed for plotting the figures
   * a folder called figures, where the plotted figures are in

## Code structure
What the code does:
* Initialize the the network: FHN network with structural connectivity, set all parameters and initial conditions
* Define the control task: set the cost functional and an initial control
* Run Gradient descent method to find a minimum of the cost functional (see section IIIB of the pdf-file 'draft_paper.pdf')
* Plot or save results

## How the code can be used
In the file example.ipynb parameters can be changed to compute the desired control and save the results. The parameters are:
* goal : control task can be either 'switch' to switch between bistable states or 'sync' to synchronize the network dynamics.
* bifurcation : can be 'low' or 'high'. Parameters sigma and mu are chosen to be close to the low or high bifurcation.
* switchfromto : needed if goal='switch'. Can be 'lowtohigh' for switching from the low to the high state or 'hightolow' for the opposite. 
* noise_stregth : strength of the white Gaussian additive noise. Should be 0, if the task is to switch between states.
* T : simulation and control time.
* I_p : parameter of the cost functional weighing precision.
* I_e : parameter of the cost functional weighing the energy cost of the control.
* I_s : parameter of the cost functional weighing the sparsity of the control in space.

All other parameters are automatically chosen correctly for each case.

## Problems
* For high values of sparsity the gradient does sometimes not match the functional and the code finishes with warnflag 3. I am sorry, I could not find why this happens.
* The computational time (expecially for synchronizing the network dynamics) is very long. The code should be made more efficient.

## Minimal example
This is an minimal example for the calculation of an optimal control input to a coupled network of FitzHugh-Nagumo oscillators.
In this example the control task is to switch between predefined bistable states.
For more precise documentationa and for altering parameters please use the jupyter notebook file example.ibynb!

The calculation of this example might take a few minutes.

      import sys
      import os
      absFilePath = os.path.abspath('example_state_switching.py')
      fileDir = os.path.dirname(absFilePath)

      import functions as functions
      from FRoptimize import FR_algorithm

      import numpy as np
      import pickle
      import matplotlib.pyplot as plt
      from pylab import figure, cm

      goal='switch'
      bifurcation='low'
      switchfromto='lowtohigh' 
      noise_strength= 0.0 
      realizations=1
      noise=0
      parameters = functions.set_parameters(goal,bifurcation,fileDir) 

      #Set dimensions
      T= 400 #simulation and control time 
      dt=0.1 #time stepsize
      tsteps=int(T/dt) #number of timesteps
      d=2 #dimension of each FitzHugh-Nagumo node

      #set parameters of the cost functional
      I_p_in= 0.0005
      I_p=np.zeros((tsteps,parameters['N']))
      I_p[int(tsteps-25/dt):]=I_p_in*np.ones((int(25/dt),parameters['N']))
      I_e= 1.0 
      I_s= 0.0


      #choose initial condition for control
      control=-np.ones((tsteps,parameters['N']))*0.002
      control=control.reshape(tsteps*parameters['N'])

      node_ic,target_output=functions.read_node_inis_from_file(bifurcation,switchfromto,fileDir,tsteps,d,dt,parameters)

      #make dictionary with all parameters
      args = {
          'tsteps':tsteps,
          'dt':dt,
          'd':d,
          'I_p':I_p,
          'I_e':I_e,
          'I_s':I_s,
          'target_output':target_output,
          'node_ic':node_ic,
          'realizations':realizations,
          'noise':noise
          }
      args.update(parameters)

In this case we want to find an optimal control that induces a switching from an initial low state to a high target state.
The uncontrolled state and the target state are plotted:

      def create_plot(data,ylabel,title):
          fs=30 #fontsize
          fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

          # set min and max for scope of y-axis
          minn=np.min(data)
          maxx=np.max(data)
          add=(maxx-minn)/25

          im1=ax.plot(np.arange(0,int(T),dt),data)
          ax.set_xlabel('t',size=fs)
          ax.set_ylabel(ylabel,size=fs)
          ax.set_ylim(minn-add,maxx+add)
          ax.set_title(title,size=fs, pad=20)
          ax.tick_params(labelsize=fs)
          ax.grid(True)
          ax.margins(0) # remove default margins (matplotlib verision 2+)

          return fig.tight_layout()

      #calculate the uncontrolled state
      nocontrol=np.zeros((tsteps,parameters['N']))
      state_uncontrolled=functions.plot_runge_kutta(functions.ODE_FHN_network,nocontrol, **args)

      create_plot(state_uncontrolled[:,0,:],'axtivity $x_{k1}$','uncontrolled state')
      plt.show()
      create_plot(target_output,'axtivity $x_{k1}$','target state')
      plt.show()

To find the optimal control we define the cost functional and its gradient and start the optimization loop.

      #define the functional and its gradient
      functional=functions.f_switch
      gradient=functions.fprime_switch

      #initialize the control loop
      iteration=0 
      #warnflag=1-> running, warnflag=0->finished, warnflag=3->error
      warnflag=1
      #start the optmization
      result = FR_algorithm(functional, control, gradient,**args)
      control=result[0]
      warnflag=result[2]
      iteration+=result[4]
      print('Code finished after ',iteration,' iterations with warnflag',result[2])
    
The results can now be plottet:

      #calculate the controlled state
      control=control.reshape(tsteps,parameters['N'])
      state_controlled=functions.plot_runge_kutta(functions.ODE_FHN_network,control, **args)

      create_plot(state_controlled[:,0,:],'axtivity $x_{k1}$','controlled state')
      plt.show()
      create_plot(control,'control $u_{k1}$','otpimal control input')
      plt.show()

