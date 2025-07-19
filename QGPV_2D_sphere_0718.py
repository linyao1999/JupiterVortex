import numpy as np
import dedalus.public as d3
import logging
import scipy
import dedalus.extras.flow_tools as flow_tools
from mpi4py import MPI
import dedalus.tools.logging as mpi_logging
import matplotlib.pyplot as plt
import pyshtools as pysh

logger = logging.getLogger(__name__)

###### Numerical Parameters ####
Nφ        = 512       # Number of l modes
Nθ        = 256       # Number of m modes
dealias   = 3/2 
dtype     = np.float64
seed0     = 1

# Set restart = True to restart from a given checkpoint
restart    = False #True
# Specify checkpoint directory
cp_path    = '/projects/GEOCLIM/hengquan/xxxx/checkpoints/checkpoints_sxx.h5'

##### dimensional parameters
# L         = 1       # sphere radius
# Omega     = 1000    # rotation angular velocity
# T         = 0.5/Omega
# U         = L/T

##### non-dimensional parameters
dt_ratio  = 1             # multiply timestep by this to reduce
 
R         = 1.0             # Sphere radius
timestep  = 2 * dt_ratio    # initial timestep
stop_sim_time = 400000      # stop time
max_timestep = 2 * dt_ratio

k_nu = 2e-11            # nabla4 eddy viscosity, inverse modified Reynolds number, non-dim
k_fric = 1e-10             # linear damping, inverse frictional Reynolds number, non-dim
k_d2 = 0.01                   # Rossby deformation radius, (L_d/R)^2, base value 100 approximately incompressible
k_rad = 3               # radiative damping, base value 4
k_inject = 1e-10            # energy injection rate, non-dim

# base values to match Scott 2008 and Saito 2014
# k_nu = 1e-11            # nabla4 eddy viscosity, inverse modified Reynolds number, non-dim
# k_fric = 1e-10             # linear damping, inverse frictional Reynolds number, non-dim
# k_d2 = 0.01                   # Rossby deformation radius, (L_d/R)^2, base value 100 approximately incompressible
# k_rad = 4               # radiative damping, base value 4
# k_inject = 5e-7            # energy injection rate, non-dim

#Specify forcing range
ls = [64]    #specify spherical meridional wavenumber(s) to be forced

#Timestepper settings
timestepper = d3.RK443       

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist   = d3.Distributor(coords, dtype=dtype)
basis  = d3.SphereBasis(coords, (Nφ, Nθ), radius=R, dealias=dealias, dtype=dtype)

# Fields
f      = dist.Field(              name='f',   bases=basis)
ζ      = dist.Field(              name='ζ', bases=basis) # vorticity
#q      = dist.Field(              name='q', bases=basis)
ψ      = dist.Field(              name='ψ',   bases=basis) # stream function
ψ_f    = dist.Field(              name='ψ_f',   bases=basis) # the "streamfunction" for stochastic forcing
τ      = dist.Field(              name='τ')
g      = dist.Field(              name='g',   bases=basis)   # cos theta

# Coordinates
φ, θ = dist.local_grids(basis)
lat = np.pi / 2 - θ + 0*φ

# sinlat2 = dist.Field(              name='sinlat',   bases=basis)
# sinlat2['g'] = np.sin(lat) * np.sin(lat)

# f
f['g'] = np.sin(lat)

# initial condition: equatorial subrotation
n_c = 36
theta_c = np.pi/n_c
u_eq_init = 0.0003
#u['g'][0] = np.where(np.abs(lat)<theta_c,0.5*u_eq_init*(np.cos(n_c*lat)+1),0)
ζ['g'] = np.where(np.abs(lat)<theta_c,0.5*n_c*u_eq_init*np.sin(n_c*lat),0)
#ψ['g'] = np.where(np.abs(lat)<theta_c, -0.5 * 1/n_c * u_eq_init * np.sin(n_c*lat) - 0.5*u_eq_init*lat,0)

# Problem (nondimensional)
problem = d3.IVP([ζ, ψ, τ], namespace=locals())
problem.add_equation("ζ - d3.lap(ψ) + τ = 0")  # diagnostic relation
# problem.add_equation("q - ζ = -sinlat2 * ψ/k_d2")
problem.add_equation("dt(ζ - MulCosine(MulCosine(ψ))/k_d2) + k_nu*d3.lap(d3.lap(ζ)) + k_fric*ζ = - d3.skew(d3.grad(ψ))@d3.grad(ζ + f - MulCosine(MulCosine(ψ))/k_d2) + d3.lap(ψ_f) + k_rad*MulCosine(MulCosine(ψ))")
problem.add_equation("ave(ψ) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#Set parameters, if restart: load checkpoint
if restart == False:
  file_handler_mode = 'overwrite'
  initial_timestep = max_timestep
elif restart == True:
  write, initial_timestep = solver.load_state(cp_path)
  file_handler_mode = 'append'

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=500*max_timestep, max_writes=10000,mode=file_handler_mode)
snapshots.add_task(ζ, name='vorticity')
snapshots.add_task(d3.skew(d3.grad(ψ)),   name ='u')
snapshots.add_task(d3.skew(d3.grad(ψ)),   name='u_c', layout='c') #OPTIONAL: uncomment if you want to output coefficients 
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=10000*max_timestep, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# Scalar Data
analysis1 = solver.evaluator.add_file_handler("scalar_data", sim_dt=max_timestep*10,mode=file_handler_mode)
analysis1.add_task(d3.Average(0.5*(d3.skew(d3.grad(ψ))@d3.skew(d3.grad(ψ))),coords), name="Ekin")
analysis1.add_task(d3.Average(ζ**2, coords), name='Enstrophy')

# Flow properties, dimensional
flow_prop_cad = 100
flow = d3.GlobalFlowProperty(solver, cadence = flow_prop_cad)
flow.add_property(d3.Average(-0.5*ψ*ζ, coords), name = 'avg_Energy')         # x U**2 to redimensionalize
flow.add_property(d3.Average(ζ**2, coords), name = 'avg_Enstrophy')         # x (UL)**2 to redimensionalize
flow.add_property(d3.Average(-k_nu*ψ*d3.lap(d3.lap(ζ)),coords), name='diss_hyper') # x U**2/T to redimensionalize
flow.add_property(d3.Average(-k_fric*ψ*ζ,coords), name='diss_fric')                 # x U**2/T to redimensionalize
flow.add_property(d3.Average(k_rad*ψ*ψ,coords), name='diss_rad')                    # x U**2/T to redimensionalize

#GlobalArrayReducer
reducer   = flow_tools.GlobalArrayReducer(comm=MPI.COMM_WORLD)

# CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=10, safety=0.8, threshold=0.8, max_change=100000, min_change=0.00001, max_dt=max_timestep)
CFL.add_velocity(d3.skew(d3.grad(ψ)))

# Initialise random seed to seed0
np.random.seed(seed0)

#set MPI rank and size
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

it0 = solver.iteration

#############
# MAIN LOOP #
#############
g.preset_scales(dealias)
ψ_f.preset_scales(dealias)

try:
    logger.info('Starting main loop')
    while solver.proceed:

        #INITIALISE ALL ARRAYS TO CORRECT SIZE IN FIRST ITERATION
        if solver.iteration == it0:
          φ,θ = basis.local_grids(dist=dist,scales=(dealias,dealias));
          lat = np.pi / 2 - θ + 0*φ
          φ_mat   = φ   + 0*θ
          θ_mat = θ + 0*φ
          g['g'] = np.cos(lat)
        
        elif solver.iteration > it0: 
         
          ## FORCING
          ψ_f['g'] = np.zeros_like(ψ_f['g'])
          power = np.zeros(Nθ); power[ls[0]] = 1;
          clm = pysh.SHCoeffs.from_random(power, seed=seed0+solver.iteration, kind='real', lmax=int(dealias*Nθ)-1)
          grid_glq = clm.expand(grid='GLQ')
          global_random_field = np.transpose(grid_glq.data)
          
          #reduce global field to individual CPUs
          ψ_f.load_from_global_grid_data(global_random_field)

          #compute variance and build vorticity forcing
          f_temp     = d3.skew(d3.grad(ψ_f)).evaluate()
          f_temp_var = reducer.global_mean(g['g']*(f_temp['g'][0]**2 + f_temp['g'][1]**2))/reducer.global_mean(g['g'])
          ψ_f['g']  = ψ_f['g'] / np.sqrt(f_temp_var) * np.sqrt(2*k_inject/timestep)
             
        timestep = CFL.compute_timestep()
        
        solver.step(timestep)
        
        if (solver.iteration) % flow_prop_cad == 0:
            avg_Energy     = flow.max('avg_Energy')
            avg_Enstrophy  = flow.max('avg_Enstrophy')
            diss_hyper     = flow.max('diss_hyper')
            diss_fric      = flow.max('diss_fric')
            diss_rad       = flow.max('diss_rad')
            logger.info('Iteration=%i, Time=%e, dt=%e, Energy=%.2e, Enstrophy=%.2e, Energy/time=%.2e, diss_hyper =%.2e, diss_fric=%.2e, diss_rad=%.2e'  %(solver.iteration, solver.sim_time, timestep, avg_Energy, avg_Enstrophy, avg_Energy/solver.sim_time, diss_hyper, diss_fric, diss_rad))
 
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()