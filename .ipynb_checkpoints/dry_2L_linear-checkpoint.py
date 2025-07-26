import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
import os
import h5py
logger = logging.getLogger(__name__)
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)

# prob_class = 'EVP'
prob_class = 'IVP' 
restart = False
init_pattern = 'random'  # 'evp_max_growth'; 'random'

# Parameters
# Physical paramters
a = 6.99e7 # radius (m)
P = 9.925 * 3600 # rotation period (s)
R_dim = a / 2
gamma_dim = 2 * (2 * np.pi / P) / a**2
# Dimensionless parameters
L = 1e7 # m
U = 100 # m/s
T = L / U # s
F = 51.8 # = L**2 / Ld**2
gamma = gamma_dim * L**2 * T
R = R_dim / L
print(f"gamma = {gamma}")
print(f"R     = {R}")

output_dir = f'/net/fs06/d0/linyao/GFD_Polar_vortex/ddloutput/{prob_class}/'
os.makedirs(output_dir, exist_ok=True)

# numerical parameters
m_max = 71
Nphi = 4 * (m_max + 1)
# Nphi = 2 * (m_max + 1)
# Nphi = 128
Nr = 256

if prob_class == 'EVP':
    dtype = np.complex128
    kn_zeros = 5 # the number of zeros of the Bessel function to store
    nu = 0
elif prob_class == 'IVP':
    dtype = np.float64
    timestep = 1e-3
    timestepper = d3.RK443
    stop_sim_time = 10
    dealias = 3/2
    initv_scale = 1
    nu = 1e-6

    if init_pattern == 'evp_max_growth':
        m_max_growth = 19
        init_pattern_file = 'EVP.h5'
    
    # snapshots_name = f'snapshots_F{int(np.floor(F))}_U_{U}_linear_init{init_pattern}'
    # snapshots_file = output_dir + snapshots_name
    snapshots_file = 'snapshots'
    checkpoint_path = 'checkpoints'


# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=R, dtype=dtype)
phi, r = dist.local_grids(disk)

# Fields
s = dist.Field(name='s')
psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
tau_psi1 = dist.Field(name='tau_psi1', bases=disk.edge)
tau_psi2 = dist.Field(name='tau_psi2', bases=disk.edge)

# Substitutions
if prob_class == 'EVP':
    dt = lambda A: s*A
    
psi2u = lambda A: d3.Skew(d3.Gradient(A))
lift_basis = disk
lift = lambda A: d3.Lift(A, lift_basis, -1)
q1 = d3.lap(psi1) - F * (psi1 - psi2)
q2 = d3.lap(psi2) + F * (psi1 - psi2)
u1 = psi2u(psi1)
u2 = psi2u(psi2)

# Background
r_sq = dist.Field(bases=disk.radial_basis)
r_sq['g'] = r**2  # radial coordinate
Psi1 = r_sq / 2
Psi2 = - r_sq / 2
Q1 = d3.lap(Psi1) - F * (Psi1 - Psi2) - 0.5 * gamma * r_sq
Q2 = d3.lap(Psi2) + F * (Psi1 - Psi2) - 0.5 * gamma * r_sq
U1 = psi2u(Psi1)
U2 = psi2u(Psi2)

# Problem
if prob_class == 'EVP':
    problem = d3.EVP([psi1, psi2, tau_psi1, tau_psi2], eigenvalue=s, namespace=locals())
elif prob_class == 'IVP':
    problem = d3.IVP([psi1, psi2, tau_psi1, tau_psi2], namespace=locals())
    
problem.add_equation("dt(q1) + u1@grad(Q1) + U1@grad(q1) + nu * q1 + lift(tau_psi1) = 0")
problem.add_equation("dt(q2) + u2@grad(Q2) + U2@grad(q2) + nu * q2 + lift(tau_psi2) = 0")
problem.add_equation("psi1(r=R) = 0")
problem.add_equation("psi2(r=R) = 0")

# Solver   ==== to be continued ===

if prob_class == 'EVP':
    # solver = problem.build_solver(ncc_cutoff=1e-6, entry_cutoff=1e-6)
    
    m_range = np.arange(1, m_max+1)
    evals_list = []
    psi1_list = []
    psi2_list = []
    def custom_key(z, tol=1e-6):
        if np.abs(z.real) < tol:
            return (0, np.abs(z.imag))
        else:
            return (-z.real, 0)
    for m in m_range:
        # Solve
        solver = problem.build_solver()
        sp = solver.subproblems_by_group[(m, None)]
        solver.solve_dense(sp)
        evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
        # evals = evals[np.argsort(-evals.real)]
        evals = sorted(evals, key=lambda z: custom_key(z))
        evals_list.append(np.copy(evals[0:kn_zeros*2]))
        print(f"m={m}, λ_max={evals[0]}")

        psi1_eigen = []
        psi2_eigen = []
        for nk in np.arange(kn_zeros*2):
            solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[nk])), sp.subsystems[0])

            psi1_eigen.append(np.copy(psi1['g'].real))
            psi2_eigen.append(np.copy(psi2['g'].real))
            
        psi1_list.append(np.copy(psi1_eigen))
        psi2_list.append(np.copy(psi2_eigen))
        
    hfile = h5py.File(f'EVP.h5', 'w')
    tasks = hfile.create_group('tasks')
    tasks.create_dataset('evals', data=evals_list)
    tasks.create_dataset('psi1', data=psi1_list)
    tasks.create_dataset('psi2', data=psi2_list)
    tasks.create_dataset('phi', data=phi)
    tasks.create_dataset('r', data=r)

elif prob_class == 'IVP':
    solver = problem.build_solver(timestepper, ncc_cutoff=1e-6, entry_cutoff=1e-6)
    solver.stop_sim_time = stop_sim_time

    # Initial conditions
    if not restart:
        file_handler_mode = 'overwrite'
        if init_pattern == 'random':
            psi1.fill_random('g', seed=42, distribution='standard_normal')
            psi2.fill_random('g', seed=42, distribution='standard_normal') 
            psi1.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
            psi2.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
        elif init_pattern == 'evp_max_growth':
            with h5py.File(init_pattern_file, 'r') as f:
                psi1.load_from_global_grid_data(f["tasks/psi1"][:][m_max_growth-1,0,:,:].real)
                psi2.load_from_global_grid_data(f["tasks/psi2"][:][m_max_growth-1,0,:,:].real)
            
        psi1['g'] *= initv_scale
        psi2['g'] *= initv_scale
    else:
        write, initial_timestep = solver.load_state(checkpoint_path)
        file_handler_mode = 'append'

    # Analysis
    snapshots = solver.evaluator.add_file_handler(snapshots_file, sim_dt=0.1, max_writes=10000, mode=file_handler_mode)
    snapshots.add_task(psi1, scales=(1, 1), name='psi1')
    snapshots.add_task(psi2, scales=(1, 1), name='psi2')
    snapshots.add_task(q1, scales=(1, 1), name='q1')
    snapshots.add_task(q2, scales=(1, 1), name='q2')

    checkpoints = solver.evaluator.add_file_handler(checkpoint_path, sim_dt=5, max_writes=1, mode=file_handler_mode)
    checkpoints.add_tasks(solver.state)
    
    flow = d3.GlobalFlowProperty(solver, cadence=100)
    flow.add_property((u1)**2, name='ke')  # could be a vector; to be check 

    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            solver.step(timestep)
            # # apply low_pass_filter at each time step for psi1 and psi2; BAD 
            # psi1.low_pass_filter(scales=0.8) # Keep only lower fourth of the modes
            # psi2.low_pass_filter(scales=0.8) # Keep only lower fourth of the modes
            
            if (solver.iteration-1) % 1000 == 0:
                max_psi1 = np.sqrt(flow.max('ke'))
                logger.info("Iteration=%i, Time=%e, dt=%e, max(psi1)=%e" %(solver.iteration, solver.sim_time, timestep, max_psi1))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()


    