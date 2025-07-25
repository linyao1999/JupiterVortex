import numpy as np 
import dedalus.public as d3
import h5py 
import matplotlib.pyplot as plt
import os 
import logging
logger = logging.getLogger(__name__)
# loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# for logger in loggers:
#     logger.setLevel(logging.WARNING)
    
# mpirun -n 6 python3 

# --------- CHOOSE THE PROBLEM -----------
# prob_class = 'EVP'
prob_class = 'IVP'
restart = False 

# ----------- Physical parameters -----------------
# ======= fixed ==========
L = 1e7 # horizontal length scale; appro radius of polar vortex
T = 9.925*3600  # period of Jupiter rotation
a = 6.99e7 
# ======= derived =========
omega = 2 * np.pi / T  # rotation speed
gamma = 2 * omega / a / a # dimensional gamma
a_norm = a / L / 2.0   # dimensionless disk radius
# ======= defined/changed ==========
F1 = 51.8  # L**2/Ld1**2; can vary between 0.1 and 100
delta = 1.0 # H2/H1
F2 = delta**2 * F1 # L**2/Ld1**2;
U = 100
Gamma = gamma * (L**3) / U

# ------------- Numerical parameters --------------------
Nphi = 128
Nr = 256
output_dir = f'/net/fs06/d0/linyao/GFD_Polar_vortex/ddloutput/{prob_class}/'
os.makedirs(output_dir, exist_ok=True)

if prob_class == 'EVP':
    dtype = np.complex128
    max_kphi = 30
elif prob_class == 'IVP':
    dtype = np.float64
    timestep = 1e-3
    timestepper = d3.RK443
    stop_sim_time = 50
    initv = 1e-4
    snapshots_name = f'snapshots_F{int(np.floor(F1))}_U_{U}_linear_noNu'
    snapshots_file = output_dir + snapshots_name

    checkpoint_path = snapshots_file + '/checkpoints_s1.h5'
    initial_timestep = 1e-3
    file_handler_mode = 'append'
    

# --------------- Bases ------------------------
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=a_norm, dtype=dtype)
phi, r = dist.local_grids(disk)

# --------------- Fields -----------------------
s = dist.Field(name='s')
psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
tau1 = dist.Field(name='tau1', bases=disk.edge)
tau2 = dist.Field(name='tau2', bases=disk.edge)

# ----------------- Substitutions ----------------------
if prob_class == 'EVP':
    dt = lambda A: s*A

lift_basis = disk.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)
q1 = d3.lap(psi1) - F1 * (psi1 - psi2)
q2 = d3.lap(psi2) + F2 * (psi1 - psi2)
u1 = d3.Skew(d3.Gradient(psi1))
u2 = d3.Skew(d3.Gradient(psi2))

# ------------------ Background ------------------------
r2 = dist.Field(bases=disk.radial_basis)
r2['g'] = 0.5 * (r**2)
Psi1 = r2 
Psi2 = - r2
U1 = d3.Skew(d3.Gradient(Psi1))
U2 = d3.Skew(d3.Gradient(Psi2))
Q1 = d3.lap(Psi1) - F1 * (Psi1 - Psi2) - Gamma * r2
Q2 = d3.lap(Psi2) + F2 * (Psi1 - Psi2) - Gamma * r2

# ------------------ Problem ------------------------------
if prob_class == 'EVP':
    problem = d3.EVP([psi1, psi2, tau1, tau2], eigenvalue=s, namespace=locals())
elif prob_class == 'IVP':
    problem = d3.IVP([psi1, psi2, tau1, tau2], namespace=locals())
    
problem.add_equation(" dt(q1) + u1 @ grad(Q1) + U1 @ grad(q1) + lift(tau1) = 0 ")
problem.add_equation(" dt(q2) + u2 @ grad(Q2) + U2 @ grad(q2) + lift(tau2) = 0 ")
problem.add_equation("psi1(r=a_norm) = 0")
problem.add_equation("psi2(r=a_norm) = 0")


if prob_class == 'EVP':
    solver = problem.build_solver(ncc_cutoff=1e-6, entry_cutoff=1e-6)
    # eval_list = []
    for kphi in range(1, max_kphi+1):
        sp = solver.subproblems_by_group[(kphi, None)]
        solver.solve_dense(sp)
        evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
        evals = evals[np.argsort(-evals.real)]
        print(f"m={kphi}; λ = {evals[0]}")
        # eval_list.append(evals[0])
        solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), sp.subsystems[0]) 

        # hfile = h5py.File(f'{output_dir}{prob_class}_dry_2L_linearEQ_noNu_F{F1}_U{U}_m{kphi}.h5', 'w')
        # tasks = hfile.create_group('tasks')
        # tasks.create_dataset('psi1', data=psi1['g'].real)
        # tasks.create_dataset('psi2', data=psi2['g'].real)
        # tasks.create_dataset('phi', data=phi)
        # tasks.create_dataset('r', data=r)
        
        scales = (32,4)
        psi1.change_scales(scales)
        psi2.change_scales(scales)
        phi, r = dist.local_grids(disk, scales=scales)
        x, y = coords.cartesian(phi, r)
    
        print(psi1['g'].real.shape)
    
        cmap = 'RdBu_r'
        fig, ax = plt.subplots(1, 2, figsize=(6, 6))
        ax[0].pcolormesh(x, y, psi1['g'].real, cmap=cmap)
        ax[0].set_title(r"$\psi_1$")
        ax[1].pcolormesh(x, y, psi2['g'].real, cmap=cmap)
        ax[1].set_title(r"$\psi_2$")
    
        for axi in ax.flatten():
            axi.set_aspect('equal')
            axi.set_axis_off()
        fig.tight_layout()
        fig.savefig(f'./plots/EVP_dry_phi_F{F1}_U{U}_m{kphi}.png', dpi=200)

    # phi, r = dist.local_grids(disk)
    # hfile = h5py.File(f'{output_dir}{prob_class}_dry_2L_linearEQ_noNu_F{F1}_U{U}_background.h5', 'w')
    # tasks = hfile.create_group('tasks')
    # tasks.create_dataset('Psi1', data=Psi1['g'].real)
    # tasks.create_dataset('Psi2', data=Psi2['g'].real)
    # tasks.create_dataset('U1', data=U1['g'].real)
    # tasks.create_dataset('U2', data=U2['g'].real)
    # tasks.create_dataset('Q1', data=Q1['g'].real)
    # tasks.create_dataset('Q2', data=Q2['g'].real)
    # tasks.create_dataset('phi', data=phi)
    # tasks.create_dataset('r', data=r)

elif prob_class == 'IVP':
    solver = problem.build_solver(timestepper, ncc_cutoff=1e-10, entry_cutoff=1e-10)
    solver.print_subproblem_ranks(solver.subproblems, timestep)
    solver.stop_sim_time = stop_sim_time

    # Initial conditions
    if not restart:
        file_handler_mode = 'overwrite'
        psi1.fill_random('g', seed=42, distribution='standard_normal') # Random noise
        psi1['g'] *= initv
        # psi1.low_pass_filter(scales=0.9) # Keep only lower fourth of the modes
        
        psi2.fill_random('g', seed=42, distribution='standard_normal') # Random noise
        psi2['g'] *= initv
        # psi2.low_pass_filter(scales=0.9) # Keep only lower fourth of the modes
    else:
        write, initial_timestep = solver.load_state(checkpoint_path)
        file_handler_mode = 'append'

        
    # Analysis
    snapshots = solver.evaluator.add_file_handler(snapshots_file, sim_dt=0.1, max_writes=10000, mode=file_handler_mode)
    snapshots.add_task(psi1, scales=(1, 1), name='psi1')
    snapshots.add_task(psi2, scales=(1, 1), name='psi2')

    checkpoints = solver.evaluator.add_file_handler(checkpoint_path, sim_dt=5, max_writes=1, mode=file_handler_mode)
    checkpoints.add_tasks(solver.state)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=1)
    flow.add_property((u1)**2, name='ke')  # could be a vector; to be check 
    
    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            # timestep=1e-3
            #timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 1000 == 0:
                # max_u = np.sqrt(flow.max('u2'))
                max_psi1 = np.sqrt(flow.max('ke'))
                logger.info("Iteration=%i, Time=%e, dt=%e, max(psi1)=%e" %(solver.iteration, solver.sim_time, timestep, max_psi1))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()


