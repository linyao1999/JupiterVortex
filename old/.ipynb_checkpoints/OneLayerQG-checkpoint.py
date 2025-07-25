# Set environment variables for best performance

# Minimize logging output
import logging
logging.disable(logging.DEBUG)

# Check for Dedalus
try:
    import dedalus.public as de
    print("Dedalus already installed :)")
except:
    print("Dedalus not installed yet.")
    print("See website for installation instructions:")
    print("https://dedalus-project.readthedocs.io/en/latest/pages/installation.html")
    
import numpy as np
np.seterr(over="raise")
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import h5py
import matplotlib as mpl
from matplotlib import animation
from IPython.display import HTML

pi = np.pi
d2r = pi/180

# Specifying simulation parameters (user-defined variables)

Simulation_Name = 'OneLayerQG'

DepCase = 'Deep'
InitialType = 'Restart'
epsilon = 5e-7
alpham = 5e-5
Ld = 0.2
nuhyper_factor = 15

stop_sim_time = 2000
Output_Interval = 20
Diagnose_Interval = 1000
safety = 0.2

lat1 = 15*d2r
lat2 = 80*d2r

ytoxfac = 2
Ly = 2*( np.cos(lat1)-np.cos(lat2) )
Ny = 1024

kf = (Ny/4) * (2*np.pi/Ly)

RestartFolder = '../'
RestartName = 'Data_flow_fields_' + Simulation_Name
RestartFile = RestartFolder + RestartName + '/' + RestartName + '_s1.h5'


# Set up the domain

Lx = Ly/ytoxfac
Nx = Ny/ytoxfac
dx = Lx/Nx
dy = Ly/Ny
nuhyper = epsilon**(1/3)*dx**(10/3)/nuhyper_factor

print('epsilon=%e, nuhyper=%e, alpham=%e' %(epsilon,nuhyper,alpham))

if dx != dy:
    print('Not square grid!')

if Lx > 2*pi*np.cos(lat2):
    print('x domain too wide!')

print('Lx =',Lx,'Ly =',Ly,'max Lx =',2*pi*np.cos(lat2),'dx =',dx,'dy =',dy)

dtype = np.float64
dealias = 3/2

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
ex, ey = coords.unit_vector_fields(dist)

xc = dist.local_grid(xbasis)
yc = dist.local_grid(ybasis)


# Read field data

FieldFileFolder = '/home/yxzeng/Boussinesq/256x1024/Create_Beta/data/'
betaFileName = FieldFileFolder + 'beta_' + DepCase + '.txt'
readbeta = np.loadtxt(betaFileName,dtype = np.float64, delimiter=',')
WindowFileName = FieldFileFolder + 'Window.txt'
readWindow = np.loadtxt(WindowFileName,dtype = np.float64, delimiter=',')
AlphaWindowFileName = FieldFileFolder + 'AlphaWindow.txt'
readAlphaWindow = np.loadtxt(AlphaWindowFileName,dtype = np.float64, delimiter=',')
kywindowFileName = FieldFileFolder + 'kywindow.txt'
readkywindow = np.loadtxt(kywindowFileName,dtype = np.float64, delimiter=',')

beta = dist.Field(name='beta', bases=ybasis)
AlphaWindow = dist.Field(name='AlphaWindow', bases=ybasis)

slices = dist.grid_layout.slices(beta.domain, (1,1))
slices2 = dist.coeff_layout.slices(beta.domain, (1,1))

beta['g'] = readbeta[slices[1]]
Window = readWindow[slices[1]]
AlphaWindow['g'] = readAlphaWindow[slices[1]]
kywindow = readkywindow[slices2[1]]


# Initial condition

psi = dist.Field(name='psi', bases=(xbasis,ybasis))

if InitialType == 'Zero':
    psi['g'] = 0

if InitialType == 'Restart':
    snapshots = h5py.File(RestartFile, mode='r')
    psi.load_from_hdf5(file=snapshots,index=-1)

# Schotastic forcing

Fw = dist.Field(name='Fw', bases=(xbasis,ybasis))
if DepCase == 'Deep':
    Fwflip = dist.Field(name='Fwflip', bases=(xbasis,ybasis))
    Sym = dist.Field(name='Sym', bases=(xbasis,ybasis))

kfw = 0.2  * (2*np.pi/Lx)    # Forcing bandwidth
kx = xbasis.wavenumbers[dist.local_modes(xbasis)]
ky = ybasis.wavenumbers[dist.local_modes(ybasis)]
dkx = 2 * np.pi / Lx
dky = 2 * np.pi / Ly
seed = None     # Random seed
rand = np.random.RandomState(seed)
eta = epsilon * kf**2  # Enstrophy injection rate

def draw_gaussian_random_field():
    """Create Gaussian random field concentrating on a ring in Fourier space with unit variance."""
    #k = ((kx*Nx/Ny)**2 + ky**2)**0.5
    k = (kx**2 + ky**2)**0.5
    # 1D power spectrum: normalized Gaussian, no mean
    P1 = np.exp(-(k-kf)**2/2/kfw**2) / np.sqrt(kfw**2 * np.pi / 2) * (k != 0)
    # 2D power spectrum: divide by polar Jacobian
    P2 = P1 / 2 / np.pi / (k + (k==0))
    # 2D coefficient poewr spectrum: divide by mode power
    Pc = P2 / 2**((kx == 0).astype(float) + (ky == 0).astype(float) - 2)
    # Forcing amplitude, including division between sine and cosine
    f_amp = (Pc / 2 * dkx * dky)**0.5
    # Forcing with random phase
    f = f_amp * rand.randn(*k.shape)
    return f

def set_vorticity_forcing(timestep):
    """Set vorticity forcing field from scaled Gaussian random field."""
    # Set forcing to normalized Gaussian random field
    Fw['c'] = draw_gaussian_random_field()
    # Rescale by forcing rate, including factor for 1/2 in kinetic energy
    Fw['c'] *= (2 * eta / timestep)**0.5
    Fw.change_scales(1)

def set_symmetric_forcing(timestep):
    w_value = w.evaluate()
    Fwflip.preset_layout('c')
    Sym.preset_layout('c')
    Fwflip['c'] = Fw['c']*kywindow
    Fw.change_scales(1)
    Fw['g'] = Fw['g']*(1-Window) + Fwflip['g']*Window*2**0.5
    Sym['c'] = -w_value['c']*(1-kywindow)/timestep
    Sym.change_scales(1)
    Sym['g'] *= Window


## Setting up solver

# CFL parameters

max_dt = 1
initial_dt = safety*dx/20
print('Initial_dt=%e, max_dt=%e, stop_sim_time=%e, Output_Interval=%e'%(initial_dt,max_dt,stop_sim_time, Output_Interval))

tau_psi = dist.Field(name='tau_psi')

# Creating problem

u = -ey@d3.grad(psi)
v = ex@d3.grad(psi)
velo = ex*u+ey*v
#w = ex@d3.grad(v) - ey@d3.grad(u)
w = d3.lap(psi)
e=(u**2+v**2)/2
pe=(psi**2/Ld/Ld)/2

Advec = u * ex@d3.grad(w-psi/Ld/Ld) + v * ey@d3.grad(w-psi/Ld/Ld)
Betaterm = beta*v

Visc = nuhyper*d3.lap(d3.lap(w))
Damp = alpham*w

#Ein = Fw*psi
Ed = Damp*psi
if DepCase == 'Deep':
    Ed = Damp*AlphaWindow*psi
Ev = Visc*psi

problem = d3.IVP([psi, tau_psi], namespace=locals())
if DepCase == 'Deep':
    problem.add_equation("dt(w-psi/Ld/Ld) + Damp*AlphaWindow + Visc + Betaterm + tau_psi = -Advec +Fw +Sym")
if DepCase == 'Shallow':
    problem.add_equation("dt(w-psi/Ld/Ld) + Damp + Visc + Betaterm + tau_psi = -Advec +Fw")
problem.add_equation("integ(psi)=0")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# CFL
CFL = d3.CFL(solver, initial_dt=initial_dt, cadence=10, safety=safety, max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.05)
CFL.add_velocity(velo)

# Analysis

DiagnoseField1 = 'Data_flow_fields_'
DiagnoseScalar1 = 'Data_energy_'
DiagnoseField1 += Simulation_Name
DiagnoseScalar1 += Simulation_Name

snapshots = solver.evaluator.add_file_handler(DiagnoseField1, sim_dt=Output_Interval)
snapshots.add_task(psi, name='psi')
snapshots.add_task(u, name='u')
snapshots.add_task(v, name='v')
snapshots.add_task(w, name='w')

scalars = solver.evaluator.add_file_handler(DiagnoseScalar1, sim_dt=Output_Interval, mode='overwrite')
ave = d3.Average
scalars.add_task(ave(e), name='E')
scalars.add_task(ave(pe), name='PE')
#scalars.add_task(ave(Ein), name='Ein')
scalars.add_task(ave(Ev), name='Evisc')
scalars.add_task(ave(Ed), name='Edamp')

## Run the model

if DepCase == 'Deep':
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            set_vorticity_forcing(timestep)
            solver.step(timestep)
            set_symmetric_forcing(timestep)
            if (solver.iteration-1) % Diagnose_Interval == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()


if DepCase == 'Shallow':
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            set_vorticity_forcing(timestep)
            solver.step(timestep)
            if (solver.iteration-1) % Diagnose_Interval == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()





