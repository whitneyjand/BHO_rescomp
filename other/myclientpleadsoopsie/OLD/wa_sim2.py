import sys
import numpy as np
import rescomp as rc
# import dill as pickle
import networkx as nx
from other._common import *
from other.my_rcopt import *
from scipy import sparse
from matplotlib import pyplot as plt
# from rescomp import optimizer as rcopt
from ipyparallel import Client
from bayes_opt import BayesianOptimization as bopt

import warnings
warnings.filterwarnings("ignore")


##########################################################################################
###################################### GET VPT TIME ######################################
##########################################################################################

def nrmse(true, pred):
    """ Normalized root mean square error. (A metric for measuring difference in orbits)
    Parameters:
        Two mxn arrays. Axis zero is assumed to be the time axis (i.e. there are m time steps)
    Returns:
        err (ndarray): Error at each time value. 1D array with m entries
    """
    sig = np.std(true, axis=0)
    err = np.linalg.norm((true-pred) / sig, axis=1, ord=2)
    return err

def valid_prediction_index(err, tol):
    """First index i where err[i] > tol. err is assumed to be 1D and tol is a float. If err is never greater than tol, then len(err) is returned."""
    mask = np.logical_or(err > tol, ~np.isfinite(err))
    if np.any(mask):
        return np.argmax(mask)
    return len(err)

def get_vptime(system, ts, Uts, pre, vpttol=0.5):
    """
    Valid prediction time for a specific instance.
    """
    err = nrmse(Uts, pre)
    idx = valid_prediction_index(err, vpttol)
    if idx == 0:
        vptime = 0.
    else:
        if system.is_driven:
            vptime = ts[0][idx-1] - ts[0][0]
        else:
            vptime = ts[idx-1] - ts[0]
        
    return vptime

##########################################################################################
###################################### REMOVE EDGES ######################################
##########################################################################################

def remove_edges(A,nedges):
    """ Randomly removes 'nedges' edges from a sparse matrix 'A'
    """
    B = A.copy().todok() # - - - - - - - -  set A as copy

    keys = list(B.keys()) # - - - - remove edges
    
    remove_idx = np.random.choice(range(len(keys)),size=nedges, replace=False)
    remove = [keys[i] for i in remove_idx]
    for e in remove:
        B[e] = 0
    return B

##########################################################################################
########################################## MAIN ##########################################
##########################################################################################


if __name__ == '__main__':
    
    # SET PARAMS
    n = 500
    p = 1/250
    pthin = sys.argv[1]
    rho = sys.argv[2]
    system_name = 'lorenz'
    system = get_system(system_name)

    # DOUBLE CHECK CORRECT PARAMS
    print('pthin: ',pthin)
    print('rho: ',rho,'\n')

    # SET UP PARALLEL
    client = Client(profile=sys.argv[3])
    dview = client[:]
    bview = client.load_balanced_view()

    # IMPORT PACKAGES
    dview.block = True
    dview.execute('import networkx as nx')
    dview.execute('import sys')
    dview.execute('import rescomp as rc')
    dview.execute('from scipy import sparse')
    dview.execute('import numpy as np')
    dview.execute('from _common import *')
    dview.execute('from my_rcopt import *')
    # dview.execute('from bayes_opt import BayesianOptimization as bopt')


    # BAYESIAN OPTIMIZATION
    def function_to_be_optimized(g, s, a):

        def run_once(n_):
            print('define the stuff')
            class System:
                """
                Abstract class for defining a system for a reservoir computer to train and predict on.
                Requires implementing get_train_test_data(), get_random_test(), and optionally df_eq()
                """
                def __init__(self, name, train_time, test_time, dt, signal_dim, drive_dim=0, is_diffeq=False, is_driven=False):
                    self.name=name
                    self.train_time=train_time
                    self.test_time=test_time
                    self.dt=dt
                    self.signal_dim=signal_dim
                    self.drive_dim=drive_dim
                    self.is_diffeq=is_diffeq
                    self.is_driven=is_driven
                
                def get_train_test_data(self, cont_test=True):
                    """
                    Returns training and test data for use by a reservoir computer.
                    
                    Arguments:
                        cont_test (bool): True for continued test, False for random test
                    
                    Returns:
                        If not driven:
                            tr, Utr, ts, Uts
                        If driven:
                            tr, (Utr, Dtr), (ts, Dts), Uts
                    Where:
                        tr, ts ((n,) ndarray): training/test time values
                        Utr, Uts ((n,m) ndarray): training/test system state values
                        Dtr, Dts ((n,l) ndarray): training/test system driving values
                    """
                    raise NotImplemented("this function has not been implemented")
                
                def get_random_test(self):
                    """
                    Returns test data from an arbitrary initial condition.
                    
                    Returns:
                        If not driven:
                            ts, Uts
                        If driven:
                            (ts, Dts), Uts
                    Where:
                        ts ((n,) ndarray): training/test time values
                        Uts ((n,m) ndarray): training/test system state values
                        Dts ((n,l) ndarray): training/test system driving values
                    """
                    raise NotImplemented("this function has not been implemented")
                
                def df_eq(self, t, U, D=None):
                    """
                    The differential equation governing the system, if applicable.
                    Only used if self.is_diffeq is True.
                    
                    Parameters:
                        t: 1-dimensional array, the timesteps of the system
                        U: 2-dimensional array, describing the system state at either one point or over a time series.
                        D: 2-dimensional array, describing the system's drive state at either one point or over a time series. Only ever passed if self.is_driven is True.
                            U[t,:] is the system's state at time t, and likewise for D.
                    """
                    raise NotImplemented("this function has not been implemented")
                
            def unpack(X, cols=True):
                """ Splits 2d numpy arrays into tuples 
                    Parameters
                    ----------
                    X (ndarray): 2d numpy array
                    cols (bool): If True, split the array into column vectors, if false split it into row vectors
                    
                    Returns
                    -------
                    unpack (tuple): A tuple of the rows of X or a tuple of the columns of X. 
                        If X is not a 2d numpy array, unpack=X.
                """
                if type(X) is np.ndarray:
                    if len(X.shape) > 1:
                        m, n = X.shape
                        if cols:
                            unpack = tuple([np.reshape(X[:,i], (m, 1)) for i in range(n)])
                        else:
                            unpack = tuple([X[i, ] for i in range(m)])
                        return unpack
                return X

            def lorenz(t, X, sigma=10., beta=8./3, rho=28.0):
                """Compute the time-derivative of a Lorenz system."""
                (x, y, z) = unpack(X)
                return np.hstack((sigma * (y - x), x * (rho - z) - y, x * y - beta * z))

            SYSTEMS = {
                "lorenz" : {
                    "domain_scale" : [20, 20, 20],
                    "domain_shift" : [-10, -10, 10],
                    "signal_dim" : 3,
                    "time_to_attractor" : 40.0,
                    "df" : lorenz,
                    "rcomp_params" : {
                                    "res_sz" : 1000,
                                    "activ_f" : lambda x: 1/(1 + np.exp(-1*x)),
                                    "gamma" : 5.632587,
                                    "mean_degree" : 0.21,
                                    "ridge_alpha" : 2e-7,
                                    "sigma" : 0.078,
                                    "spect_rad" : 14.6
                    }}}

            def random_initial(system):
                a = SYSTEMS[system]["domain_scale"]
                b = SYSTEMS[system]["domain_shift"]
                dim = SYSTEMS[system]["signal_dim"]
                u0 = a * np.random.rand(dim) + b
                return u0

            def orbit(system, initial=None, duration=10, dt=0.01, trim=False):
                """ Returns the orbit of a given system.
                
                    Parameters
                    ----------
                    system (str): A supported dynamical system from ["lorenz", "thomas", "rossler"]
                    initial (ndarray): An initial condition for the system. Defaults to a random choice.
                    duration (float): Time duration of the orbit (default duration=10 means 10 seconds)
                    dt (float): Timestep size. Default dt=0.01
                    trim (bool): Option to trim off transient portion of the orbit (To ensure the orbit 
                        is on the chaotic attractor for the full duration.)
                        
                    Returns
                    -------
                    U (ndarray): mxn numpy array where m is the number of timesteps (duration x dt) and n is 
                        dimension of the system (probably 3).
                """
                transient_timesteps = 0
                time_to_attractor = 0
                if initial is None:
                    initial = random_initial(system)
                if trim:
                    time_to_attractor = SYSTEMS[system]["time_to_attractor"]
                    transient_timesteps = int(time_to_attractor / dt)
                timesteps = int(duration / dt) + 1
                # Make enough timesteps so that the transients can be trimmed leaving a full duration orbit
                t = np.linspace(0, time_to_attractor + duration, transient_timesteps + timesteps)
                # Locate the correct derivative function
                df = SYSTEMS[system]["df"]
                U = integrate.odeint(df, initial, t, tfirst=True)
                # Trim off transient states
                U, t = U[transient_timesteps: , :], t[transient_timesteps:]
                return t, U

            def train_test_orbit(system, duration=10, dt=0.01, trainper=0.5, trim=True):
                """ Returns a time scale and orbit of length `duration` split into two pieces:
                        tr, Utr, ts, Uts
                        where `tr` contains `trainper` percent of the total orbit.
                        The output of numpy.vstack((Utr, Uts)) will be an unbroken orbit
                        from the given system.
                    Parameters
                    ----------
                        system (str): A builtin rescomp system name ["rossler", "thomas", "lorenz"]
                        duration (float): How long of an orbit
                            Defaults to 10.0
                        dt (float): Stepsize in time (For numerical integration)
                            Defaults to 0.01
                        trainper (float): Must be between 0 and 1. Percent of the orbit to place in training data
                            Defaults to 0.5.
                        trim (bool): If true, return an orbit of lenfth duration on the attractor. Otherwise,
                            include the pre attractor orbit. (Defaults to True)
                    Returns
                    -------
                        tr (ndarray): 1 dimensional array of time values corresponding to training orbit
                        Utr (ndarray): 2 dimensional training orbit. Utr[i, :] is the state of the system at time tr[i]
                        ts (ndarray): 1 dimensional array of time values corresponding to test orbit
                        Uts (ndarray): 2 dimensional test orbit. Uts[i, :] is the state of the system at time ts[i]
                """
                t, U = rc.orbit(system, duration=duration, dt=dt, trim=trim)

                N = len(t)
                mid = int(N * trainper)
                tr, Utr = t[:mid], U[:mid, :]
                ts, Uts = t[mid:], U[mid:, :]
                return tr, Utr, ts, Uts

            class ChaosODESystem(System):
                """
                Class that implements the Lorenz, Thomas, and Rossler systems.
                """
                def __init__(self, name, train_time, test_time, dt):
                    if name not in {'lorenz','thomas','rossler'}:
                        raise ValueError("Unsupported system type by this class")
                    
                    super().__init__(name, train_time, test_time, dt, signal_dim=3, is_diffeq=True, is_driven=False)
                    self.df = rc.SYSTEMS[name]['df']
                
                def get_train_test_data(self, cont_test=True):
                    if cont_test:
                        duration = self.train_time + self.test_time
                        trainper = self.train_time / duration
                        tr, Utr, ts, Uts = rc.train_test_orbit(self.name, duration=duration, trainper=trainper, dt=self.dt)
                    else:
                        tr, Utr = rc.orbit(self.name, duration=self.train_time, trim=True)
                        ts, Uts = rc.orbit(self.name, duration=self.test_time, trim=True)
                    return tr, Utr, ts, Uts
                
                def get_random_test(self):
                    return rc.orbit(self.name, duration=self.test_time, trim=True)

            def get_system(system_name):
                """
                Gets the system with the given name.
                If system_name is one of 'lorenz', 'thomas', 'rossler', or 'softrobot', uses the predefined system object.
                Otherwise, attempts to load a file.
                
                For 'lorenz', 'thomas', 'rossler', and 'softrobot', the train_time, test_time, and dt parameters can be
                specified as  keyword arguments.
                
                Returns a rescomp.optimizer.System object
                """
                
                #Numerical parameters are, in order:
                #   -train time
                #   -test time
                #   -dt
                if system_name == 'lorenz':
                    params = {'train_time':6.6, 'test_time':8., 'dt':0.01}
                    return ChaosODESystem('lorenz', **params)

            def nrmse(true, pred):
                """ Normalized root mean square error. (A metric for measuring difference in orbits)
                Parameters:
                    Two mxn arrays. Axis zero is assumed to be the time axis (i.e. there are m time steps)
                Returns:
                    err (ndarray): Error at each time value. 1D array with m entries
                """
                sig = np.std(true, axis=0)
                err = np.linalg.norm((true-pred) / sig, axis=1, ord=2)
                return err

            def valid_prediction_index(err, tol):
                """First index i where err[i] > tol. err is assumed to be 1D and tol is a float. If err is never greater than tol, then len(err) is returned."""
                mask = np.logical_or(err > tol, ~np.isfinite(err))
                if np.any(mask):
                    return np.argmax(mask)
                return len(err)

            def get_vptime(system, ts, Uts, pre, vpttol=0.5):
                """
                Valid prediction time for a specific instance.
                """
                err = nrmse(Uts, pre)
                idx = valid_prediction_index(err, vpttol)
                if idx == 0:
                    vptime = 0.
                else:
                    if system.is_driven:
                        vptime = ts[0][idx-1] - ts[0][0]
                    else:
                        vptime = ts[idx-1] - ts[0]
                    
                return vptime

            def remove_edges(A,nedges):
                """ Randomly removes 'nedges' edges from a sparse matrix 'A'
                """
                B = A.copy().todok() # - - - - - - - -  set A as copy

                keys = list(B.keys()) # - - - - remove edges
                
                remove_idx = np.random.choice(range(len(keys)),size=nedges, replace=False)
                remove = [keys[i] for i in remove_idx]
                for e in remove:
                    B[e] = 0
                return B

            pthin = 0.98
            rho = 0.5

            # GET TRAINING AND TESTING SIGNALS
            system_name = 'lorenz'
            system = get_system(system_name)
            tr, Utr, ts, Uts = rc.train_test_orbit(system_name, duration=1000, trainper=980/1000)

            # CREATE GRAPH
            A = nx.erdos_renyi_graph(500,1/250,directed=True)
            num_edges = len(A.edges)
            A = sparse.dok_matrix(nx.adj_matrix(A).T)
            A = remove_edges(A,nedges = int((pthin * num_edges)//1))
            
            # MAKE RESERVOIR
            res = rc.ResComp(A.tocoo(), spect_rad=rho, sigma=s, gamma=g, ridge_alpha=a)
            res.train(tr, Utr)
            
            # GET VPT
            pred, p_sig = res.predict(ts, r0=res.r0, return_states=True)
            vpt = get_vptime(system, ts, Uts, pred, vpttol=0.6)
            return vpt
        
        vpts = bview.map_sync(run_once, range(20))
        return np.mean(vpts)
        
    
    optimizer = bopt(
        f=function_to_be_optimized,
        pbounds={'g': (0.1, 50), 's': (1e-3, 10), 'a': (1e-8, 1)},
        verbose=2
    )
    
    optimizer.set_gp_params(alpha=1e-3)

    sys.stdout = open(f'wa_simlog_{pthin}_{rho}.txt', 'a')
    print(optimizer.maximize(init_points=10, n_iter=15))

    #print("\nBest result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

    print('\n############################################################################################')
    print('\n############################################################################################')

    # LOG FILE
    #sys.stdout = open(f'wa_simlog_{pthin}_{rho}.txt', 'a')
    print('pthin: ',pthin)
    print('rho: ',rho,'\n')
    print("\nBest result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))
    sys.stdout.close()
    
