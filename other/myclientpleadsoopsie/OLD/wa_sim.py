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

    # LOG RESULTS
    # stdoutOrigin=sys.stdout 
    # sys.stdout = open('test2.txt', 'w')
    best_vpt = dict()
    best_params = dict()
    print('pthin: ',pthin)
    print('rho: ',rho,'\n')

    client = Client(profile=sys.argv[3])
    dviw = client[:]
    bview = client.load_balanced_view()

    for i in range(100):
        print('\n\n##############################################')

        # GET TRAINING AND TESTING SIGNALS
        system_name = 'lorenz'
        system = get_system(system_name)
        tr, Utr, ts, Uts = rc.train_test_orbit(system_name, duration=1000, trainper=980/1000)

        
        # CREATE UNTHINNED GRAPH
        A = nx.erdos_renyi_graph(n,p,directed=True)
        # nx.draw_networkx(A,node_size=1)
        num_edges = len(A.edges)
        A = sparse.dok_matrix(nx.adj_matrix(A).T)

        
        # THIN GRAPH
        A = remove_edges(A,nedges = int(pthin * num_edges))

        
        # BAYESIAN OPTIMIZATION
        def function_to_be_optimized(g, s, a):
            res = rc.ResComp(A.tocoo(), spect_rad=rho, sigma=s, gamma=g, ridge_alpha=a)
            res.train(tr, Utr)
            
            pred, p_sig = res.predict(ts, r0=res.r0, return_states=True)
            vpt = get_vptime(system, ts, Uts, pred, vpttol=0.6)
            return vpt
        
        optimizer = bopt(
            f=function_to_be_optimized,
            pbounds={'g': (0.1, 50), 's': (1e-3, 10), 'a': (1e-8, 1)},
            verbose=2
        )
        
        optimizer.set_gp_params(alpha=1e-3)
        optimizer.maximize(init_points=10, n_iter=20)

        print("\nBest result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))
        best_vpt[i] = optimizer.max["target"]
        best_params[i] = optimizer.max["params"]

    print('\n############################################################################################')
    print('\n############################################################################################')

    # CLOSE LOG FILE
    sys.stdout = open(f'wa_simlog_{pthin}_{rho}.txt', 'a')
    print('pthin: ',pthin)
    print('rho: ',rho,'\n')
    print(f'best_vpt = {best_vpt}')
    print(f'\n\nbest_params = {best_params}')
    sys.stdout.close()
