import sys
import numpy as np
import rescomp as rc
import networkx as nx
from scipy import sparse
from ipyparallel import Client
from scipy.interpolate import CubicSpline
from bayes_opt import BayesianOptimization as bopt

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # SET PARAMS
    rho = sys.argv[1]
    pthin = sys.argv[2]
    
    init_points=20
    n_iter=15
    per = 50

    alpha=0.001

    # DOUBLE CHECK CORRECT PARAMS
    print('pthin: ',pthin)
    print('rho: ',rho,'\n')

    # SET UP PARALLEL
    client = Client(profile=sys.argv[3])
    dview = client[:]
    bview = client.load_balanced_view()

    # IMPORT PACKAGES
    dview.block = True
    dview.execute('import numpy as np')
    dview.execute('import rescomp as rc')
    dview.execute('import networkx as nx')
    dview.execute('from scipy import sparse')
    dview.execute('from scipy.interpolate import CubicSpline')

     # BAYESIAN OPTIMIZATION
    def function_to_be_optimized(g, s, a):
        
        def run_once(n_):
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

            def wa_vptime(ts, Uts, pre, vpttol=0.5):
                """
                Valid prediction time for a specific instance.
                """
                err = nrmse(Uts, pre)
                idx = valid_prediction_index(err, vpttol)
                if idx == 0:
                    vptime = 0.
                else:
                    vptime = ts[idx-1] - ts[0]
        
                return vptime

            # SET PARAMS
            rho = 50
            pthin = 1.00

            # OTHER PARAMS
            n = 500
            c = 1
            
            # GET TRAINING AND TESTING SIGNALS
            t, U = rc.orbit('lorenz', duration=100)
            u = CubicSpline(t, U)
            tr = t[:9000]
            Utr = u(t[:9000])
            ts = t[9000:]
            Uts = u(t[9000:])

            # CREATE GRAPH
            A = nx.erdos_renyi_graph(n,c*(1-pthin)/((n-1)),directed=True)
            A = sparse.dok_matrix(nx.adj_matrix(A).T)
            
            # MAKE RESERVOIR
            res = rc.ResComp(A.tocoo(), spect_rad=rho, sigma=s, gamma=g, ridge_alpha=a)
            res.train(tr, Utr)
            
            # GET VPT
            Upred, _ = res.predict(ts, r0=res.r0, return_states=True)
            vpt = wa_vptime(ts[:-2], Uts[:-2], Upred[2:], vpttol=0.5)

            return vpt

        vpts = np.sort(bview.map_sync(run_once, range(per)))
        print('\n\n',repr(vpts))
        print('mean:',np.mean(vpts))
        print('var:',np.var(vpts))
        return np.mean(vpts)
            
    
    optimizer = bopt(
        f=function_to_be_optimized,
        pbounds={'g': (0.1, 50), 's': (1e-3, 10), 'a': (1e-8, 1)},
        verbose=2
    )

    # SET ALPHA PARAM - ACCOUNT FOR EXTRA FLEXIBILITY
    optimizer.set_gp_params(alpha=alpha)
    sys.stdout = open(f'wa_alpha_test_{rho}_{pthin}_a_{alpha}_2.txt', 'a')
    print(optimizer.maximize(init_points=init_points, n_iter=n_iter))

    print('num_trials: ',init_points,' inits, ',n_iter,' iters, ',per,' per')
    print('alpha: ',alpha)
    print('rho: ',rho)
    print('pthin: ',pthin)
    print("\nBest VPT: {}".format(optimizer.max["target"]))
    print("Best result: {}".format(optimizer.max["params"]))
    sys.stdout.close()