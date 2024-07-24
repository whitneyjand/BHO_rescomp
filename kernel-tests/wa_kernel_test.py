import sys
import numpy as np
import rescomp as rc
import networkx as nx
from scipy import sparse
from ipyparallel import Client
import sklearn.gaussian_process as GP
from scipy.interpolate import CubicSpline
from bayes_opt import BayesianOptimization as bopt


import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # SET PARAMS
    nu = float(sys.argv[1])
    ls = float(sys.argv[2])
    alpha=float(sys.argv[3])
    
    init_points=30
    n_iter=30
    per=50

    # SET UP PARALLEL
    client = Client(profile=sys.argv[4])
    dview = client[:]
    bview = client.load_balanced_view()

    # IMPORT PACKAGES
    dview.block = True
    dview.execute('import numpy as np')
    dview.execute('import rescomp as rc')
    dview.execute('import networkx as nx')
    dview.execute('from scipy import sparse')
    dview.execute('import sklearn.gaussian_process as GP')
    dview.execute('from scipy.interpolate import CubicSpline')

     # BAYESIAN OPTIMIZATION
    def function_to_be_optimized(g, s, a):

        def run_once(n_):
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
                err = np.linalg.norm((Uts-pre), axis=1, ord=2)
                idx = valid_prediction_index(err, vpttol)
                if idx == 0:
                    vptime = 0.
                else:
                    vptime = ts[idx-1] - ts[0]
        
                return vptime

            # SET PARAMS
            rho = 50
            pthin = 0.90

            # OTHER PARAMS
            n = 500
            c = 4
            
            # GET TRAINING AND TESTING SIGNALS
            t, U = rc.orbit('lorenz', duration=150)
            u = CubicSpline(t, U)
            tr = t[:4000]
            Utr = u(t[:4000])
            ts = t[4000:]
            Uts = u(t[4000:])    
            
            try:
                
                if pthin == 0:
                    res = rc.ResComp(sparse.dok_matrix(np.zeros((n,n))).tocoo(), spect_rad=rho, sigma=s, gamma=g, ridge_alpha=a)
                else:
                    # CREATE GRAPH
                    A = nx.erdos_renyi_graph(n,c*(1-pthin)/(n-1),directed=True)
                    A = sparse.dok_matrix(nx.adj_matrix(A).T)        

                    # SCALE AND MAKE RESERVOIR
                    B = A*(rho/np.abs(sparse.linalg.eigs(A.astype(float),k=1)[0][0]))
                    res = rc.ResComp(B.tocoo(), spect_rad=rho, sigma=s, gamma=g, ridge_alpha=a)
                res.train(tr, Utr)

                # GET VPT
                Upred, _ = res.predict(ts, r0=res.r0, return_states=True)
                vpt = wa_vptime(ts[:-2], Uts[:-2], Upred[2:], vpttol=5)

                return vpt
            
            except:
                pass

        vpts = bview.map_sync(run_once, range(per))
        vpts = np.array(list(filter(lambda item: item is not None, vpts)))
        print(np.sort(vpts))
        print('var:',np.var(vpts))
        return np.mean(vpts)
            
    
    optimizer = bopt(
        f=function_to_be_optimized,
        pbounds={'g': (0.1, 50), 's': (1e-3, 10), 'a': (1e-8, 1)},
        verbose=2
    )

    # SET KERNEL AND ALPHA PARAMS 
    optimizer.set_gp_params(alpha=alpha, kernel=GP.kernels.Matern(nu=nu, length_scale=ls))
    sys.stdout = open(f'wa_kernel_test__ls_{ls}_nu_{nu}_alpha_{alpha}.txt', 'a')
    print(optimizer.maximize(init_points=init_points, n_iter=n_iter))

    print('num_trials: ',init_points,' inits, ',n_iter,' iters, ',per,' per')
    print('rho: ',50)
    print('pthin: ',0.90)
    print("\nBest VPT: {}".format(optimizer.max["target"]))
    print("Best result: {}".format(optimizer.max["params"]))
    print(f"\nKERNEL: nu={nu}, lengthscale={ls}, alpha={alpha}")
    sys.stdout.close()
