from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest



import numpy as np
import os, time, h5py, glob
import torch

'''
def normalize(X):
    """Standardize dataset
    Args:
        X (np.ndarray): Original Dataset
    Returns:
        np.ndarray: Normalized Dataset
    """    

    X_norm = X.copy()
    
    for j in range(X_norm.shape[1]):
        column = X_norm[:, j]

        mean = np.mean(column)
        std = np.std(column)
    
        if np.min(column) < 0:
            column = (column-mean)*1./ std
        elif np.max(column) > 1.0:                                                                                                                                        
            column = column *1./ mean
    
        X_norm[:, j] = column
    
    return X_norm
'''

def normalize(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        vec  = dataset[:, j]
        if np.min(vec) < 0:
            vec = vec- mean
            vec = vec *1./ std
        elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.                                                                                                            \                                                                                                                                                                                             
            vec = vec *1./ mean
        dataset_new[:, j] = vec
    return dataset_new

class non_res(rv_continuous):
    def _pdf(self, x):
        return (1/2) * (x**2) * np.exp(- x) # C = 1/2 for x^2 exp(-x), C = 256 for x^2 exp(-8x)

def nonres_sig(N_S, seed):
    # this function can be used to generate non-resonant signal events.
    my_sig = non_res(momtype = 0, a=0, b=None, seed=seed) # momtype = 0 is pdf, a lower bound (default=-inf), b upper bound (default=+inf)
    sig_sample = my_sig.rvs(size = N_S)
    return sig_sample



def get_logflk_config(M,flk_sigma,lam,weight,iter=[1000000],seed=None,cpu=False):
    # it returns logfalkon parameters
    return {
            'kernel' : GaussianKernel(sigma=flk_sigma),
            'M' : M, #number of Nystrom centers,
            'penalty_list' : lam, # list of regularization parameters,
            'iter_list' : iter, #list of number of CG iterations,
            'options' : FalkonOptions(cg_tolerance=np.sqrt(float(1e-7)), keops_active='no', use_cpu=cpu, debug = False),
            'seed' : seed, # Random seed. Can be used to make results stable across runs.
                            # Randomness is present in the center selection algorithm, and in
                            # certain optimizers.
            'loss' : WeightedCrossEntropyLoss(kernel=GaussianKernel(sigma=flk_sigma), neg_weight=weight),
            }


def compute_t(preds,Y,weight):
    # it returns extended log likelihood ratio from predictions
    diff = weight*np.sum(1 - np.exp(preds[Y==0]))
    return 2 * (diff + np.sum(preds[Y==1]))

def trainer(X,Y,flk_config):
    # trainer for logfalkon model
    Xtorch=torch.from_numpy(X)
    Ytorch=torch.from_numpy(Y)
    model = LogisticFalkon(**flk_config)
    model.fit(Xtorch, Ytorch)
    return model.predict(Xtorch).numpy()

'''
def standardize(X):
    # standardize data as in HIGGS and SUSY
    for j in range(X.shape[1]):
        column = X[:, j]
        mean = np.mean(column)
        std = np.std(column)
        if np.min(column) < 0:
            column = (column-mean)*1./ std
        elif np.max(column) > 1.0:
            column = column *1./ mean
        X[:, j] = column

    return X
'''
def return_best_chi2dof(tobs):
    """
    Returns the most fitting value for dof assuming tobs follows a chi2_dof distribution,
    computed with a Kolmogorov-Smirnov test, removing NANs and negative values.
    Parameters
    ----------
    tobs : np.ndarray
        observations
    Returns
    -------
        best : tuple
            tuple with best dof and corresponding chi2 test result
    """

    dof_range = np.arange(np.nanmedian(tobs) - 10, np.nanmedian(tobs) + 10, 0.1)
    ks_tests = []
    for dof in dof_range:
        test = kstest(tobs, lambda x:chi2.cdf(x, df=dof))[0]
        ks_tests.append((dof, test))
    ks_tests = [test for test in ks_tests if test[1] != 'nan'] # remove nans
    ks_tests = [test for test in ks_tests if test[0] >= 0] # retain only positive dof
    best = min(ks_tests, key = lambda t: t[1]) # select best dof according to KS test result

    return best

def emp_zscore(t0,t1):
    if max(t0) <= t1:
        p_obs = 1 / len(t0)
        Z_obs = round(norm.ppf(1 - p_obs),2)
        return Z_obs
    else:
        p_obs = np.count_nonzero(t0 >= t1) / len(t0)
        Z_obs = round(norm.ppf(1 - p_obs),2)
        return Z_obs

def chi2_zscore(t1, dof):
    p = chi2.cdf(float('inf'),dof)-chi2.cdf(t1,dof)
    return norm.ppf(1 - p)




def learn_t(X_train, Y_train, weight, flk_config, seed):
    flk_config['seed']=seed # select different centers for different toys
    preds = trainer(X_train,Y_train,flk_config)
    return  compute_t(preds,Y_train,weight)



def BuildSample_DY(N_Events, INPUT_PATH, rng):

    files = glob.glob(INPUT_PATH + '/*.h5')
    nfiles = len(files)
    #random integer to select Zprime file between n files                                                                                                                
    u = np.arange(nfiles)#np.random.randint(100, size=100)                                                                                                               
    rng.shuffle(u)
    #BACKGROUND                                                                                                                                                          
    #extract N_Events from files                                                                                                                                         
    toy_label = INPUT_PATH.split("/")[-2]
    print(toy_label)

    HLF = np.array([])

    for u_i in u:
        if not os.path.exists(INPUT_PATH+toy_label+str(u_i+1)+".h5"): continue
        f = h5py.File(INPUT_PATH+toy_label+str(u_i+1)+".h5", 'r')
        keys=list(f.keys())
        #check whether the file is empty                                                                                                                                  
        if len(keys)==0:
            continue
        cols=np.array([])
        for i in range(len(keys)):
            feature = np.array(f.get(keys[i]))
            feature = np.expand_dims(feature, axis=1)
            if i==0:
                cols = feature
            else:
                cols = np.concatenate((cols, feature), axis=1)

        rng.shuffle(cols) #don't want to select always the same event first                                                                                        

        if HLF.shape[0]==0:
            HLF=cols
            i=i+1
        else:
            HLF=np.concatenate((HLF, cols), axis=0)
        f.close()
        #print(HLF_REF.shape)                                                                                                                                             
        if HLF.shape[0]>=N_Events:
            HLF=HLF[:N_Events, :]
            break
    #print('HLF shape')                                                                                                                                                   
    print(HLF.shape)
    return HLF