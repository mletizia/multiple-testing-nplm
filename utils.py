from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest



import numpy as np
import os, h5py, glob
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl

class non_res(rv_continuous):
    def _pdf(self, x):
        return 256 * (x**2) * np.exp(- 8 * x)

def nonres_sig(N_S, seed):
    # this function can be used to generate non-resonant signal events.
    my_sig = non_res(momtype = 0, a=0, b=1, seed=seed)
    sig_sample = my_sig.rvs(size = N_S)
    return sig_sample

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

def candidate_sigma(data, perc=90):
    # this function estimates the width of the gaussian kernel.
    # use on a (small) sample of reference data (standardize first if necessary)
    pairw = pdist(data)
    return np.around(np.percentile(pairw,perc),3)


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


def err_bar(hist, n_samples):
    bins_counts = hist[0]
    bins_limits = hist[1]
    x   = 0.5*(bins_limits[1:] + bins_limits[:-1])
    bins_width = 0.5*(bins_limits[1:] - bins_limits[:-1])
    err = np.sqrt(np.array(bins_counts)/(n_samples*np.array(bins_width)))

    return x, err

def plot_data(data, label, name=None, dof=None, out_path=None, title=None,
                 density=True, bins=10,
                 c='mediumseagreen', e='darkgreen'):
    """
    Plot reference vs new physics t distribution
    Parameters
    ----------
    data : np.ndarray or list
        (N_toy,) array of observed test statistics
    dof : int
        degrees of freedom of the chi-squared distribution
    name : string
        filename for the saved figure
    out_path : string, optional
        output path where the figure will be saved. The default is ./fig.
    title : string
        title of the plot
    density : boolean
        True to normalize the histogram, false otherwise.
    bins : int or string, optional
        bins for the function plt.hist(). The default is 'fd'.
    Returns
    -------
    plot
    """
    plt.figure(figsize=(10,7))
    plt.style.use('classic')
    mpl.rc("figure", facecolor="white")


    hist = plt.hist(data, bins = bins, color=c, edgecolor=e,
                        density=density, label = str(label))
    x_err, err = err_bar(hist, data.shape[0])
    plt.errorbar(x_err, hist[0], yerr = err, color=e, marker='o', ms=6, ls='', lw=1,
                 alpha=0.7)

    plt.ylim(bottom=0)

    # results data
    md_t = round(np.median(data), 2)
    if dof:
        z_chi2 = round(chi2_zscore(np.median(data),dof=dof),2)
    if dof:
        res = "md t = {} \nZ_chi2 = {}".format(md_t,z_chi2)
    else:
        res = "md t = {}".format(md_t)
    # plot chi2 and set xlim
    if dof:
        chi2_range = chi2.ppf(q=[0.00001,0.999], df=dof)
        x = np.arange(chi2_range[0], chi2_range[1], .05)
        chisq = chi2.pdf(x, df=dof)
        plt.plot(x, chisq, color='#d7191c', lw=2, label='$\chi^2(${}$)$'.format(dof))
        xlim = (min(chi2_range[0], min(data)-5), max(chi2_range[1], max(data)+5))
        plt.xlim(chi2_range)
    else:
        xlim = (min(data)-5, max(data)+5)
        plt.xlim(xlim)

    if title:
        plt.title(title, fontsize=20)

    plt.ylabel('P(t)', fontsize=20)
    plt.xlabel('t', fontsize=20)

    # Axes ticks
    ax = plt.gca()
    plt.legend(loc ="upper right", frameon=True, fontsize=18)
    ax.text(0.75, 0.65, res, color='black', fontsize=12,
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=.5'),transform = ax.transAxes)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path+"/data_{}.pdf".format(name), bbox_inches='tight')
    plt.show()
    plt.close()


def plot_ref_data(ref, data, name=None, dof=None, out_path=None, title=None,
                 density=True, bins=10, xlim=None,
                 c_ref='#abd9e9', e_ref='#2c7bb6', c_sig='#fdae61', e_sig='#d7191c'):
    """
    Plot reference vs new physics t distribution
    Parameters
    ----------
    T_ref : np.ndarray or list
        (N_toy,) array of observed test statistics in the reference hypothesis
    T_sig : np.ndarray or list
        (N_toy,) array of observed test statistics in the New Physics hypothesis
    dof : int
        degrees of freedom of the chi-squared distribution
    name : string
        filename for the saved figure
    out_path : string, optional
        output path where the figure will be saved. The default is ./fig.
    title : string
        title of the plot
    density : boolean
        True to normalize the histogram, false otherwise.
    bins : int or string, optional
        bins for the function plt.hist(). The default is 'fd'.
    Returns
    -------
    plot
    """
    plt.figure(figsize=(10,7))
    plt.style.use('classic')
    mpl.rc("figure", facecolor="white")

    #set uniform bins across all data points
    bins = np.histogram(np.hstack((ref,data)), bins = bins)[1]
    # reference
    hist_ref = plt.hist(ref, bins = bins, color=c_ref, edgecolor=e_ref,
                        density=density, label = 'Reference')
    x_err, err = err_bar(hist_ref, ref.shape[0])
    plt.errorbar(x_err, hist_ref[0], yerr = err, color=e_ref, marker='o', ms=6, ls='', lw=1,
                 alpha=0.7)
    # data
    hist_sig = plt.hist(data, bins = bins, color=c_sig, edgecolor=e_sig,
                        alpha=0.7, density=density, label='Data')
    x_err, err = err_bar(hist_sig, data.shape[0])
    plt.errorbar(x_err, hist_sig[0], yerr = err, color=e_sig, marker='o', ms=6, ls='', lw=1,
                 alpha=0.7)
    plt.ylim(bottom=0)
    # results data
    md_tref = round(np.median(ref), 2)
    md_tdata = round(np.median(data), 2)
    max_zemp = emp_zscore(ref,np.max(ref))
    zemp = emp_zscore(ref,np.median(data))
    if dof:
        z_chi2 = round(chi2_zscore(np.median(data),dof=dof),2)

    if dof:
        res = "md t_ref = {} \nmd t_data = {} \nmax Z_emp = {}  \nZ_emp = {} \nZ_chi2 = {}".format(
            md_tref,
            md_tdata,
            max_zemp,
            zemp,
            z_chi2
        )
    else:
        res = "md tref = {} \nmd tdata = {} \nmax Zemp = {} \nZemp = {}".format(
            md_tref,
            md_tdata,
            max_zemp,
            zemp
        )

    # plot chi2 and set xlim
    if dof:
        chi2_range = chi2.ppf(q=[0.00001,0.999], df=dof)
        #r_len = chi2_range[1] - chi2_range[0]
        x = np.arange(chi2_range[0], chi2_range[1], .05)
        chisq = chi2.pdf(x, df=dof)
        plt.plot(x, chisq, color='#d7191c', lw=2, label='$\chi^2(${}$)$'.format(dof))
        XLIM = (min(chi2_range[0], min(ref)-1), max(chi2_range[1], max(data)+1))
        plt.xlim(XLIM)
    elif xlim: plt.xlim(xlim)
    else:
        XLIM = (min(ref)-1, max(data)+1)
        plt.xlim(XLIM)
    if title:
        plt.title(title, fontsize=20)

    plt.ylabel('P(t)', fontsize=20)
    plt.xlabel('t', fontsize=20)

    # Axes ticks
    ax = plt.gca()

    plt.legend(loc ="upper right", frameon=True, fontsize=18)

    ax.text(0.75, 0.55, res, color='black', fontsize=12,
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=.5'),transform = ax.transAxes)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path+"/refdata_{}.pdf".format(name), bbox_inches='tight')
    plt.show()
    plt.close()

def learn_t(X_train, Y_train, weight, flk_config, seed):
    flk_config['seed']=seed # select different centers for different toys
    preds = trainer(X_train,Y_train,flk_config)
    return  compute_t(preds,Y_train,weight)
