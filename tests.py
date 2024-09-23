import numpy as np
import sys
from scipy.stats import norm, beta
from scipy.special import logsumexp

# discrepancies are considered right-sided by default
# hence tests based on the p-value have a minus sign in front 
# because more discrepant results return smaller values

def ClopperPearson_interval(total, passed, level):
    low_b, up_b = 0.5*(1-level), 0.5*(1+level)
    low_q=beta.ppf(low_b, passed, total-passed+1, loc=0, scale=1)
    up_q=beta.ppf(up_b, passed, total-passed+1, loc=0, scale=1)
    return np.around(passed*1./total,5), np.around(passed*1./total-low_q, 5), np.around(up_q-passed*1./total,5)

def power(t_ref,t_data,zalpha=[.5,1,2,2.5]):
    # alpha=np.array([0.309,0.159,0.06681,0.0228,0.00620]))
    alpha = z_to_p(zalpha)
    #print(alpha)
    quantiles = np.quantile(t_ref,1-alpha,method='higher')
    total = len(t_data) 
    passed_list = [np.sum(t_data>=quantile) for quantile in quantiles]
    err_list = [ClopperPearson_interval(total, passed, level=0.68) for passed in passed_list]
    power_list = [passed/total for passed in passed_list]
    return p_to_z(alpha), power_list, err_list
    #return p_to_z(alpha), emp_pvalues_biased(t_data,quantiles)


def min_p(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    
    p_ref, p_data = return_pvalues(ref,data)
    return -np.min(p_ref,axis=1), -np.min(p_data,axis=1)
    #return -np.log(np.min(p_ref,axis=1)), -np.log(np.min(p_data,axis=1))

def fused_p(ref,data,T=1):
    # ref: n_refxd numpy array
    # data: n_dataxd numpy array
    
    p_ref, p_data = return_pvalues(ref,data)
    return -np.log(fusion(p_ref,-T)), -np.log(fusion(p_data,-T))


def avg_p(ref,data):
    # ref: n_refxd numpy array
    # data: n_dataxd numpy array

    p_ref, p_data = return_pvalues(ref,data)
    return -np.log(np.mean(p_ref,axis=1)), -np.log(np.mean(p_data,axis=1))
    #return -np.mean(p_ref,axis=1), -np.mean(p_data,axis=1)
    

def prod_p(ref,data):
    # ref: n_refxd numpy array
    # data: n_dataxd numpy array

    p_ref, p_data = return_pvalues(ref,data)
    return -np.sum(np.log(p_ref),axis=1), -np.sum(np.log(p_data),axis=1)
    #return -np.log(np.mean(p_ref,axis=1)), -np.log(np.mean(p_data,axis=1))

def fused_t(ref,data,T):
    fused_ref = fusion(ref,T)
    fused_data = fusion(data,T)

    return emp_pvalues(fused_ref,fused_data)

def emp_pvalue_biased(ref,t):
    # this definition is such that p=1/(N+1)!=0 if data is the most extreme value
    p = np.count_nonzero(ref > t) / (len(ref)) #+ (np.count_nonzero(ref > t)==0)*1. / (len(ref)) 
    return p

def emp_pvalue(ref,t):
    # this definition is such that p=1/(N+1)!=0 if data is the most extreme value
    p = (np.count_nonzero(ref > t)+1) / (len(ref)+1)
    return p

def emp_pvalues(ref,data):
    return np.array([emp_pvalue(ref,t) for t in data])

def emp_pvalues_biased(ref,data):
    return np.array([emp_pvalue_biased(ref,t) for t in data])

def p_to_z(pvals):
    return norm.ppf(1 - pvals)

def z_to_p(z):
    # sf=1-cdf (sometimes more accurate than cdf)
    return norm.sf(z)

def Zscore(ref,data):
    return p_to_z(emp_pvalues(ref,data))



def fusion(x,T):
    return T * logsumexp(1/T*x, axis=1, b=1/x.shape[1])

'''
#def bootstrap_pn(pn,rng=None):

    rnd = check_rng(rng)

    return rnd.choice(pn,size=len(pn))

#def bootstrap_pval(pn,t,rng=None):

    return emp_pvalue(bootstrap_pn(pn,rng=rng),t)


#def return_pvalues(ref,data,rng=None):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value by bootstrapping col (the value of t is removed first)
        p_ref[:,idx] = np.transpose([bootstrap_pval(np.delete(col,idx2),el,rng=rng) for idx2, el in enumerate(col)])

    # p-values under the laternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data
'''

def return_pvalues(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value with respect to the other ts
        p_ref[:,idx] = np.transpose([emp_pvalue(np.delete(col,idx2),el) for idx2, el in enumerate(col)])

    # p-values under the laternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data


def return_pvalues_biased(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value with respect to the other ts
        p_ref[:,idx] = np.transpose([emp_pvalue_biased(np.delete(col,idx2),el) for idx2, el in enumerate(col)])

    # p-values under the alternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues_biased(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data


def check_rng(rnd):
    if isinstance(rnd, np.random._generator.Generator):
        return rnd
    elif rnd is None:
        return np.random.default_rng(seed=rnd)
    elif isinstance(rnd, int):
        return np.random.default_rng(seed=rnd)
    else: 
        sys.exit("rnd must be None, an integer or an instance of np.random._generator.Generator")



def return_pvalues_2(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value with respect to the other ts
        p=emp_pvalues(col,col)
        p_ref[:,idx] = np.transpose(p)

    # p-values under the laternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data

def return_pvalues_2_biased(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value with respect to the other ts
        p=emp_pvalues_biased(col,col)
        p_ref[:,idx] = np.transpose(p)

    # p-values under the laternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues_biased(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data