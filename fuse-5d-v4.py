import numpy as np
import os, time, json, csv
from sklearn.utils import shuffle


from falkon_utils import compute_t, trainer, get_logflk_config

from utils import BuildSample_DY, normalize

output_path = './output_nest_5d/'

tests = ['t','avg-f','avg-exp-f'] # ['t','avg-f','avg-exp-f']

# multiple testing definition
#flk_sigmas =  [0.31,0.76,1.77,3.15,4.21]#1,10,50,90,99 percentiles
flk_sigmas =  [0.32,0.77,1.79,3.18,4.23]#1,10,50,90,99 percentiles
M          = [10000,10000,10000,10000,10000]
lam        = [1e-6,1e-6,1e-6,1e-6,1e-6]
Ntoys      = 1000

# problem definition
sig = 'EFT' # Z200, Z300, EFT, EFT2, EFT5
N_ref      = 100000
N_Bkg      = 20000
N_Sig      = 0
weight    = N_Bkg*1./N_ref
dim = 6
tr_dim = 5

# mean and std of reference data compute from the entire dataset - used to normalize
#mean_ref = [3.05212537e+00, 4.59876795e-04, 3.66978211e-04, 4.67488118e+01, 3.60544255e+01]
#std_ref = [0.52342366, 1.24487457, 1.25051629, 15.86122412, 8.95046637]
mean_ref = [3.05212537e+00,4.59876795e-04,3.66978211e-04,4.67488118e+01,3.60544255e+01]
std_ref = [0.52342366,1.24487457,1.25051629,15.86122412,8.95046637]

cut_mll = 60
cut_pt = 20
cut_eta = 2.4

reference_path = '/data/marcol/HEPDATA/DILEPTON/DiLepton_SM/'
#reference_path = '/data/marcol/HEPDATA/DILEPTON/DiLepton_SM_new/'
if sig=='EFT':
    #reference_path = '/data/marcol/HEPDATA/DILEPTON/DiLepton_SM_new/'
    #data_path = '/data/marcol/HEPDATA/DILEPTON/DiLepton_EFT01_new/'
    data_path = '/data/marcol/HEPDATA/DILEPTON/DiLepton_EFT06/'
if sig=='EFT2':
    data_path = '/data/marcol/HEPDATA/DILEPTON/DiLepton_EFT06_2/'
if sig=='EFT5':
    data_path = '/data/marcol/HEPDATA/DILEPTON/DiLepton_EFT06_5/'
elif sig=='Z300':
    data_path = '/data/marcol/HEPDATA/DILEPTON/DiLepton_Zprime300/'
elif sig=='Z200':
    data_path = '/data/marcol/HEPDATA/DILEPTON/DiLepton_Zprime200/'

#['delta_phi', 'eta1', 'eta2', 'mll', 'pt1', 'pt2'] how features are returned when loaded

if N_Sig: 
    filename = f'Ntoys{Ntoys}_NR{N_ref}_NB{N_Bkg}_NS{N_Sig}_'+sig+f'_cut{cut_mll}'
else: filename = f'Ntoys{Ntoys}_NR{N_ref}_NB{N_Bkg}_cut{cut_mll}_null'
os.makedirs(output_path+filename, exist_ok = True)

rng = np.random.default_rng(seed=time.time_ns())
seeds = rng.integers(0, high=1e8, size=Ntoys)

with open(output_path+filename+"/seeds.txt", 'w') as f:
    for line in seeds.tolist():
        f.write(f"{line}\n")

#seeds = np.loadtxt("/home/marcol/nplm-fuse/output_5d/Ntoys1000_NR100000_NB20000_cut60_null/seeds.txt",dtype=int)

with open(output_path+filename+"/flk_model.txt", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([M,lam,flk_sigmas])


# initialize arrays with test statistics
if 't' in tests: t_array = np.zeros(shape=(Ntoys,len(flk_sigmas)))
if 'avg-f' in tests: t_avg_array = np.zeros(shape=(Ntoys))
if 'avg-exp-f' in tests: t_avg_exp_array = np.zeros(shape=(Ntoys))

for i in range(Ntoys):
    seed = seeds[i]
    rng = np.random.default_rng(seed=seed)

    # randomized number of events
    N_Bkg_Pois  = rng.poisson(lam=N_Bkg, size=1)[0]

    if N_Sig: N_Sig_Pois = rng.poisson(lam=N_Sig, size=1)[0]
    else: N_Sig_Pois = N_Sig
    Ntot = N_Bkg_Pois+N_Sig_Pois+N_ref

    # initialize dataset
    dataset = np.zeros(shape=(Ntot,dim), dtype=np.float64)
    if 'EFT' in sig and N_Sig != 0:
        # fill with ref
        dataset[:N_ref,:] = BuildSample_DY(N_Events=N_ref, INPUT_PATH=reference_path, rng=rng) #['delta_phi', 'eta1', 'eta2', 'mll', 'pt1', 'pt2']
        # fill with bkg and signal
        dataset[N_ref:,:] = BuildSample_DY(N_Events=N_Bkg_Pois+N_Sig_Pois, INPUT_PATH=data_path, rng=rng) 
    else:
        # fill with ref and bkg
        dataset[:N_ref+N_Bkg_Pois,:] = BuildSample_DY(N_Events=N_ref+N_Bkg_Pois, INPUT_PATH=reference_path, rng=rng) 
        # fill with signal
        if N_Sig: dataset[N_ref+N_Bkg_Pois:,:] = BuildSample_DY(N_Events=N_Sig_Pois, INPUT_PATH=data_path, rng=rng) 
    # initialize labes
    target = np.zeros(shape=(Ntot,1), dtype=np.float64)
    # fill with data labels
    target[N_ref:,:] = np.ones((N_Bkg_Pois+N_Sig_Pois,1), dtype=np.float64)
    
    weights = np.ones_like(dataset)
    weights[:N_ref] = np.full(shape=(N_ref,1),fill_value=weight)

    mask_idx = np.where((dataset[:, 4] <= cut_pt) | (dataset[:, 5] <= cut_pt) | (np.abs(dataset[:, 1]) > cut_eta) | (np.abs(dataset[:, 2]) > cut_eta) | (dataset[:, 3] <= cut_mll))[0]
    dataset = np.delete(dataset, mask_idx, axis=0)
    target = np.delete(target, mask_idx, axis=0)
    weights = np.delete(weights, mask_idx, axis=0)

    cut_Ntot = len(dataset)

    dataset = normalize(dataset[:,[0,1,2,4,5]],mean_ref,std_ref)


    dataset, target, weights = shuffle(dataset, target, weights, random_state=seed)

    if 't' in tests: t_toy = np.zeros(shape=len(flk_sigmas))
    if 'avg-f' in tests: avg_preds= np.zeros(shape=(cut_Ntot,1))
    if 'avg-exp-f' in tests: logsumexp_preds= np.zeros(shape=(cut_Ntot,1))

    for idx, flk_sigma in enumerate(flk_sigmas):
        flk_config = get_logflk_config(M[idx],flk_sigma,[lam[idx]],weight=weight,iter=[1000000],seed=seed,cpu=False) # seed for center is re-set inside learn_t
        st_time = time.time()
        preds = trainer(dataset,target,flk_config)
        dt = round(time.time()-st_time,2)
        print(f"training time = {dt}")
        if 't' in tests: t_toy[idx] = compute_t(preds,target,weight)
        if 'avg-f' in tests: avg_preds += preds
        if 'avg-exp-f' in tests: logsumexp_preds += np.exp(preds)

    if 'avg-f' in tests: 
        avg_preds = avg_preds/len(flk_sigmas)
        t_avg = compute_t(avg_preds,target,weight)
    if 'avg-exp-f' in tests: 
        logsumexp_preds = np.log(logsumexp_preds)-np.log(len(flk_sigmas)) #np.log(np.mean(np.exp(array_preds),axis=1))
        t_avg_exp = compute_t(logsumexp_preds,target,weight)
        
    print(f"Toy {i}:")
    if 't' in tests: 
        print("t = "+np.array2string(t_toy,separator=','))
        t_array[i] = t_toy
    if 'avg-f' in tests: 
        print(f"t-avg = {t_avg}")
        t_avg_array[i] = t_avg
    if 'avg-exp-f' in tests: 
        print(f"t-avg-exp = {t_avg_exp}")
        t_avg_exp_array[i] = t_avg_exp
        
        #st_time = time.time()
        #t = learn_t(dataset, target,  weight=weight, seed=seed, flk_config=flk_config)
        #dt = round(time.time()-st_time,2)
        #print(f"Toy: {i} --------- (M,lambda,sigma)={M[idx],lam[idx],flk_sigma}: t={t}, time={dt}.")
        #t_array[i,idx] = t


if 't' in tests: np.save(output_path+filename+"/t_array.npy", t_array)
if 'avg-f' in tests: np.save(output_path+filename+"/t_avg_array.npy", t_avg_array)
if 'avg-exp-f' in tests: np.save(output_path+filename+"/t_avg_exp_array.npy", t_avg_exp_array)