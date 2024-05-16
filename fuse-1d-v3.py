import numpy as np
import os, time, csv

from falkon_utils import compute_t, trainer, get_logflk_config

from utils import nonres_sig

output_path = './output_1d/'

tests = ['t','avg-f','avg-exp-f']

# multiple testing definition
flk_sigmas = [0.1, 0.3, 0.7, 1.4, 2.4, 3.0] #10,25,50,75,90,95 percentiles
M          = [5000,5000,5000,5000,5000,5000]
lam        = [1e-10,1e-10,1e-10,1e-10,1e-10,1e-10]
Ntoys      = 200

# problem definition
resonant   = True
N_ref      = 200000
N_Bkg      = 2000
N_Sig      = 13
weight     = N_Bkg*1./N_ref
Sig_loc    = 4
Sig_std    = 0.64

#if N_Sig: filename = f'Ntoys{Ntoys}_NR{N_ref}_NB{N_Bkg}_NS{N_Sig}_loc{Sig_loc}_std{Sig_std}'
#else: filename = f'Ntoys{Ntoys}_NR{N_ref}_NB{N_Bkg}_null'
if N_Sig: 
    if resonant: filename = f'Ntoys{Ntoys}_NR{N_ref}_NB{N_Bkg}_NS{N_Sig}_loc{Sig_loc}_std{Sig_std}'
    else: filename = f'Ntoys{Ntoys}_NR{N_ref}_NB{N_Bkg}_NS{N_Sig}_nonres'
else: filename = f'Ntoys{Ntoys}_NR{N_ref}_NB{N_Bkg}_null'
os.makedirs(output_path+filename, exist_ok = True)

rng = np.random.default_rng(seed=time.time_ns())
seeds = rng.integers(0, high=1e8, size=Ntoys)
#seeds_nonres = np.loadtxt("/home/marcol/nplm-fuse/output_1d/Ntoys300_NR200000_NB2000_NS90_nonres/seeds.txt",dtype=int)
#seeds = seeds_nonres[-200:]

with open(output_path+filename+"/seeds.txt", 'w') as f:
    for line in seeds.tolist():
        f.write(f"{line}\n")

with open(output_path+filename+"/flk_model.txt", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([M,lam,flk_sigmas])

# initialize arrays with test statistics
if 't' in tests: t_array = np.zeros(shape=(Ntoys,len(flk_sigmas)))
if 'avg-f' in tests: t_avg_array = np.zeros(shape=(Ntoys))
if 'avg-exp-f' in tests: t_avg_exp_array = np.zeros(shape=(Ntoys))


for i in range(Ntoys):
    print(f"Processing toy: {i}")
    seed = seeds[i]
    rng = np.random.default_rng(seed=seed)
    # data
    N_Bkg_Pois  = rng.poisson(lam=N_Bkg, size=1)[0]

    if N_Sig: N_Sig_Pois = rng.poisson(lam=N_Sig, size=1)[0]
    else: N_Sig_Pois = N_Sig
    Ntot = N_Bkg_Pois+N_Sig_Pois+N_ref

    dataset = np.zeros(shape=(Ntot,1))

    dataset[:N_Bkg_Pois] = rng.exponential(scale=1, size=(N_Bkg_Pois, 1))
    #if N_Sig:
    #    dataset[N_Bkg_Pois:N_Bkg_Pois+N_Sig_Pois]  = rng.normal(loc=Sig_loc, scale=Sig_std, size=(N_Sig_Pois,1))
    if N_Sig:
            if resonant: dataset[N_Bkg_Pois:N_Bkg_Pois+N_Sig_Pois]  = rng.normal(loc=Sig_loc, scale=Sig_std, size=(N_Sig_Pois,1))
            else: dataset[N_Bkg_Pois:N_Bkg_Pois+N_Sig_Pois]  = nonres_sig(N_Sig_Pois,seed).reshape(N_Sig_Pois,1)
    dataset[N_Bkg_Pois+N_Sig_Pois:]  = rng.exponential(scale=1, size=(N_ref, 1))

    # target
    target   = np.zeros_like(dataset)
    target[:N_Bkg_Pois+N_Sig_Pois] = np.ones(shape=(N_Bkg_Pois+N_Sig_Pois,1))
    
    weights = np.ones_like(dataset)
    weights[N_Bkg_Pois+N_Sig_Pois:] = np.full(shape=(N_ref,1),fill_value=weight)


    if 't' in tests: t_toy = np.zeros(shape=len(flk_sigmas))
    if 'avg-f' in tests: avg_preds= np.zeros(shape=(Ntot,1))
    if 'avg-exp-f' in tests: logsumexp_preds= np.zeros(shape=(Ntot,1))
    
    for idx, flk_sigma in enumerate(flk_sigmas):
        print(f"Sigma: {flk_sigma}")
        flk_config = get_logflk_config(M[idx],flk_sigma,[lam[idx]],weight=weight,iter=[1000000],seed=seed,cpu=False)
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


if 't' in tests: np.save(output_path+filename+"/t_array.npy", t_array)
if 'avg-f' in tests: np.save(output_path+filename+"/t_avg_array.npy", t_avg_array)
if 'avg-exp-f' in tests: np.save(output_path+filename+"/t_avg_exp_array.npy", t_avg_exp_array)
