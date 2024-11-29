import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')

from tests import return_pvalues
import numpy as np

from tests import *




def plot_correlations(ref_folder, data_folders, tests_idx, labels_tests, output_folder):

    color=['#67a9cf','#1c9099','#016c59', 'black', '#f6eff7','#bdc9e1',]
    k=0
    corr_list = []
    for data_folder in [ref_folder] + data_folders:
        print(data_folder)
        fig  = plt.figure(figsize=(16, 12.5))
        fig.patch.set_facecolor('white')
        ref = np.load(ref_folder+"/t_array.npy")[:,tests_idx]
        data = np.load(data_folder+"/t_array.npy")[:,tests_idx]
        for i in range(len(labels_tests)):
            for j in range(len(labels_tests)):
                if j<=i: continue
                ax= fig.add_axes([0.07+i*0.18, 0.07+(j-1)*0.18, 0.18, 0.18])
                if j>i:
                    p_ref_i, p_data_i = return_pvalues(ref[:,i].reshape((-1,1)),data[:,i].reshape((-1,1)))
                    p_ref_j, p_data_j = return_pvalues(ref[:,j].reshape((-1,1)),data[:,j].reshape((-1,1)))
                    corr=np.corrcoef(p_data_i[:400,0], p_data_j[:400,0])[0][1]
                    corr_list.append(corr)
                    plt.scatter(p_data_i[:400,0], p_data_j[:400,0], color=color[k], s=12, marker='o', 
                                label=r'$\rho=$%s'%(str(np.around(corr, 2)))#labels[k]
                            )
                    font = font_manager.FontProperties(family='serif', size=24)
                    plt.legend(prop=font, ncol=1, loc='upper left', scatterpoints=1, 
                            labelspacing=0.1, handletextpad=0.4, markerscale=1.3, frameon=True)        
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.1, 1.1)
                if not i:
                    plt.ylabel(labels_tests[j],fontsize=24, fontname='serif')
                    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1],[0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24, fontname='serif')
                else: 
                    ax.tick_params(axis='y', which='both', labelleft=False)
                if i<=(j-2):
                    ax.tick_params(axis='x', which='both', labelbottom=False)
                else:
                    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1],[0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24, fontname='serif')
                if j==len(labels_tests)-1:
                    plt.title(labels_tests[i],fontsize=24, fontname='serif')
        plt.savefig(output_folder+'/scatter_grid_5D_%i.pdf'%(k))
        k+=1
        plt.show()
        print('avg pair-wise correlation: ', np.sum(corr_list)/len(corr_list))



def plot_power(files_dict, tests, flk_sigmas, labels_plot, output_dir):


    colors = ['#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59', 'black']    
    font = font_manager.FontProperties(family='serif', size=26) 

    xlabels_tests= [ r'$\sigma=%s$'%(str(flk_sigma)) for flk_sigma in flk_sigmas]

    #Z_alpha_ini = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0,]#[0,0.2, 0.5, 1, 1.2, 1.5, 1.7, 2, 2.5, 3.5, 4.5]
    #Z_alpha_ini = np.array(Z_alpha_ini)

    Z_alpha_ini = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4, 5]#[0,0.2, 0.5, 1, 1.2, 1.5, 1.7, 2, 2.5, 3.5, 4.5]
    Z_alpha_ini = np.array(Z_alpha_ini) 

    lw=2
    ms=5
    ls='-'
    zorder=2
    plot_legend=True
    plt_aggreg_min = True

    ref = np.load(files_dict['ref'])[:,tests]
    
    for key in files_dict.keys():
        data = np.load(files_dict[key])[:,tests]

        fig  = plt.figure(figsize=(7*(1+0.4*(plot_legend==1)),7))
        fig.patch.set_facecolor('white')
        ax1 = fig.add_axes([0.2*1/(1+0.4*(plot_legend==1)), 0.15+0.01*(plot_legend==1), 0.7/(1+0.4*(plot_legend==1)), 0.7+0.01*(plot_legend==1)])

        for i in range(ref.shape[1]):
            t0  = ref[:, i]
            t = data[:, i]
            mask0 = (~np.isnan(t0))*(~np.isinf(t0))
            mask  = (~np.isnan(t))*(~np.isinf(t))
            t0, t = t0[mask0], t[mask]
            eff_ref = power(t0,t0,zalpha=Z_alpha_ini)[2]#efficiency_root(t0, thr)
            eff_data = power(t0,t,zalpha=Z_alpha_ini)[2]#efficiency_root(t, thr)
            alpha, alpha_edw, alpha_eup = np.array([p[0] for p in eff_ref]), np.array([p[1] for p in eff_ref]), np.array([p[2] for p in eff_ref])
            power_val, power_edw, power_eup = np.array([p[0] for p in eff_data]), np.array([p[1] for p in eff_data]), np.array([p[2] for p in eff_data])
            Z_alpha   = np.array([p_to_z(alpha[i]) for i in range(len(alpha))])
            Z_alpha_eup = np.array([p_to_z(alpha[i]-alpha_edw[i])-p_to_z(alpha[i]) for i in range(len(alpha))])
            Z_alpha_edw = np.array([p_to_z(alpha[i])-p_to_z(alpha[i]+alpha_eup[i]) for i in range(len(alpha))]) 

            x = Z_alpha[~np.isinf(Z_alpha)]
            y = power_val[~np.isinf(Z_alpha)]
            y_dw, y_up = power_edw[~np.isinf(Z_alpha)], power_eup[~np.isinf(Z_alpha)]
            x_dw, x_up = Z_alpha_edw[~np.isinf(Z_alpha)], Z_alpha_eup[~np.isinf(Z_alpha)]
            ax1.errorbar(x, y, 
                            yerr=[y_dw, y_up], 
                            xerr=[x_dw, x_up], 
                            marker='o', label=xlabels_tests[i], color=colors[i],#r'NPLM M=8530, $\lambda=1^{-4}$', color='#1c9099', 
                            lw=lw, ms=ms, ls=ls, elinewidth=1, zorder=zorder)


            i+=1
        if plt_aggreg_min:
            pmin_ref, pmin_data =  min_p(ref,data)
            t0  = pmin_ref[:]
            t = pmin_data[:]
            mask0 = (~np.isnan(t0))*(~np.isinf(t0))
            mask  = (~np.isnan(t))*(~np.isinf(t))
            t0, t = t0[mask0], t[mask]
            eff_ref = power(t0,t0,zalpha=Z_alpha_ini)[2]#efficiency_root(t0, thr)
            eff_data = power(t0,t,zalpha=Z_alpha_ini)[2]#efficiency_root(t, thr)
            alpha, alpha_edw, alpha_eup = np.array([p[0] for p in eff_ref]), np.array([p[1] for p in eff_ref]), np.array([p[2] for p in eff_ref])
            power_val, power_edw, power_eup = np.array([p[0] for p in eff_data]), np.array([p[1] for p in eff_data]), np.array([p[2] for p in eff_data])
            Z_alpha   = np.array([p_to_z(alpha[i]) for i in range(len(alpha))])
            Z_alpha_eup = np.array([p_to_z(alpha[i]-alpha_edw[i])-p_to_z(alpha[i]) for i in range(len(alpha))])
            Z_alpha_edw = np.array([p_to_z(alpha[i])-p_to_z(alpha[i]+alpha_eup[i]) for i in range(len(alpha))]) 

            x = Z_alpha[~np.isinf(Z_alpha)]
            y = power_val[~np.isinf(Z_alpha)]
            y_dw, y_up = power_edw[~np.isinf(Z_alpha)], power_eup[~np.isinf(Z_alpha)]
            x_dw, x_up = Z_alpha_edw[~np.isinf(Z_alpha)], Z_alpha_eup[~np.isinf(Z_alpha)]
            ax1.errorbar(x, y, 
                            yerr=[y_dw, y_up], 
                            xerr=[x_dw, x_up], 
                            marker='o', label=r'min-$p$', color='black',#r'NPLM M=8530, $\lambda=1^{-4}$', color='#1c9099', 
                            lw=lw, ms=ms, ls=ls, elinewidth=1, zorder=zorder)
        ##########
        ax1.fill_between(Z_alpha_ini[~np.isinf(Z_alpha_ini)], 
                        y1=np.zeros_like(Z_alpha_ini[~np.isinf(Z_alpha_ini)]), 
                        y2=1-norm.cdf(Z_alpha_ini[~np.isinf(Z_alpha_ini)]), 
                            color='lightgrey',
                        )
        #########
        plt.yticks(fontsize=22, fontname='serif')
        plt.xticks([0, 1, 2, 3,4], [0, 1, 2, 3,4], fontsize=22, fontname='serif')
        plt.ylim(0,1)
        plt.xlim(-0.3,4.3)
        ax1.tick_params(axis='y')
        ax1.tick_params(axis='x')
        ax1.set_ylabel(r'${\rm P}(Z>Z_{\alpha})$', fontsize=24, fontname='serif')
        ax1.set_xlabel(r'$Z_{\alpha}$',      fontsize=24, fontname='serif')
        plt.grid(axis='y', lw=0.25, ls=':')
        ax1.set_title(labels_plot[key], fontsize=22, fontname='serif')
        if plot_legend:
            ax1.legend(prop=font, loc='upper left', bbox_to_anchor=(1., 0.9),
                    frameon=False, ncol=1, numpoints=1)
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(1) 
        if plot_legend and plt_aggreg_min:
            fig.savefig(output_dir+'/power_%s_minp.pdf'%(key))
        elif plot_legend:
            fig.savefig(output_dir+'/power_%s.pdf'%(key))
        elif plt_aggreg_min:
            fig.savefig(output_dir+'/power_%s_minp.pdf'%(key))
        else:
            fig.savefig(output_dir+'/power_noleg_%s.pdf'%(key))
        plt.show()
        plt.close()


# Class to duplicate output to both console and file
class Tee:
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, "w")
    
    def write(self, message):
        self.console.write(message)  # Write to the console
        self.file.write(message)    # Write to the file
    
    def flush(self):
        self.console.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()