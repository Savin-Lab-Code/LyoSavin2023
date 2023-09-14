import os, sys
project_root = os.path.abspath("")
base_dir = os.path.dirname(project_root)
sys.path.append(os.path.join(base_dir, 'core'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import add_noise_to_dataset

def show_datapoint(datapoint, ax, xlim=[-1,1], ylim=[-1,1]):
    """Show a single datapoint"""
    ax.scatter(datapoint[0], datapoint[1], s=10, marker='.', c='r')
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect('equal')
    ax.set(title='a single sample from $q(x_0)$')

def show_datapoint_in_context(data, idx, ax, xlim=[-1,1], ylim=[-1,1]):
    '''show data in the context of all other datapoints'''
    ax.scatter(data[:,0], data[:,1], s=10, marker='.', c='grey')
    show_datapoint(data[idx], ax, xlim, ylim)
    ax.set_aspect('equal')
    ax.set(title='the sample in the context of $q(x_0)$')

def plot_data_distribution(data_samples, lims=[-1,1], show_ticks=False):
    '''
    plot the data distribution
    '''
    fig, ax = plt.subplots(1, 1)
    ax.scatter(data_samples[:,0], data_samples[:,1], s=1, color='orange')
    ax.set_aspect('equal')
    ax.set(title='data distribution $p(x)$', xlim=lims, ylim=lims)
    fig.tight_layout()
    if show_ticks==False:
        from utils import remove_all_ticks_and_labels
        remove_all_ticks_and_labels(ax)
    
def plot_trimodal_data_distribution_separate_colors(data_samples, n_samples_per_manifold, lims=[-1.5, 1.5]):
    '''
    plot the trimodal data distribution (of N samples each) with separate colors for each manifold
    '''
    N = n_samples_per_manifold
    fig, ax = plt.subplots(1, 1)
    alpha = 0.3
    ax.scatter(*data_samples[:N].T, s=2, color='orange', alpha=alpha)
    ax.scatter(*data_samples[N:2*N].T, s=2, color='black', alpha=alpha)
    ax.scatter(*data_samples[2*N:].T, s=2, color='black', alpha=alpha)
    ax.set_aspect('equal')
    # ax.set(title='data distribution $p(x)$', xlim=lims, ylim=lims)
    fig.tight_layout()
    
    from utils import remove_all_ticks_and_labels
    remove_all_ticks_and_labels(ax)
    
    save=False
    if save:
        import os
        savedir = 'plots/neurips'
        figname = f'prior-trimodal.pdf'    
        plt.savefig(os.path.join(savedir, figname), transparent=True, bbox_inches='tight', dpi=400)
    
    return fig, ax

def plot_data_distribution_rescaled(data_samples, rescaled_data):
    '''
    plot the underlying data distribution and its rescaled version
    '''
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(data_samples[:,0], data_samples[:,1], s=5, marker='.')
    ax[1].scatter(rescaled_data[:,0], rescaled_data[:,1], s=5, marker='.')
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].set(title='data distribution $p(x)$')
    ax[1].set(title='rescaled')
    fig.tight_layout()
    
def plot_single_datapoint_in_context(rescaled_data):
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    show_datapoint(rescaled_data[0], ax[0], xlim=[-1,1], ylim=[-1,1])
    show_datapoint_in_context(rescaled_data, 0, ax[1])

def plot_progressively_noisy_data(rescaled_data, example_noise_levels):
    example_noisy_data = torch.empty((3,rescaled_data.shape[0],rescaled_data.shape[1]))
    for i in range(0,3):
        example_noisy_data_samples = add_noise_to_dataset(rescaled_data, example_noise_levels[i])
        example_noisy_data[i] = example_noisy_data_samples

    # plot the noisy distributions
    fig, ax = plt.subplots(1, len(example_noise_levels)+1, figsize=(4*len(example_noise_levels)+1, 4))
    ax[0].scatter(rescaled_data[:,0], rescaled_data[:,1], s=1)
    for i in range(1, 4):
        ax[i].scatter(example_noisy_data[i-1,:,0], example_noisy_data[i-1,:,1], s=1)

    lims=[-2,2]
    for i in range(0,4):
        ax[i].set_aspect('equal')
        ax[i].set(xlim=lims, ylim=lims)
        if i==0:
            ax[i].set(title=f'$p(x)$')
        else:
            ax[i].set(title=f'noise={example_noise_levels[i-1]}')
            
def compare_clean_and_noisy_data(train_batches, noise_level_title):
    fig, ax = plt.subplots(1,1)
    i=0
    for batch in train_batches:
        print('sample of noisy data\n', batch[0][0].numpy())  # 1 sample of noisy data
        print('sample of clean data\n', batch[1][0].numpy())  # 1 sample of clean data, at same index
        ax.scatter(batch[0][:,0], batch[0][:,1], label='noisy data')
        ax.scatter(batch[1][:,0], batch[1][:,1], alpha=0.7, label='clean data')
        ax.legend()
        i+=1
        if i==1:
            break
    ax.set_aspect('equal')
    ax.set(title=f'noisy data overlaid on clean data, sigma = {noise_level_title}')
    
def forward_process_with_plots(num_steps, 
                               dataset, 
                               schedule='sigmoid', 
                               start=1e-5, 
                               end=2e-2, 
                               plot=True, save_plot=False, model_name='model', 
                               device='cpu',
                               lims=[-3, 3]
                               ):
    
    from utils import make_beta_schedule
    betas = make_beta_schedule(schedule, n_timesteps=num_steps, start=start, end=end)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_prod_log = torch.log(1 - alphas_prod)
    one_minus_alphas_prod_sqrt = torch.sqrt(1 - alphas_prod)
    one_minus_alphas_prod = 1 - alphas_bar_sqrt

    def q_x(x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0) * 0.2
        alphas_t = extract(alphas_bar_sqrt, t, x_0)
        alphas_1_m_s_t = extract(one_minus_alphas_prod_sqrt, t, x_0)
        alphas_1_m_t = extract(one_minus_alphas_prod, t, x_0)
        # return (alphas_t * x_0 + extract(1-alphas_prod, t, x_0) * 0.5 + alphas_1_m_t * noise)
        return (alphas_t * x_0 + 0.5* alphas_1_m_t + alphas_1_m_s_t * noise)

    if plot:
        # plot the anti-diffusion process
        fig, ax = plt.subplots(1, 11, figsize=(28, 3))
        for i in range(10):
            q_i = q_x(dataset, torch.tensor([i * 10]))
            q_i = torch.clamp(q_i, min=0)
            ax[i].scatter(q_i[:, 0], q_i[:, 1], color='white', edgecolor='peru', s=1)
            ax[i].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
        q_10 = q_x(dataset, torch.tensor([99]))
        q_10 = torch.clamp(q_10, min=0)
        ax[10].scatter(*q_10.T, color='white', edgecolor='peru', s=1)
        ax[10].set_title('$q(\mathbf{x}_{99})$')
        for i in range(11):
            # ax[i].set_axis_off()
            ax[i].set(xlim=lims, ylim=lims)
            ax[i].set_aspect('equal')
            
        if save_plot==True:
            plot_savedir = 'plots'
            figname = f'forward-process-{model_name}.png'
            plt.savefig(os.path.join(plot_savedir, figname))
            
    params = (betas, alphas, alphas_prod, alphas_prod_p, alphas_bar_sqrt, one_minus_alphas_prod_log, one_minus_alphas_prod_sqrt)
    return params


def draw_constraint_line(ax, v, constraint_sigma, lim, show_stdev=True):
    ls = np.linspace(-lim, lim, 50)
    v1 = v[1][0] / v[0][0]
    v0 = v[0][0] / v[0][0]
    
    ax.plot(ls * v0, ls * v1, c='green')
    if show_stdev:
        below = ls * v1 - 2 * constraint_sigma / np.cos(np.arctan(v1/v0))
        above = ls * v1 + 2 * constraint_sigma / np.cos(np.arctan(v1/v0))
        ax.fill_between(ls * v0, below, above, color='green', alpha=0.3)
        
def save_fig(fig, figname, savedir='plots', format='pdf'):
    import os
    savedir = os.path.join(base_dir, 'core', savedir)
    if format=='pdf':
        savedir = os.path.join(savedir, f'{figname}.pdf')
    else:
        savedir = os.path.join(savedir, figname)
    fig.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)