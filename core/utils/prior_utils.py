import os, sys
project_root = os.path.abspath("")
if project_root[-12:] == 'LyoSavin2023':
    base_dir = project_root
else:
    base_dir = os.path.dirname(project_root)
sys.path.append(os.path.join(base_dir, 'core'))

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange
device = 'cpu'

def forward_process(num_steps, device, schedule='sigmoid', start=1e-5, end=2e-2):
    '''calculate the forward process for the given noise schedule'''
    from utils import make_beta_schedule
    betas = make_beta_schedule(schedule=schedule, n_timesteps=num_steps, start=start, end=end, device=device)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1], device=device).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_prod_log = torch.log(1 - alphas_prod)
    one_minus_alphas_prod_sqrt = torch.sqrt(1 - alphas_prod)
    return betas, alphas, alphas_prod, alphas_prod_p, alphas_bar_sqrt, one_minus_alphas_prod_log, one_minus_alphas_prod_sqrt


def forward_diffusion_sample(x_0, t, alphas, device='cpu'):
    '''
    takes an image and a timestep as input and returns a noisy version of it
    '''
    from utils import get_index_from_list
    noise = torch.randn_like(x_0)
    
    alphas_cumprod = torch.cumprod(alphas, axis=0)  # \prod{\alpha_t}
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # \sqrt{\prod{\alpha_t}}
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # mean + variance at one value of `t`, rescaled by alphas. 
    # The formula is: q(x_t|x_0) = \sqrt{\prod{\alpha_s}} * x_0 + \sqrt{1-\prod{\alpha_s}} * z
    mean_rescaled = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
    noise_rescaled = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    
    # x_noisy, noise = mean_rescaled + noise_rescaled, noise.to(device)
    x_noisy, noise = mean_rescaled + noise_rescaled, noise_rescaled.to(device)
    return x_noisy, noise

def diffused_sample(num_steps, t, x_0, schedule='sigmoid', start=1e-5, end=2e-2):
    '''returns a diffused sample of x_0 at time t'''
    from utils import extract
    _, _, alphas_prod, _, alphas_bar_sqrt, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device, schedule, start=start, end=end)
    batch_size = x_0.shape[0]
    
    t = torch.tensor([t], dtype=torch.long)

    scale_factor_t = extract(alphas_bar_sqrt, t, x_0)
    
    # Generate z
    z = torch.randn_like(x_0, device=device)

    # Fixed sigma
    sigma_t = extract(one_minus_alphas_prod_sqrt, t, x_0)
    
    x_t = scale_factor_t * x_0 + sigma_t * z
    return x_t


# ----------------- sampling from the vanilla reverse process ---------------- #
def plot_reverse_samples_10_steps(model, sample_size, embedding_dims, num_steps, normalized_beta_schedule=False):    
    # num_dim_data = 2
    num_dim_data = embedding_dims

    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device)

    sample_size = int(6e2)
    lims = [-2,2]
    x_seq = p_sample_loop(model, (sample_size, num_dim_data), num_steps, device, normalized_beta_schedule=normalized_beta_schedule)

    fig, axes = plt.subplots(1, 10, figsize=(12, 2), sharey=True)
    for i in range(1, 11):
        cur_x = x_seq[i * 10].detach().cpu()
        
        ax = axes[i-1]
        ax.scatter(cur_x[:, 0], cur_x[:, 1], s=1, color='grey')
        ax.set_aspect('equal')
        ax.set(xlim=lims, ylim=lims)
    return alphas, betas, one_minus_alphas_prod_sqrt, x_seq, fig, axes


def prior_sampling(num_steps, model, num_dim_data, dim1=0, dim2=1, sample_size=int(1e3), plot=True):
    '''
    performs the vanilla reverse process and returns the sequence of x_t's as the t varies.  
    '''
    from prior_utils import p_sample_loop
    
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device)
    x_seq = p_sample_loop(model, (sample_size, num_dim_data), num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device)

    if plot:
        lims = [-1.5, 1.5]
        fig, axes = plt.subplots(1, 5, figsize=(10, 3), sharey=True)
        
        for i in range(1, 6):
            cur_x = x_seq[i * 20].detach().cpu()
            ax = axes[i-1]
            
            # x-y plane
            ax.scatter(cur_x[:, dim1], cur_x[:, dim2],color='white',edgecolor='indianred', s=0.5);
            
            # y-z plane
            ax.set_aspect('equal')
            ax.set(xlim=lims, ylim=lims)
            ax.set_yticks([-1, 0, 1])
            
            fig.suptitle(f'x=dim {dim1}, y=dim {dim2}')
        fig.tight_layout()
    else: 
        x_seq = torch.stack(x_seq)
        return x_seq


# ----------------- generate samples from the reverse process ---------------- #
@torch.no_grad()
def p_sample(model, x, t, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device, normalized_beta_schedule=False):
    '''one step of the reverse process. takes a noisy data sample x_{t-1} and returns a less noisy sample x_t.'''
    from utils import extract
    
    t = torch.tensor([t], device=device)
    
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_prod_sqrt, t, x))
    
    # Model output
    if normalized_beta_schedule:
        T = t.repeat(x.shape[0], 1) / n_steps
    else:
        T = t.repeat(x.shape[0], 1)
    eps_theta = model(x, T)
    
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    
    # Generate z
    z = torch.randn_like(x, device=device)
    
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return sample

@torch.no_grad()
def p_sample_loop(model, shape, n_steps, device='cpu', init_x=None, normalized_beta_schedule=False, schedule='sigmoid'):
    '''takes a model and returns the sequence of x_t's during the reverse process
    '''
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(n_steps, device, schedule)
    
    if init_x == None:
        cur_x = torch.randn(shape, device=device)
    else:
        cur_x = init_x
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, n_steps, alphas,betas,one_minus_alphas_prod_sqrt, device, normalized_beta_schedule)
        x_seq.append(cur_x)
    x_seq = torch.stack(x_seq, dim=0).detach()
    return x_seq



def rad_ad_various_temps(data_2d, model, alphas, betas, one_minus_alphas_prod_sqrt, temps = [10, 30, 50, 100], lims=[-2, 2]):
    '''plot the distribution of datapoints at different temperatures, taken over one cycle of the oscillation'''
    fig, ax = plt.subplots(1, 4, figsize=(11,5))

    x_init = data_2d
    idx = 0
    for t in temps:
        for j in range(6):
            x_rad, _ = p_rev_loop(model, x_init, x_init.shape, t, alphas, betas, one_minus_alphas_prod_sqrt, device)
            x_rad = torch.stack(x_rad).detach().cpu()
            ax[idx].scatter(x_rad[-1,:,0], x_rad[-1,:,1], color='orange', s=5)
        
        # plot the underlying distribution
        ax[idx].scatter(data_2d[:,0], data_2d[:,1], color='white', edgecolor='grey', s=5, label="initial points")
        
        # plot details
        ax[idx].set_aspect('equal')
        ax[idx].set(xlim=lims, ylim=lims)
        ax[idx].legend()
        ax[idx].set_title(f't={t}')
        
        idx += 1

    # add figure details
    fig.suptitle(f'Distribution of datapoints during RAD process\n at different temperatures')
    fig.tight_layout()


# -------------------------- neural forward process -------------------------- #
@torch.no_grad()
def p_sample_rev(model, x, t, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device, normalized_beta_schedule=False):
    """
    returns one step of the neural forward process. takes a noisy data sample x_{t-1} and returns a more noisy x_t.
    """
    from utils import extract
    
    t = torch.tensor([t], device=device)
    
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_prod_sqrt, t, x))
    
    # Model output
    if normalized_beta_schedule:
        T = t.repeat(x.shape[0], 1) / n_steps
    else:
        T = t.repeat(x.shape[0], 1)
    eps_theta = model(x, T)
    
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    
    # Generate z
    z = torch.randn_like(x, device=device)
    
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return sample, mean

@torch.no_grad()
def p_rev_loop(model, x_0, shape, n_steps, device, normalized_beta_schedule=False, schedule='sigmoid'):
    """
    for each initial datapoint, this function returns the datapoints corresponding to every step of the neural forward process
    """
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(n_steps, device, schedule)
    
    cur_x = x_0
    x_seq = [cur_x]
    mean_seq = [cur_x]

    for i in range(n_steps):
        cur_x, cur_mean = p_sample_rev(model, cur_x, i, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device, normalized_beta_schedule)
        x_seq.append(cur_x)
        mean_seq.append(cur_mean)
    return x_seq, mean_seq



# ------- alternating between the neural forward and reverse processes ------- #
def neural_fwd_rev_cycle(x_fwd_i, model, num_steps, normalized_beta_schedule=False, schedule='sigmoid'):
    '''
    undergoes one cycle of the oscillation.
    diffuses the datapoints using the neural forward process, and then anti-diffuses using the neural reverse process.
    '''
    
    if type(x_fwd_i) == np.ndarray:
        x_fwd_i = torch.tensor(x_fwd_i, dtype=torch.float)
    else: 
        x_fwd_i = x_fwd_i
    
    # neural forward process
    x_fwd_seq = p_rev_loop(model, x_fwd_i, x_fwd_i.shape, num_steps, device, normalized_beta_schedule, schedule)[0]
    x_fwd_seq = torch.stack(x_fwd_seq).detach().cpu()
    x_fwd_f = x_fwd_seq[-1, :, :]
    
    # neural reverse process
    x_rev_i = x_fwd_f
    x_rev_seq = p_sample_loop(model, x_rev_i.shape, num_steps, device, x_rev_i, normalized_beta_schedule, schedule)
    x_rev_f = x_rev_seq[-1, :, :]

    # return the final datapoint after one cycle of the oscillation, and the sequences of datapoints during the forward and reverse processes
    return x_rev_f, x_fwd_seq, x_rev_seq


def sequential_prior_sampler(model, init_x, num_cycles, num_steps=100, disable_tqdm=False, normalized_beta_schedule=False, schedule='sigmoid'):
    if type(init_x) == np.ndarray:
        init_x = torch.tensor(init_x, dtype=torch.float)
    else: 
        init_x = init_x

    num_cycles = int(num_cycles)
    
    # burn the first sample
    x, x_fwd, x_rev = neural_fwd_rev_cycle(init_x, model, num_steps, normalized_beta_schedule, schedule)
    
    seq_x = []
    seq_fwd_x = []
    seq_rev_x = []
    for i in trange(num_cycles, disable=disable_tqdm):
        x, x_fwd, x_rev = neural_fwd_rev_cycle(x, model, num_steps, normalized_beta_schedule, schedule)
        seq_x.append(x)
        seq_fwd_x.append(x_fwd)
        seq_rev_x.append(x_rev)
    seq_x = torch.stack(seq_x).detach().numpy().reshape(num_cycles, -1)
    seq_fwd_x = torch.stack(seq_fwd_x).detach().numpy().reshape(num_cycles, num_steps+1, -1)
    seq_rev_x = torch.stack(seq_rev_x).detach().numpy().reshape(num_cycles, num_steps+1, -1)

    return seq_x, seq_fwd_x, seq_rev_x



# ------------------------------- line manifold ------------------------------ #
def generate_sequential_samples_line(ground_truth_manifold, model_line, num_steps, num_cycles, alphas, betas, one_minus_alphas_prod_sqrt):
    manifold_initial_point = ground_truth_manifold[np.random.randint(ground_truth_manifold.shape[0])].reshape(1, -1)
    xf_ad = neural_fwd_rev_cycle(manifold_initial_point, model_line, num_steps, alphas, betas, one_minus_alphas_prod_sqrt)
    xfs_line = []
    # for i in range(num_iters-1):
    # for i in trange(num_cycles-1, desc='sequential sampling', unit='cycles'):
    for i in range(num_cycles):
        xf_ad = neural_fwd_rev_cycle(xf_ad, model_line, num_steps, alphas, betas, one_minus_alphas_prod_sqrt)
        xfs_line.append(xf_ad)
    return torch.stack(xfs_line)



# ------------------------------------ SNN ----------------------------------- #
@torch.no_grad()
def p_sample_snn(model, x, t, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device):
    t = torch.tensor([t], device=device)
    
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_prod_sqrt, t, x))
    
    # Model output
    T = t.repeat(x.shape[0], 1) / n_steps
    eps_theta = model(x, T)
    
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    
    # Generate z
    z = torch.randn_like(x, device=device)
    
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    # sample = mean + sigma_t * z
    sample = mean
    return sample

@torch.no_grad()
def p_sample_loop_snn(model, shape, n_steps, device='cpu', init_x=None):
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(n_steps, device)
    
    if init_x == None:
        cur_x = torch.randn(shape, device=device)
    else:
        cur_x = init_x
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample_snn(model, cur_x, i, n_steps, alphas,betas,one_minus_alphas_prod_sqrt, device)
        x_seq.append(cur_x)
    x_seq = torch.stack(x_seq, dim=0)
    return x_seq