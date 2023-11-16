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


def calculate_loss(model, x0_data, n_steps=100, forward_schedule='sine', norm='l2', device='cpu'):
    from utils import extract
    from prior_utils import forward_process
    batch_size = x0_data.shape[0]
    x0_data = x0_data.reshape(batch_size, -1)
    x0_data = rescale_to_neg_one_to_one(x0_data)
    
    _, _, _, _, alphas_bar_sqrt, _, one_minus_alphas_bar_sqrt = forward_process(n_steps, device, forward_schedule, start=1e-5, end=1.)
    
    # Select a random step for each example
    # t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,), device=device)
    # t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    t = torch.randint(0, n_steps, size=(batch_size,), device=device).long()
    
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, x0_data)
    
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, x0_data)
    e = torch.randn_like(x0_data, device=device)
    
    # model input
    x = x0_data * a + e * am1

    output = model(x, t)

    if norm == 'l1':
        return torch.abs(e - output).mean()
    elif norm == 'l2':
        return (e - output).square().mean()
    
    
def generate_noisy_sample(num_steps, t, x_0, schedule='sigmoid', start=1e-5, end=2e-2):
    '''returns a diffused sample of x_0 at time t'''
    from prior_utils import forward_process
    from utils import extract
    _, _, alphas_prod, _, alphas_bar_sqrt, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device, schedule, start=start, end=end)
    batch_size = x_0.shape[0]
    x_0 = rescale_to_neg_one_to_one(x_0)
    
    t = torch.tensor([t], dtype=torch.long)

    scale_factor_t = extract(alphas_bar_sqrt, t, x_0)
    
    # Generate z
    z = torch.randn_like(x_0, device=device)

    # Fixed sigma
    sigma_t = extract(one_minus_alphas_prod_sqrt, t, x_0)
    
    x_t = scale_factor_t * x_0 + sigma_t * z
    return x_t


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return torch.clip(betas, 0, 0.999)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    import math
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def rescale_to_neg_one_to_one(img):
    return img * 2 - 1

def unscale_to_zero_to_one(t):
    return (t + 1) * 0.5



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