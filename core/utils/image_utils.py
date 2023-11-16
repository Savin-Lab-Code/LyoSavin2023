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

def beta_schedule(schedule='linear', n_timesteps=100, start=1e-5, end=1e-2, device='cpu'):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "sigmoid":
        betas = sigmoid_beta_schedule(n_timesteps)
        betas = betas.to(device)
    elif schedule == 'sine':
        linspace = torch.linspace(0, np.pi, n_timesteps, device=device)
        modulator = 1 - torch.cos(linspace)
        betas = (end - start)/2 * (modulator) + start
    return betas


def calculate_coefficients(num_steps, device, schedule='sigmoid', start=3e-3, end=1.):
    '''calculate the forward process for the given noise schedule'''
    betas = beta_schedule(schedule=schedule, n_timesteps=num_steps, start=start, end=end, device=device)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1], device=device).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_prod_log = torch.log(1 - alphas_prod)
    one_minus_alphas_prod_sqrt = torch.sqrt(1 - alphas_prod)
    return betas, alphas, alphas_prod, alphas_prod_p, alphas_bar_sqrt, one_minus_alphas_prod_log, one_minus_alphas_prod_sqrt


def calculate_loss(model, x0_data, n_steps=100, forward_schedule='sine', norm='l2', device='cpu'):
    from utils import extract
    
    batch_size = x0_data.shape[0]
    x0_data = x0_data.reshape(batch_size, -1)
    x0_data = rescale_to_neg_one_to_one(x0_data)
    
    _, _, _, _, alphas_bar_sqrt, _, one_minus_alphas_bar_sqrt = calculate_coefficients(n_steps, device, forward_schedule)
    
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
    
    
def generate_noisy_sample(num_steps, t, x_0, schedule='sigmoid', start=3e-3, end=2e-2):
    '''returns a diffused sample of x_0 at time t'''
    from utils import extract
    _, _, alphas_prod, _, alphas_bar_sqrt, _, one_minus_alphas_prod_sqrt = calculate_coefficients(num_steps, device, schedule, start=start, end=end)
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
    t = torch.linspace(0, timesteps, steps) / timesteps
    
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return torch.clip(betas, 0, 0.999)


def rescale_to_neg_one_to_one(img):
    return img * 2 - 1

def unscale_to_zero_to_one(t):
    return (t + 1) * 0.5