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
from tqdm.auto import trange, tqdm
# device = 'cpu'

from prior_utils import p_rev_loop, p_sample_loop


def get_coefficients(num_steps, device, schedule='sigmoid', start=1e-5, end=2e-2):
    '''calculate the forward process for the given noise schedule'''
    from utils import make_beta_schedule
    betas = make_beta_schedule(schedule=schedule, n_timesteps=num_steps, start=start, end=end, device=device)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    one_minus_alphas_prod_sqrt = torch.sqrt(1 - alphas_prod)
    return betas, alphas, one_minus_alphas_prod_sqrt


@torch.no_grad()
def p_sample(model, x, t, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device):
    """
    returns one step of the neural forward process. takes a noisy data sample x_{t-1} and returns a more noisy x_t.
    """
    from utils import extract
    t = torch.tensor([t], device=device)
    T = t.repeat(x.shape[0], 1)
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_prod_sqrt, t, x))
    # Model output
    eps_theta = model(x, T)
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x, device=device)
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return sample

# ------- alternating between the neural forward and reverse processes ------- #
def perform_one_cycle(init_x, model, num_steps, coefficients, schedule='sine', device='cpu'):
    '''
    undergoes one cycle of the oscillation.
    diffuses the datapoints using the neural forward process, and then anti-diffuses using the neural reverse process.
    '''
    betas, alphas, one_minus_alphas_prod_sqrt = coefficients
    
    if type(init_x) == np.ndarray:
        init_x = torch.tensor(init_x, dtype=torch.float)
    else: 
        init_x = init_x
    
    # neural forward process
    cur_x = init_x
    for i in range(num_steps):
        cur_x = p_sample(model, cur_x, i, num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device)
    
    # neural reverse process
    for i in reversed(range(num_steps)):
        cur_x = p_sample(model, cur_x, i, num_steps, alphas,betas,one_minus_alphas_prod_sqrt, device)
    return cur_x


def neural_sampling_custom_betas(model, init_x, num_cycles, betas, period=1, amplitude=1, disable_tqdm=False, schedule='sine', device='cpu'):
    # customize beta schedule
    from utils import convert_beta_t_to_beta_l
    import numbers
    
    num_cycles = int(num_cycles)
    
    if isinstance(period, numbers.Number) and isinstance(amplitude, numbers.Number):
        # both period and amplitude are scalars
        num_steps = int(100*period)
        b_l_different_amp_min = convert_beta_t_to_beta_l(1-amplitude, betas).to(device)
        coefficients = get_coefficients(num_steps, device, schedule, end=b_l_different_amp_min)
    elif isinstance(period, (list, torch.Tensor, np.ndarray)) and isinstance(amplitude, (list, torch.Tensor, np.ndarray)):
        # both period and amplitude are lists
        num_steps_per_cycle = (100*period).to(torch.int64)
        b_l_different_amp_min = convert_beta_t_to_beta_l(1-amplitude, betas)
        coeff_per_cycle = []
        for i in range(num_cycles):
            coeff = get_coefficients(num_steps_per_cycle[i], device, schedule, end=b_l_different_amp_min[i])
            coeff_per_cycle.append(coeff)
    else:
        raise ValueError('period and amplitude must both be scalars or lists/array/tensors')
    
    if type(init_x) == np.ndarray:
        init_x = torch.tensor(init_x, dtype=torch.float)
    else: 
        init_x = init_x
    
    # burn the first sample
    coefficients = get_coefficients(100, device, schedule)
    x = perform_one_cycle(init_x, model, 100, coefficients, schedule, device)
    
    seq_x = []    
    for i in tqdm(range(num_cycles), disable=disable_tqdm):
        if not isinstance(period, numbers.Number) and not isinstance(amplitude, numbers.Number):
            coefficients = coeff_per_cycle[i]
            num_steps = num_steps_per_cycle[i]
        x = perform_one_cycle(x, model, num_steps, coefficients, schedule, device)
        seq_x.append(x)
    seq_x = torch.stack(seq_x).detach().cpu().numpy().reshape(num_cycles, -1)
    return seq_x

def repeat_scalar(scalar, repeats):
    return torch.tensor(scalar).tile(int(repeats))
    