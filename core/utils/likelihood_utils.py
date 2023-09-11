import os, sys
project_root = os.path.abspath("")
if project_root[-12:] == 'LyoSavin2023':
    base_dir = project_root
else:
    base_dir = os.path.dirname(project_root)

import numpy as np
import torch
from tqdm.auto import trange

from prior_utils import forward_process

def calculate_x_from_z(z_t, L, sigma, seed=None):
    '''
    transforms the latent space Z to the data space X
    '''
    if seed is not None:
        torch.random.manual_seed(seed)
    num_samples = z_t.shape[0]
    Z_dim = z_t.shape[1]
    assert L.shape[1] == Z_dim
    X_dim = L.shape[0]
    
    epsilon = torch.randn(X_dim, num_samples)
    X = torch.mm(L, z_t.T) + sigma*epsilon
    return X

def compute_continuous_likelihood(z_i, x, L, sigma):
    '''
    computes p(X = x_i | Z = z_i)
    Given a data point z_i, compute the likelihood of observing x
    '''
    # L = torch.tensor(L, dtype=torch.float)
    # z_i = torch.tensor(z_i, dtype=torch.float)
    # x = torch.tensor(x, dtype=torch.float)
    
    x_mean = torch.mm(L, z_i.T)
    # print(x_mean.shape)
    # print(x.shape)
    assert x_mean.shape[0] == x.shape[0]
    
    prob = torch.exp(-torch.sum(torch.square(x - x_mean), axis=0) / (2 * sigma**2))
    # prob = prob.requires_grad_(True)
    return prob


def compute_log_continuous_likelihood(z_i, x, L, sigma):
    return torch.log(compute_continuous_likelihood(z_i, x, L, sigma))


# ----------------------- calculate scores/flow fields ----------------------- #
def compute_continuous_likelihood_score(z, x, L, sigma):
    '''
    takes in the likelihood model and computes the score for a given data point x
    '''
    z = z.detach().requires_grad_(True)
    probs = compute_continuous_likelihood(z, x, L, sigma)
    logits = torch.log(probs)  # this is = log p(x|z)
    return torch.autograd.grad(logits, z, torch.ones_like(logits))[0]

def compute_occlusion_score(zs, Mm, sigma=1):
    '''takes in a batch of z and calculates the score for a given deterministic constraint
    '''
    scores = []
    for z in zs:
        z = z.reshape(1, -1).requires_grad_(True)
        # residual = z - z_c = (I - MM^T) z
        residual = (torch.eye(2) - Mm) @ z.T
        dist = torch.norm(residual)**2 / (2*sigma**2)
        score = -torch.autograd.grad(dist, z, torch.ones_like(dist))[0]
        scores.append(score)
        z = z.detach()

    return torch.stack(scores).reshape(-1, 2)

def compute_occlusion_score_flow_field(Mm, lim=1.45, num_vectors_per_dim=11, vector_rescale_factor=0.4):
    '''computes the flow field for a continuous likelihood model, for a grid of hypothetical data points x_hyps
    '''
    # construct a grid of hypothetical data points
    # score_lim = lim - 0.1
    score_lim = lim
    x_hyps = []
    for i in np.linspace(-score_lim, score_lim, num_vectors_per_dim):
        x_hyp = np.linspace(start=[-i, score_lim], stop=[-i, -score_lim], num=num_vectors_per_dim)
        x_hyps.append(x_hyp)
    x_hyps = np.vstack(x_hyps)
    x_hyps = torch.tensor(x_hyps, dtype=torch.float)
    
    scores = compute_occlusion_score(x_hyps, Mm)
    scores = (scores.T/torch.norm(scores, dim=1)**vector_rescale_factor).T
    
    color = np.hypot(scores[:, 0], scores[:, 1])**2
    return x_hyps, scores, color

def compute_diffuser_score(diffusion_model, x, t):
    '''computes the score of the diffusion model at a given point x
    score = -grad log p(x_t|x_t+1) = x_t - estimated mean = estimated error = model output 
    '''
    eps_theta = diffusion_model(x, t)
    return eps_theta

def calculate_prior_score_flow_field(prior_sampler, lim=1.5, num_vectors_per_dim=15, t=0, vector_rescale_factor=0.7,):
    '''display the flow field for the prior sampler
    '''
    t = torch.tensor([t])

    score_xs = []
    score_ys = []
    for sample_y in np.linspace(-lim, lim, num_vectors_per_dim):
        for sample_x in np.linspace(-lim, lim, num_vectors_per_dim):
            
            # compute the score 
            x = torch.tensor([[sample_x, sample_y]], dtype=torch.float)
            diffuser_score = compute_diffuser_score(prior_sampler, x, t=t).detach()
            
            # rescales vectors so you don't have such a huge difference between the largest and smallest
            diffuser_score = diffuser_score/torch.norm(diffuser_score, dim=1)**vector_rescale_factor
            
            score_x, score_y = -diffuser_score[0,0], -diffuser_score[0,1]
            
            # collect data
            score_xs.append(score_x)
            score_ys.append(score_y)
    score_xs = torch.stack(score_xs).reshape(num_vectors_per_dim, num_vectors_per_dim)
    score_ys = torch.stack(score_ys).reshape(num_vectors_per_dim, num_vectors_per_dim)

    color = np.hypot(score_xs, score_ys)**2
    return score_xs, score_ys, color

def calculate_posterior_score_flow_field_bu(temps, lim, num_vectors_per_dim, diffusion_model, Mm, llh_sigma=0.2, llh_weight=0.5, vector_rescale_factor=0.6):
    '''display the flow field for the posterior score for a continuous likelihood
    '''
    
    if type(temps) == list:
        temps = torch.tensor(temps, dtype=torch.float).reshape(-1, 1)
    
    score_temps = []
    for t in temps:
        score_xs = []
        score_ys = []
        for sample_y in np.linspace(-lim, lim, num_vectors_per_dim):
            for sample_x in np.linspace(-lim, lim, num_vectors_per_dim):
                
                # compute the score 
                x = torch.tensor([[sample_x, sample_y]], dtype=torch.float)
                diffuser_score = compute_diffuser_score(diffusion_model, x, t=t).detach()
                
                likelihood_score = compute_occlusion_score(x, Mm, sigma=llh_sigma)
                
                posterior_score = -diffuser_score + llh_weight*likelihood_score
                
                posterior_score = posterior_score/torch.norm(posterior_score, dim=1)**vector_rescale_factor
                
                score_x, score_y = posterior_score[0,0], posterior_score[0,1]
                
                score_xs.append(score_x)
                score_ys.append(score_y)
        score_xs = torch.stack(score_xs).reshape(num_vectors_per_dim, num_vectors_per_dim)
        score_ys = torch.stack(score_ys).reshape(num_vectors_per_dim, num_vectors_per_dim)
        
        color = np.hypot(score_xs, score_ys)**2
        
        score_temps.append((score_xs, score_ys, color))
    return score_temps

# ---------------------------- posterior sampling ---------------------------- #
def posterior_sample(prior_sampler, t, z, x, L, sigma, s, num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device='cpu'):    
    '''
    Given a data point x, sample from the posterior transition distribution p(z_t|z_{t+1},x)
    '''
    from utils import extract
    
    # diffusion model
    t = torch.tensor([t], device=device)
    
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, z)) / extract(one_minus_alphas_prod_sqrt, t, z))
    
    # Diffusion model output
    T = t.repeat(z.shape[0], 1) / num_steps
    eps_theta = prior_sampler(z, T)
    
    # Final values
    mean = (1 / extract(alphas, t, z).sqrt()) * (z - (eps_factor * eps_theta))
    
    # Likelihoood score
    likelihood_score = compute_continuous_likelihood_score(z, x, L, sigma)
    
    # Generate z
    noise = torch.randn_like(z, device=device, dtype=torch.float)
    
    # Fixed sigma
    sigma_t = extract(betas, t, z).sqrt()
    
    # Posterior mean: \mu + \sigma_t * \nabla_x log p(y | x)
    posterior_mean = mean + s * sigma_t * likelihood_score
    
    sample = posterior_mean + sigma_t * noise
    # sample = mean + sigma_t * noise
    return (sample)

def posterior_sample_loop(prior_sampler, x, L, sigma, s, shape, n_steps, device='cpu'):
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(n_steps, device)
    
    cur_z = torch.randn(shape, device=device, dtype=torch.float)
    z_seq = [cur_z]
    for t in reversed(range(n_steps)):
        cur_z = posterior_sample(prior_sampler, t, cur_z, x, L, sigma, s, n_steps, alphas,betas,one_minus_alphas_prod_sqrt, device)
        z_seq.append(cur_z)
    z_seq = torch.stack(z_seq)
    return z_seq


def posterior_sample_occlusion(prior_sampler, t, z, Mm, sigma, s, num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device='cpu'):    
    '''
    Given a data point x, sample from the posterior transition distribution p(z_t|z_{t+1},x)
    '''
    from utils import extract
    
    # diffusion model
    t = torch.tensor([t], device=device)
    
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, z)) / extract(one_minus_alphas_prod_sqrt, t, z))
    
    # Diffusion model output
    T = t.repeat(z.shape[0], 1) / num_steps
    # T = t.repeat(z.shape[0], 1) 
    eps_theta = prior_sampler(z, T)
    
    # Final values
    mean = (1 / extract(alphas, t, z).sqrt()) * (z - (eps_factor * eps_theta))
    
    # Likelihoood score
    likelihood_score = compute_occlusion_score(z, Mm, sigma)
    
    # Generate z
    noise = torch.randn_like(z, device=device, dtype=torch.float)
    
    # Fixed sigma
    sigma_t = extract(betas, t, z).sqrt()
    
    # Posterior mean: \mu + \sigma_t * \nabla_x log p(y | x)
    posterior_mean = mean + s * sigma_t * likelihood_score
    
    sample = posterior_mean + sigma_t * noise
    # sample = mean + sigma_t * noise
    return (sample)

def posterior_sample_loop_occlusion(prior_sampler, M, sigma, s, shape, n_steps=100, status_bar=False, device='cpu'):
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(n_steps, device)
    
    cur_z = torch.randn(shape, device=device, dtype=torch.float)
    z_seq = [cur_z]

    Mm = M @ M.T
    Mm = torch.from_numpy(Mm).float()

    step = n_steps // 10
    for t in reversed(range(n_steps)):
        if t % step == 0 and status_bar:
            print(f'step {n_steps-t}/{n_steps}')
        cur_z = posterior_sample_occlusion(prior_sampler, t, cur_z, Mm, sigma, s, n_steps, alphas,betas,one_minus_alphas_prod_sqrt, device)
        z_seq.append(cur_z)
    z_seq = torch.stack(z_seq).detach().numpy()
    return z_seq


# ----------- iid sampled posterior distribution -- forward process ---------- #
def reversed_forward_process_posterior_loop_occlusion(prior_sampler, z_0, M, sigma, s, n_steps, alphas=None, betas=None, one_minus_alphas_prod_sqrt=None, device='cpu'):
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(n_steps, device)
    
    cur_z = z_0
    z_seq = [cur_z]

    Mm = M @ M.T
    Mm = torch.from_numpy(Mm).float()

    for t in range(n_steps):
        # cur_z, _ = p_sample_rev(prior_sampler, cur_z, t, num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device='cpu')
        cur_z = posterior_sample_occlusion(prior_sampler, t, cur_z, Mm, sigma, s, n_steps, alphas,betas,one_minus_alphas_prod_sqrt, device)
        z_seq.append(cur_z)
    z_seq = torch.stack(z_seq)
    return z_seq


def sequential_posterior_cycle(cur_z, prior_sampler, Mm, sigma, s, n_steps, alphas, betas, one_minus_alphas_bar_sqrt, device='cpu'):
    # biological forward process
    z_forward = [cur_z]
    for t in range(n_steps):
        cur_z = posterior_sample_occlusion(prior_sampler, t, cur_z, Mm, sigma, s, n_steps, alphas,betas,one_minus_alphas_bar_sqrt, device)
        z_forward.append(cur_z)
    z_forward = torch.stack(z_forward)
    
    # reverse process
    z_reverse = [cur_z]
    for t in reversed(range(n_steps)):
        cur_z = posterior_sample_occlusion(prior_sampler, t, cur_z, Mm, sigma, s, n_steps, alphas,betas,one_minus_alphas_bar_sqrt, device)
        z_reverse.append(cur_z)
    z_reverse = torch.stack(z_reverse)
    
    return cur_z, z_forward, z_reverse

# ------------- sequential sampling of the posterior distribution ------------ #
def sequential_posterior_sampler(prior_sampler, z, M, sigma, s, num_cycles, n_steps=100, burn=True, device='cpu', status_bar=False):
    '''
    for a given continuous likelihood, generate samples to/from the posterior distribution sequentially (rather than iid). 
    '''
    betas, alphas, _, _, _, _, one_minus_alphas_bar_sqrt = forward_process(n_steps, device)
    
    if type(z) == np.ndarray:
        cur_z = torch.tensor(z, dtype=torch.float)
    else: 
        cur_z = z
    
    Mm = M @ M.T
    Mm = torch.from_numpy(Mm).float()
    
    # burn the first sample
    if burn:
        cur_z = sequential_posterior_cycle(cur_z, prior_sampler, Mm, sigma, s, n_steps, alphas, betas, one_minus_alphas_bar_sqrt, device)[0]
    else: 
        print('not burning')
        cur_z = cur_z
    
    z_seq = []
    z_rev_seq = []
    step = num_cycles // 10
    for j in range(num_cycles):
        if j % step == 0 and status_bar:
            print(f'cycle {j}/{num_cycles}')
        cur_z, z_forward, z_reverse = sequential_posterior_cycle(cur_z, prior_sampler, Mm, sigma, s, n_steps, alphas, betas, one_minus_alphas_bar_sqrt, device)
        z_seq.append(cur_z)
        z_rev_seq.append(z_reverse)
    z_seq = torch.stack(z_seq).detach().numpy().reshape(num_cycles, -1, 2)
    z_rev_seq = torch.stack(z_rev_seq).detach().numpy().reshape(num_cycles, -1, 2)
    
    return z_seq, z_rev_seq

# ---------------------------------------------------------------------------- #
#                               classifier utils                               #
# ---------------------------------------------------------------------------- #

def compute_classifier_score(classifier, x, class_label, t):
    x = x.detach().requires_grad_(True)

    probs = classifier(x, t)
    
    logits = torch.log(probs)  # this is = log p(y|x)
    
    per_label_logit = logits[:, class_label]
    
    return torch.autograd.grad(per_label_logit, x, torch.ones_like(per_label_logit))[0]

def compute_classifier_score_flow_field(classifier, lim=1.5, num_vectors_per_dim=15, t=0, label=0, vector_rescale_factor=0.7,):
    t = torch.tensor([t])

    score_xs = []
    score_ys = []
    for sample_y in np.linspace(-lim, lim, num_vectors_per_dim):
        for sample_x in np.linspace(-lim, lim, num_vectors_per_dim):
            
            # compute the score 
            x = torch.tensor([[sample_x, sample_y]], dtype=torch.float)
            classifier_score = compute_classifier_score(classifier, x, class_label=label, t=t)
            
            # rescales vectors so you don't have such a huge difference between the largest and smallest
            classifier_score = classifier_score/torch.norm(classifier_score, dim=1)**vector_rescale_factor
            
            score_x, score_y = classifier_score[0,0], classifier_score[0,1]
            
            # collect data
            score_xs.append(score_x)
            score_ys.append(score_y)
    score_xs = torch.stack(score_xs).reshape(num_vectors_per_dim, num_vectors_per_dim)
    score_ys = torch.stack(score_ys).reshape(num_vectors_per_dim, num_vectors_per_dim)

    color = np.hypot(score_xs, score_ys)**2
    return score_xs, score_ys, color

from utils import extract
def top_down_posterior_sample(diffusion_model, classifier, label, x, t, s, prev_mean, alphas, betas, one_minus_alphas_prod_sqrt, device):
    
    ## diffusion model
    t = torch.tensor([t], device=device)
    
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_prod_sqrt, t, x))
    
    # Diffusion model output
    T = t.repeat(x.shape[0], 1)
    eps_theta = diffusion_model(x, T)
    
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    
    # Classifier output
    if prev_mean == None:
        print('yellow')
        prev_mean = mean
    # else: 
    #     prev_mean = mean
    classifier_score = compute_classifier_score(classifier, prev_mean, label, t)
    
    # Generate z
    z = torch.randn_like(x, device=device)
    
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    
    # Posterior mean: \mu + \sigma_t * \nabla_x log p(y | x)
    posterior_mean = mean + s * sigma_t * classifier_score
    
    sample = posterior_mean + sigma_t * z
    return (sample, mean)

def top_down_posterior_sample_loop(diffuser, classifier, label, s, shape, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device):
    cur_x = torch.randn(shape, device=device)
    x_seq = [cur_x]
    prev_mean = None
    for t in reversed(range(n_steps)):
        cur_x, mean = top_down_posterior_sample(diffuser, classifier, label, cur_x, t, s, prev_mean, alphas,betas,one_minus_alphas_prod_sqrt, device)
        prev_mean = mean
        x_seq.append(cur_x)
    x_seq = torch.stack(x_seq).detach()
    return x_seq


def perform_top_down_inference(diffuser, classifier, num_steps, label=2, s=0.3, sample_size=5e2, device='cpu'):
    sample_size = int(sample_size)
    
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device)
    x_seq = top_down_posterior_sample_loop(diffuser, classifier, label, s, (sample_size, 2), num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device)
    
    return x_seq, label



# ---------------------------------------------------------------------------- #
#                    both top-down and bottom-up likelihoods                   #
# ---------------------------------------------------------------------------- #
def variable_inference_sample(prior_sampler, classifier, Mm, mode, label, sigma, s_bu, s_td, shape, t, x, prev_mean, num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device, normalized_beta_schedule=False):
    '''generate samples from the posterior or prior for a given data point x
    '''
    # ------------------------------ diffusion model ----------------------------- #
    t = torch.tensor([t], device=device)
    
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_prod_sqrt, t, x))
    
    # Diffusion model output
    if normalized_beta_schedule:
        T = t.repeat(x.shape[0], 1) / num_steps
    else:
        T = t.repeat(x.shape[0], 1)
    eps_theta = prior_sampler(x, T)
    
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    
    # -------------------------------- likelihood -------------------------------- #
    # bottom-up (continuous) likelihood
    likelihood_score = compute_occlusion_score(x, Mm, sigma)
    
    # top-down likelihood (Classifier)
    if prev_mean == None:
        # print('yellow')
        prev_mean = mean
    # else: 
    #     if t > 97 or t<1:
            # print('not yellow')
    classifier_score = compute_classifier_score(classifier, prev_mean, label, t)
    
    # Generate z
    z = torch.randn_like(x, device=device)
    
    # Fixed sigma
    beta_t = extract(betas, t, x).sqrt()
    
    if mode == 'top-down':
        if t>98:
            print('top-down mode')
        posterior_mean = mean + s_td * beta_t * classifier_score
    if mode == 'bottom-up':
        if t>98:
            print('bottom-up mode')
        posterior_mean = mean + s_bu * beta_t * likelihood_score
    if mode == 'both':
        if t>98:
            print('both mode')
        # Posterior mean: \mu + \beta_t * \nabla_x log p(y | x) + \beta_t * \nabla_x log p(MMx |x)
        posterior_mean = mean + s_td * beta_t * classifier_score + s_bu * beta_t * likelihood_score
    if mode == 'neither' or mode == 'prior-only': 
        if t>98:
            print('prior-only mode')
        posterior_mean = mean
    
    sample = posterior_mean + beta_t * z
    return (sample, mean)

def perform_variable_inference(prior_sampler, classifier, v, mode, label, sigma, s_bu, s_td, n_steps, sample_size, device='cpu', normalized_beta_schedule=False):
    '''generate samples from the posterior or prior distribution of the diffusion model
    '''
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(n_steps, device)
    sample_size = int(sample_size)
    shape = (sample_size, 2)
    
    M = v / np.linalg.norm(v)
    Mm = M @ M.T
    Mm = torch.from_numpy(Mm).float()
    
    x_t = torch.randn(shape, device=device)
    x_seq = [x_t]
    prev_mean = None
    
    for t in reversed(range(n_steps)):
        x_t, mean = variable_inference_sample(prior_sampler, classifier, Mm, mode, label, sigma, s_bu, s_td, shape, t, x_t, prev_mean, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device, normalized_beta_schedule)
        prev_mean = mean
        x_seq.append(x_t)
    x_seq = torch.stack(x_seq).detach()
    
    return x_seq, label

def compute_joint_posterior_score_flow_field(prior_sampler, classifier, Mm, lim=1.5, num_vectors_per_dim=15, t=0, label=0, s_bu=0.3, s_td=0.3, vector_rescale_factor=0.7,):
    '''
    compute the score flow field for a posterior distribution that is a combination of top-down and bottom-up likelihoods
    '''
    score_lim = lim - 0.1
    lims = [-lim, lim]

    t = torch.tensor([t])

    score_xs = []
    score_ys = []
    for sample_y in np.linspace(-score_lim, score_lim, num_vectors_per_dim):
        for sample_x in np.linspace(-score_lim, score_lim, num_vectors_per_dim):
            
            # compute the score 
            x = torch.tensor([[sample_x, sample_y]], dtype=torch.float)
            
            diffuser_score = compute_diffuser_score(prior_sampler, x, t=t).detach()
            
            classifier_score = compute_classifier_score(classifier, x, class_label=label, t=t)
            
            sensory_score = compute_occlusion_score(x, Mm)
            
            posterior_score = -diffuser_score + s_td*classifier_score + s_bu*sensory_score
            
            # rescales vectors so you don't have such a huge difference between the largest and smallest
            posterior_score = posterior_score/torch.norm(posterior_score, dim=1)**vector_rescale_factor
            
            score_x, score_y = posterior_score[0,0], posterior_score[0,1]
            
            # collect data
            score_xs.append(score_x)
            score_ys.append(score_y)
    score_xs = torch.stack(score_xs).reshape(num_vectors_per_dim, num_vectors_per_dim)
    score_ys = torch.stack(score_ys).reshape(num_vectors_per_dim, num_vectors_per_dim)

    color = np.hypot(score_xs, score_ys)**2
    
    return score_xs, score_ys, color



def calculate_histogram_for_seq_data(seq_data, num_bins, lim):
    '''
    takes in a sequence of samples of shape (num_samples, num_steps, num_amb_dim=2) and returns a histogram of the data of shape (num_bins, num_bins)
    '''
    num_steps = seq_data.shape[1]
    bins = np.linspace(-lim, lim, num_bins+1)

    histograms = []
    for theta in range(num_steps):
        histograms.append(np.histogram2d(seq_data[:, theta, 0], seq_data[:, theta, 1], bins=bins)[0])
    histograms = np.stack(histograms)
    return histograms


# ---------------------- for the class conditional model --------------------- #
@torch.no_grad()
def p_sample_class(model, x, t, c, alphas, betas, one_minus_alphas_prod_sqrt, device):
    '''
    given an x_t and the corresponding t, produces a sample of p(x_t-1). 
    '''
    t = torch.tensor([t], device=device)
    
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_prod_sqrt, t, x))
    
    # Model output
    eps_theta = model(x, t, c)
    
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    
    # Generate z
    z = torch.randn_like(x, device=device)
    
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return (sample)

@torch.no_grad()
def p_sample_loop_class(model, shape, class_label, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device):
    '''
    given a sample from a unitary Gaussian, calculates samples from each step of the reverse process 
    '''
    if type(shape==tuple):
        if len(shape)==2:
            cur_x = torch.randn((shape[0], shape[1]), device=device)
    else:
        cur_x = torch.randn((shape[0], shape[1]-1), device=device)
    
    c = torch.tensor([class_label], device=device).int()

    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample_class(model, cur_x, i, c, alphas, betas, one_minus_alphas_prod_sqrt, device)
        x_seq.append(cur_x)
    x_seq = torch.stack(x_seq, 0)
    return x_seq


def calc_classifier_loss(model, x, num_classes):
    '''
    Calculates the loss associated with the model's prediction of the data. 
    This fn is given a batch of data with associated labels. 
    '''
    batch_size = x.shape[0]
    t_as_input = x.shape[1] == 4
    if not t_as_input:
        x, c = x[:, :2], x[:, 2]
        c = c.long()
        output = model(x)  # the output is from a softmax, so it produces a probability of each discrete class   
    else:
        x, c, t = x[:, :2], x[:, 2], x[:, 3]
        c = c.long()
        t = t.long()
        output = model(x, t)  # the output is from a softmax, so it produces a probability of each discrete class   
    
    c = F.one_hot(c, num_classes)  # encodes the classes with a one hot encoding
    
    loss = (c - output).square().mean()
    return loss


def get_classifier_accuracy(model, test_data):
    '''calculate the accuracy of the classification model on the test set'''
    testset_size = test_data.shape[0]
    print('test_data shape', test_data.shape)
    t_as_input = test_data.shape[1] == 4
    print('noise level is input to model: ', t_as_input)
    if not t_as_input:
        x, c = test_data[:, :-1], test_data[:, -1]
        c = c.long()
        probs = model(x)
    else:
        x, c, t = test_data[:, :2], test_data[:, 2], test_data[:, 3]
        c = c.long()
        t = t.long()
        probs = model(x, t)
        
    pred = probs.max(dim=1)[1]
    
    print(f'predictions: {pred[:10]}')
    print(f'correct: {c[:10]}')
    
    num_correct = pred.eq(c.view_as(pred)).sum().item()
    
    return num_correct / testset_size

