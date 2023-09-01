import numpy
import torch
from tqdm.auto import trange

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


def compute_likelihood(z_i, x, L, sigma):
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


def compute_log_likelihood(z_i, x, L, sigma):
    return torch.log(compute_likelihood(z_i, x, L, sigma))


def compute_likelihood_score(z, x, L, sigma):
    '''
    takes in the likelihood model and computes the score for a given data point x
    '''
    z = z.detach().requires_grad_(True)
    probs = compute_likelihood(z, x, L, sigma)
    logits = torch.log(probs)  # this is = log p(x|z)
    return torch.autograd.grad(logits, z, torch.ones_like(logits))[0]
    

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
    likelihood_score = compute_likelihood_score(z, x, L, sigma)
    
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
    from utils import forward_process
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(n_steps, device)
    
    cur_z = torch.randn(shape, device=device, dtype=torch.float)
    z_seq = [cur_z]
    for t in reversed(range(n_steps)):
        cur_z = posterior_sample(prior_sampler, t, cur_z, x, L, sigma, s, n_steps, alphas,betas,one_minus_alphas_prod_sqrt, device)
        z_seq.append(cur_z)
    z_seq = torch.stack(z_seq)
    return z_seq


# ----------------------------- occlusion problem ---------------------------- #
def compute_occlusion_score(zs, Mm, sigma=1):
    '''
    takes in a batch of z and calculates the score for a given deterministic constraint
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

def posterior_sample_loop_occlusion(prior_sampler, M, sigma, s, shape, n_steps, status_bar=False, device='cpu'):
    from utils import forward_process
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
    z_seq = torch.stack(z_seq)
    return z_seq


# ----------- iid sampled posterior distribution -- forward process ---------- #
def reversed_forward_process_posterior_loop_occlusion(prior_sampler, z_0, M, sigma, s, n_steps, alphas=None, betas=None, one_minus_alphas_prod_sqrt=None, device='cpu'):
    from utils import forward_process
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
    for a given continuous likelihood, generate samples to/from the posterior distribution, sequentially (rather than iid). 
    '''
    from utils import forward_process
    betas, alphas, _, _, _, _, one_minus_alphas_bar_sqrt = forward_process(n_steps, device)
    
    if type(z) == numpy.ndarray:
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


# --------------------- save or load arrays to zarr array -------------------- #
def save_or_load_to_zarr(mode, name, data=False):
    import zarr
    if mode=='save':
        zarr.save(f'saved_arrays/{name}.zarr', data)
        print('saved!')
        return
    if mode=='load':
        print('loading!')
        return zarr.load(f'saved_arrays/{name}.zarr')
    
    
def save_or_load_to_pt(mode, name, data=False):
    if mode=='save':
        torch.save(data, f'saved_arrays/{name}.pt')
        print('saved!')
        return
    if mode=='load':
        print('loading the tensor onto the CPU!')
        return torch.load(f'saved_arrays/{name}.pt')


# ---------------------------------------------------------------------------- #
#                               classifier utils                               #
# ---------------------------------------------------------------------------- #

def compute_classifier_score(classifier, x, class_label, t):
    x = x.detach().requires_grad_(True)
    
    probs = classifier(x, t)
    
    logits = torch.log(probs)  # this is = log p(y|x)
    
    per_label_logit = logits[:, class_label]
    
    return torch.autograd.grad(per_label_logit, x, torch.ones_like(per_label_logit))[0]

from utils import forward_process, extract
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
    from utils import forward_process
    sample_size = int(sample_size)
    
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device)
    x_seq = top_down_posterior_sample_loop(diffuser, classifier, label, s, (sample_size, 2), num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device)
    
    return x_seq, label



# ---------------------------------------------------------------------------- #
#                    both top-down and bottom-up likelihoods                   #
# ---------------------------------------------------------------------------- #
def variable_inference_sample(prior_sampler, classifier, Mm, mode, label, sigma, s_bu, s_td, shape, t, x, prev_mean, num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device):
    ## diffusion model
    t = torch.tensor([t], device=device)
    
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_prod_sqrt, t, x))
    
    # Diffusion model output
    T = t.repeat(x.shape[0], 1)
    # T = t.repeat(x.shape[0], 1) / num_steps
    eps_theta = prior_sampler(x, T)
    
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    
    # bottom-up likelihood
    likelihood_score = compute_occlusion_score(x, Mm, sigma)
    
    # Classifier
    if prev_mean == None:
        print('yellow')
        prev_mean = mean
    else: 
        if t > 97 or t<1:
            print('not yellow')
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

def perform_variable_inference(prior_sampler, classifier, v, mode, label, sigma, s_bu, s_td, n_steps, sample_size, device):
    import numpy as np
    from utils import forward_process
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
        x_t, mean = variable_inference_sample(prior_sampler, classifier, Mm, mode, label, sigma, s_bu, s_td, shape, t, x_t, prev_mean, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device)
        prev_mean = mean
        x_seq.append(x_t)
    x_seq = torch.stack(x_seq).detach()
    
    return x_seq, label




def compute_diffuser_score(diffusion_model, x, t):
    '''
    computes the score of the diffusion model at a given point x
    score = -grad log p(x_t|x_t+1) = x_t - estimated mean = estimated error = model output 
    '''
    eps_theta = diffusion_model(x, t)
    return eps_theta



def calculate_posterior_score_fields_various_temps(temps, lim, num_samples, diffusion_model, Mm, llh_sigma=0.2, llh_weight=0.5):
    from likelihood_utils import compute_diffuser_score, compute_likelihood_score
    import numpy as np

    # display the flow field for the prior sampler 
    lim = 1.5
    score_lim = lim
    lims = [-lim, lim]
    num_samples = 14

    print('temps are:', temps.detach().numpy().T.reshape(-1).astype(int))
    
    score_temps = []
    for t in temps:
        score_xs = []
        score_ys = []
        for sample_y in np.linspace(-lim, lim, num_samples):
            for sample_x in np.linspace(-lim, lim, num_samples):
                
                # compute the score 
                x = torch.tensor([[sample_x, sample_y]], dtype=torch.float)
                diffuser_score = compute_diffuser_score(diffusion_model, x, t=t).detach()
                
                likelihood_score = compute_occlusion_score(x, Mm, sigma=llh_sigma)
                
                posterior_score = -diffuser_score + llh_weight*likelihood_score
                # posterior_score = -diffuser_score
                # posterior_score = 0.5 * likelihood_score
                
                score_x, score_y = posterior_score[0,0], posterior_score[0,1]
                
                # collect data
                score_xs.append(score_x)
                score_ys.append(score_y)
        score_xs = torch.stack(score_xs).reshape(num_samples, num_samples)
        score_ys = torch.stack(score_ys).reshape(num_samples, num_samples)

        score_temps.append(torch.stack((score_xs, score_ys), dim=2))
        color = np.hypot(score_xs, score_ys)**2
    score_temps = torch.stack(score_temps)
    print('shape of `score_temps` is:', score_temps.shape)

    return score_temps, color


def compute_cosine_similarity(vector1, vector2):
    '''
    gets two vectors of the shape (num_samples, num_dims) and returns the cosine similarity between them
    returns a vector of shape (num_samples, 1)
    '''
    
    # shape = vector1.shape
    # assert shape == vector2.shape
    # num_samples = shape[0]
    
    # sims = []
    # for n in len(num_samples):
        # sim = vector1[n] @ vector2[n] / (torch.norm(vector1[n]) * torch.norm(vector2[n]))
    #     sims.append(sim)
    # sims = torch.stack(sims)
    
    sim = vector1 @ vector2 / (torch.norm(vector1) * torch.norm(vector2))
    return sim.item()


def calculate_histogram_for_seq_data(seq_data, num_bins, lim):
    '''
    takes in a sequence of samples of shape (num_samples, num_steps, num_amb_dim=2) and returns a histogram of the data of shape (num_bins, num_bins)
    '''
    import numpy as np
    num_steps = seq_data.shape[1]
    bins = np.linspace(-lim, lim, num_bins+1)

    histograms = []
    for theta in range(num_steps):
        histograms.append(np.histogram2d(seq_data[:, theta, 0], seq_data[:, theta, 1], bins=bins)[0])
    histograms = np.stack(histograms)
    return histograms
