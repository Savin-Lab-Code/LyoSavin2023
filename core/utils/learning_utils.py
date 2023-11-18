import os, sys
project_root = os.path.abspath("")
if project_root[-12:] == 'LyoSavin2023':
    base_dir = project_root
else:
    base_dir = os.path.dirname(project_root)

import torch
import torch.nn.functional as F
import numpy as np
import os
import json


def train_model_online(model, dataset, n_epochs, lr, tb, n_steps=100, schedule='sine', device='cpu', ):
    '''
    train model during the cyclic inference process, by comparing the estimated x_0 at each timestep against the ground truth x_0
    '''
    from tqdm.auto import tqdm
    from utils import extract
    import torch.optim as optim
    
    model = model.to(device)
    model.train()
    dataset = dataset.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_steps = int(n_steps)
    
    betas, alphas, _, _, alphas_bar_sqrt, _, one_minus_alphas_bar_sqrt = forward_process(n_steps, device, schedule)
    
    def generate_noisy_sample_for_lambda(x_0, l):
        # x0 multiplier
        a = extract(alphas_bar_sqrt, l, x_0)
        
        # epsilon multiplier
        am1 = extract(one_minus_alphas_bar_sqrt, l, x_0)
        
        # epsilon
        e_l = torch.randn_like(x_0, device=device)
        
        x_l = a*x_0 + am1*e_l
        return x_l, e_l

    
    def reverse_process_one_step(model, x_t, eps_t, t, x_0, time_idx):
        '''one step of the reverse process. takes a noisy data sample x_{t} and returns a less noisy sample x_{t+1}.'''
        from utils import extract
        
        t = torch.tensor([t], device=device)
        T = t.repeat(x_t.shape[0], 1)
        
        # Model output
        x_t = x_t.detach()
        eps_hat = model(x_t, T)
        
        # compute error
        # loss = (eps_hat - eps_t).square().mean()
        
        # Factor to the model output
        eps_factor = ((1 - extract(alphas, t, x_t)) / extract(one_minus_alphas_bar_sqrt, t, x_t))

        # Final values
        mean = (1 / extract(alphas, t, x_t).sqrt()) * (x_t - (eps_factor * eps_hat))
        
        # calculate loss and update
        loss = (mean - x_0).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tb.add_scalar('Loss', loss.item(), time_idx)
        
        # Generate z
        z = torch.randn_like(x_t, device=device)
        
        # Fixed sigma
        sigma_t = extract(betas, t, x_t).sqrt()
        sample = mean + sigma_t * z
        
        return sample, z, loss.item()
    
    x_0s = []
    for i in tqdm(range(int(n_epochs)), total=int(n_epochs), desc='Training model', unit='epochs', miniters=int(n_epochs)/100, maxinterval=float("inf")):
        # grab a random sample from the dataset
        x_0 = dataset[np.random.randint(0, len(dataset))].reshape(1, -1)
        x_0s.append(x_0)

        if i<=n_epochs//2:
            sampling_range = 100
        else:
            sampling_range = 50
        
        for l in reversed(range(sampling_range)):
            if l==99:
                x_l, e_l = generate_noisy_sample_for_lambda(x_0, l=torch.tensor([l], device=device)) 
            else:
                time_idx = 2*sampling_range*i + (sampling_range-1-l)
                x_l, e_l, loss = reverse_process_one_step(model, x_l, e_l, l, x_0, time_idx)
            
        for l in range(sampling_range):
            time_idx = 2*sampling_range*i + sampling_range + l
            x_l, e_l, loss = reverse_process_one_step(model, x_l, e_l, l, x_0, time_idx)
    tb.flush()
    
    x_0s = np.stack(x_0s).squeeze()
    return model, x_0s
            

def noise_based_online_training(model, dataset, n_epochs, lr, tb, n_steps=100, schedule='sine', device='cpu', ):
    '''
    train model during the cyclic inference process, by comparing the estimated noise at each timestep against the ground truth noise
    '''
    
    from tqdm.auto import tqdm
    from utils import extract
    import torch.optim as optim
    
    model = model.to(device)
    model.train()
    dataset = dataset.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_steps = int(n_steps)
    
    betas, alphas, _, _, alphas_bar_sqrt, _, one_minus_alphas_bar_sqrt = forward_process(n_steps, device, schedule)
    
    def generate_noisy_sample_for_lambda(x_0, l):
        # x0 multiplier
        a = extract(alphas_bar_sqrt, l, x_0)
        
        # epsilon multiplier
        am1 = extract(one_minus_alphas_bar_sqrt, l, x_0)
        
        # epsilon
        e_l = torch.randn_like(x_0, device=device)
        
        x_l = a*x_0 + am1*e_l
        return x_l, e_l

    
    def reverse_process_one_step(model, x_t, eps_t, t, x_0, time_idx):
        '''one step of the reverse process. takes a noisy data sample x_{t} and returns a less noisy sample x_{t+1}.'''
        from utils import extract
        
        t = torch.tensor([t], device=device)
        T = t.repeat(x_t.shape[0], 1)
        
        # Model output
        x_t = x_t.detach()
        eps_hat = model(x_t, T)
        
        # compute error
        loss = (eps_hat - eps_t).square().mean()
        
        # Factor to the model output
        eps_factor = ((1 - extract(alphas, t, x_t)) / extract(one_minus_alphas_bar_sqrt, t, x_t))

        # Final values
        mean = (1 / extract(alphas, t, x_t).sqrt()) * (x_t - (eps_factor * eps_hat))
        
        # calculate loss and update
        # loss = (mean - x_0).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tb.add_scalar('Loss', loss.item(), time_idx)
        
        # Generate z
        z = torch.randn_like(x_t, device=device)
        
        # Fixed sigma
        sigma_t = extract(betas, t, x_t).sqrt()
        sample = mean + sigma_t * z
        
        return sample, z, loss.item()
    
    x_0s = []
    for i in tqdm(range(int(n_epochs)), total=int(n_epochs), desc='Training model', unit='epochs', miniters=int(n_epochs)/100, maxinterval=float("inf")):
        # grab a random sample from the dataset
        x_0 = dataset[np.random.randint(0, len(dataset))].reshape(1, -1)
        x_0s.append(x_0)
        
        for l in reversed(range(100)):
            if l==99:
                x_l, e_l = generate_noisy_sample_for_lambda(x_0, l=torch.tensor([l], device=device)) 
            else:
                time_idx = 200*i + (99-l)
                x_l, e_l, loss = reverse_process_one_step(model, x_l, e_l, l, x_0, time_idx)
            
        for l in range(n_steps):
            time_idx = 200*i + 100 + l
            x_l, e_l, loss = reverse_process_one_step(model, x_l, e_l, l, x_0, time_idx)
    tb.flush()
    
    x_0s = np.stack(x_0s).squeeze()
    return model, x_0s
            


def faster_online_training(model, dataset, n_epochs, lr, tb, n_steps=100, schedule='sine', device='cpu', ):
    '''
    train model during the cyclic inference process, by comparing the estimated noise at each timestep against the ground truth noise
    '''
    
    from tqdm.auto import tqdm
    from utils import extract
    import torch.optim as optim
    
    model = model.to(device)
    model.train()
    dataset = dataset.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_steps = int(n_steps)
    
    betas, alphas, _, _, alphas_bar_sqrt, _, one_minus_alphas_bar_sqrt = forward_process(n_steps, device, schedule)
    
    def generate_noisy_sample_for_lambda(x_0, l):
        # x0 multiplier
        a = extract(alphas_bar_sqrt, l, x_0)
        
        # epsilon multiplier
        am1 = extract(one_minus_alphas_bar_sqrt, l, x_0)
        
        # epsilon
        e_l = torch.randn_like(x_0, device=device)
        
        x_l = a*x_0 + am1*e_l
        return x_l, e_l

    
    def reverse_process_one_step(model, x_t, eps_t, t, x_0, time_idx):
        '''one step of the reverse process. takes a noisy data sample x_{t} and returns a less noisy sample x_{t+1}.'''
        from utils import extract
        
        t = torch.tensor([t], device=device)
        T = t.repeat(x_t.shape[0], 1)
        
        # Model output
        x_t = x_t.detach()
        eps_hat = model(x_t, T)
        
        loss = (eps_hat - eps_t).square().mean()
        
        # Factor to the model output
        eps_factor = ((1 - extract(alphas, t, x_t)) / extract(one_minus_alphas_bar_sqrt, t, x_t))

        # Final values
        mean = (1 / extract(alphas, t, x_t).sqrt()) * (x_t - (eps_factor * eps_hat))
        
        # calculate loss and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tb.add_scalar('Loss', loss.item(), time_idx)

    sampling_range = 100
    time_idx = 0
    cycle_idx = np.append(np.arange(99, -1, -1), np.arange(0, 100, 1))
    
    for i in tqdm(range(int(n_epochs)), total=int(n_epochs), desc='Training model', unit='epochs', miniters=int(n_epochs)/100, maxinterval=float("inf")):
        for l in cycle_idx:
            # grab a random sample from the dataset
            x_0 = dataset[np.random.randint(0, len(dataset))].reshape(1, -1)
            
            # generate noisy data
            x_l, eps_l = generate_noisy_sample_for_lambda(x_0, l=torch.tensor([l], device=device)) 
            
            # learn to denoise it
            reverse_process_one_step(model, x_l, eps_l, l, x_0, time_idx)
            time_idx += 1

    tb.flush()
    return model