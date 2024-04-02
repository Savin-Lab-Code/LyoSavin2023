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

def get_index_from_list(vals, t, x_shape):
    '''
    Returns a specific index t of a passed list of values `vals`, while considering
    the batch dimension
    '''
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t)
    # out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def rescale_samples_to_01(data):
    '''
    rescale the 2D data points to lie within (0, 1). Preserves aspect ratio
    '''
    min = torch.min(data)
    max = torch.max(data)

    return (data - min) / (max - min)

def rescale_samples_to_pm1(data):
    '''
    rescale the 2D data points to lie within (-1, 1). Preserves aspect ratio
    '''
    min = torch.min(data)
    max = torch.max(data)
    return ((data - min) / (max - min))*2 -1

def rescale_samples_to_pos_quad(data):
    '''
    rescale the 2D data points to lie within (0, 2). Preserves aspect ratio
    '''
    # min = torch.min(data)
    # max = torch.max(data)
    return (data + 1) / 2

def add_noise_to_image(x, noise_level):
    '''add gaussian noise to a single point'''
    identity_sample = torch.randn_like(x)
    noise = identity_sample * noise_level
    noisy_x = x + noise
    return noisy_x

def add_noise_to_dataset(data, noise_level):
    '''add gaussian noise to the data distribution without rescaling'''
    identity_sample = torch.randn_like(data)
    noise = identity_sample * noise_level
    noisy_data = data + noise
    return noisy_data

@torch.no_grad()
def generate_ascent_trail(
    model, 
    sigma_0 = 1.0, 
    sigma_L = 0.01,
    h_0 = 0.02,  # between 0 and 1. controls fraction of denoising correction that's taken
    beta = 0.8  # between 0 and 1. controls proportion of injected noise (beta=1 indicates no noise)
    ):
    '''algorithm for sampling from the implicit prior of a denoiser'''

    y_0 = y_t = torch.normal(mean=0.0, std=sigma_0, size=(2,))  # initial noisy_img
    ys = []
    
    def f(y_t):
        return model(y_t) - y_t
    
    t = 1
    sigma_t = sigma_0
    while sigma_t >= sigma_L:
        # store the y_t values
        ys.append(y_t)
        
        h_t = h_0 * t / (1 + h_0 * (t-1))
        d_t = f(y_t)
        
        sigma_sq_t = torch.square(torch.linalg.norm(d_t)) / 2
        sigma_t = torch.sqrt(sigma_sq_t)
        
        gamma_sq_t = (np.square((1 - beta * h_t)) - np.square(1 - h_t)) * sigma_sq_t
        gamma_t = torch.sqrt(gamma_sq_t)
        
        z_t = torch.randn(1)
        y_t = y_t + h_t*d_t + gamma_t*z_t
        t += 1
        
        if t > 100:
            break
        
    ys = torch.stack(ys, 0)
    return y_0, ys


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def make_beta_schedule(schedule='linear', n_timesteps=100, start=1e-5, end=1e-2, device='cpu'):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps, device=device)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == 'sine':
        linspace = torch.linspace(0, np.pi, n_timesteps, device=device)
        modulator = 1 - torch.cos(linspace)
        betas = (end - start)/2 * (modulator) + start
    return betas

def convert_beta_t_to_beta_l(beta_t, betas):
    beta_l_min = betas[0]
    beta_l_max = betas[-1]
    beta_l = (1 - beta_t) * (beta_l_max - beta_l_min) + beta_l_min
    return beta_l

def convert_beta_l_to_beta_t(beta_l, betas):
    # beta_l = beta_ls[t]
    beta_l_min = betas[0]
    beta_l_max = betas[-1]
    beta_t = 1 - (beta_l - beta_l_min) / (beta_l_max - beta_l_min)
    return beta_t


def noise_estimation_loss(model, x_0, n_steps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device, norm='l2', l1_reg=0, l1_reg_on_phase_weights=True, has_class_label=False):
    # remember, the images in this batch are also selected at random via randperm
    batch_size = x_0.shape[0]
    # has_class_label = (len(x_0[0]) == 3)
    
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,), device=device)
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    
    # if x_0 has class labels, separate the data from the class label
    if has_class_label:
        x_0, c = x_0[:, :-1], x_0[:, -1]
        c = c.long()
    
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, x_0)
    
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, x_0)
    e = torch.randn_like(x_0, device=device)
    
    # model input
    x = x_0 * a + e * am1
    
    # rescale t
    # t = t / n_steps
    # t = t
    
    if has_class_label:
        output = model(x, t, c)
    else:
        output = model(x, t)

    # we can put an L1 regularization term to penalize weights of the fully connected layer
    if l1_reg != 0:
        if l1_reg_on_phase_weights:
            # we want l1 regularization on the recurrent weights only
            l1_loss = l1_reg * torch.norm(model.features[0].linear.weight[:-1], p=1)
        else:
            # we want l1 regularization on the weights corresponding to the phase input as well (i.e. all of the fc weights)
            l1_loss = l1_reg * torch.norm(model.features[0].linear.weight, p=1)
    else:
        l1_loss = 0

    if norm == 'l1':
        return torch.abs(e - output).mean() + l1_loss
    elif norm == 'l2':
        return (e - output).square().mean() + l1_loss

def save_model_weights(model, model_name:str, model_number: int, savepath='saved_weights/'):
    '''save the state dict of a model to a given directory'''
    model_name = f'{model_name}_{model_number}.pt'
    savepath = os.path.join(base_dir, 'core', savepath, model_name)
    torch.save(model.state_dict(), savepath)
    print(f'model state dict saved in directory: {savepath}')
    
def save_model_description(model_name, model_number, description, json_savedir='model_description/'):
    '''save model description in a separate json file'''
    combined_model_name = f'{model_name}_{model_number}'
    json_name = f'{combined_model_name}.json'
    save_dir = os.path.join(base_dir, 'core', json_savedir, json_name)
    with open(save_dir, 'w') as file:
        json.dump(description, file)
    
    print(f'model description saved in a json file in directory: {json_savedir}')
    

def load_model_weights(model, model_name, model_num, device, loadpath='saved_weights/'):
    '''load the saved state dict of a model from a given directory'''
    model_name = f'{model_name}_{model_num}.pt'
    loadpath = os.path.join(base_dir, f'core/{loadpath}', model_name)
    state_dict = torch.load(loadpath, map_location=device)
    model.load_state_dict(state_dict)
    print('model loaded!')
    return model
    
    
def load_model_description(model_name, model_num, json_load_dir='model_description/'):
    '''load the model description file'''
    model_name = f'{model_name}_{model_num}.json'
    load_dir = os.path.join(base_dir, 'core', json_load_dir, model_name)
    params = json.load(open(load_dir))
    return params

def count_parameters(model): 
    '''Count the number of parameters in your model'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def remove_all_ticks_and_labels(ax, include_z_axis=False):
    def remove(ax):
        '''remove all ticks and labels from a matplotlib axis'''
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        ax.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False)
        if include_z_axis:
            ax.tick_params(
                axis='z',          # changes apply to the z-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=False)
        return ax
        
    if isinstance(ax, list) or isinstance(ax, np.ndarray):
        for a in ax:
            a = remove(a)
        return ax
    else:
        return remove(ax)


# -------------------- select and load weights into model -------------------- #
def select_model(model_name, model_version_number, device='cpu', print_details=False):
    from models import NoiseConditionalEstimatorConcat, UnbiasedNoiseConditionalEstimatorConcat3Layers, UnbiasedNoiseConditionalEstimatorConcat4Layers
    from models import VariableDendriticCircuit, VariableDendriticCircuitSomaBias
    from models import NoisyImageClassifierWithTimeEmbedding
    from models import SNN
    from utils import load_model_description, load_model_weights, count_parameters

    # diffusion model parameters
    model_details = load_model_description(model_name, model_version_number)

    model_name = model_details['model_name']
    model_num = model_details['model_number']
    num_steps = model_details['num_steps']
    num_hidden = model_details['num_hidden']

    # dim_amb = model_details['num_manifold_dims']
    if 'num_ambient_dims' in model_details.keys():
        dim_amb = model_details['num_ambient_dims']
    else: 
        if 'num_in' in model_details.keys():
            dim_amb = model_details['num_in']
        else:
        # if model_num == 47:
            dim_amb = 2
    
    
    dim_amb = int(dim_amb)

    print('ambient dimension is', dim_amb) if print_details else None
    # print('manifold dimension is', model_details['num_manifold_dims'])
    
    print('hidden units:', num_hidden) if print_details else None

    if 'manifold_offsets' in model_details.keys():
        manifold_offsets = model_details['manifold_offsets']
        print('manifold offsets are:', manifold_offsets) if print_details else None
    
    # classifiers
    if 'num_classes' in model_details.keys():
        num_classes = model_details['num_classes']
        num_classes = int(num_classes)

    # initialize model
    if model_name == 'unconditional-concat':
        model = NoiseConditionalEstimatorConcat(num_hidden)
    elif model_name[:23] == 'unconditional-dendritic' or model_name == 'unconditional-dendritic-3d-manifold':
        model = VariableDendriticCircuit(hidden_cfg=num_hidden, num_in=dim_amb, num_out=dim_amb, bias=True)
    elif model_name == 'noisy-image-classifier-with-noise-info':
        model = NoisyImageClassifierWithTimeEmbedding(dim_amb, num_hidden, num_classes, num_steps)
    elif model_name == 'snn':
        betas = forward_process(num_steps, device)[0]
        model = SNN(num_hidden, betas)

    # load model state dict
    model = load_model_weights(model, model_name, model_num, device)
    print(f'the model has {count_parameters(model)} parameters') if print_details else None
    
    model.eval()
    
    return model, num_steps, dim_amb



def load_model_weights_from_chkpt(model_name, model_num, epoch_number, checkpoint_path='saved_weights', device=torch.device('cpu')):
    from models import VariableDendriticCircuit
    # model, num_steps, ambient_dims = select_model(model_name, model_num)
    model_details = load_model_description(model_name, model_num)
    print('model loaded!', flush=True)
    model_name = model_details['model_name']
    model_num = model_details['model_number']
    num_steps = model_details['num_steps']
    num_hidden = model_details['num_hidden']
    
    if 'num_ambient_dims' in model_details.keys():
        dim_amb = model_details['num_ambient_dims']
    else: 
        if 'num_in' in model_details.keys():
            dim_amb = model_details['num_in']
        else:
        # if model_num == 47:
            dim_amb = 2
    dim_amb = int(dim_amb)

    if 'bias' in model_details:
        bias = model_details['bias']
    else: 
        bias = True

    if model_name[:23] == 'unconditional-dendritic':
        model = VariableDendriticCircuit(hidden_cfg=num_hidden, num_in=dim_amb, num_out=dim_amb, bias=bias)

    checkpoint_path = os.path.join(base_dir, 'core', checkpoint_path, f'{model_name}_{str(model_num)}')
    epoch_file = 'epoch='+str(epoch_number)
    file = torch.load(os.path.join(checkpoint_path, epoch_file, 'checkpoint.pt'), map_location=device)
    state_dict = file['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_steps, dim_amb

def load_optimizer_state_dict(optimizer, model_name, model_num, epoch_number, checkpoint_path='saved_weights', device=torch.device('cpu')):
    checkpoint_path = os.path.join(base_dir, 'core', checkpoint_path, f'{model_name}_{str(model_num)}')
    epoch_file = 'epoch='+str(epoch_number)
    file = torch.load(os.path.join(checkpoint_path, epoch_file, 'checkpoint.pt'), map_location=device)
    optimizer_state_dict = file['optimizer_state_dict']
    optimizer.load_state_dict(optimizer_state_dict)
    return optimizer


def save_or_load_to_zarr(mode, name, data=False):
    '''
    save or load numpy arrays to/from zarr format
    '''
    import zarr
    data_dir = os.path.join(base_dir, 'core/saved_arrays/')
    if mode=='save':
        zarr.save(os.path.join(data_dir, f'{name}.zarr'), data)
        print('saved!')
        return
    if mode=='load':
        print('loading!')
        return zarr.load(os.path.join(data_dir, f'{name}.zarr'))
    
    
def save_or_load_to_pt(mode, name, data=False):
    '''save or load torch tensors to/from pt format
    '''
    data_dir = os.path.join(base_dir, 'core/saved_arrays/')
    if mode=='save':
        torch.save(data, os.path.join(data_dir, f'{name}.pt'))
        print('saved!')
        return
    if mode=='load':
        print('loading the tensor onto the CPU!')
        return torch.load(os.path.join(data_dir, f'{name}.pt'))
