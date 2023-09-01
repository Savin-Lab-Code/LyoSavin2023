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


def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2, device='cpu'):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps, device=device)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == 'sine':
        # betas = torch.linspace(start, end, n_timesteps)
        # betas = torch.sin(betas)
        
        linspace = torch.linspace(0, np.pi, n_timesteps)
        modulator = 0.5 * torch.cos(linspace) + 0.5
        betas = (end - start) * (1-modulator) + start
    return betas


def noise_estimation_loss(model, x_0, n_steps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device, norm='l1', has_class_label=False):
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

    if norm == 'l1':
        return torch.abs(e - output).mean()
    elif norm == 'l2':
        return (e - output).square().mean()


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

# --------------------- save model weights/description to a directory -------------------- #
def save_model_weights(model, model_name:str, model_number: int, savepath='saved_weights/'):
    model_name = f'{model_name}_{model_number}.pt'
    savepath = os.path.join('.', savepath)
    torch.save(model.state_dict(), os.path.join(savepath, model_name))
    print(f'model state dict saved in directory: {savepath}')
    
def save_model_description(model_name, model_number, description, json_savedir='model_description/'):
    '''save model description in a separate json file'''
    
    combined_model_name = f'{model_name}_{model_number}'
    json_name = f'{combined_model_name}.json'
    with open(os.path.join(json_savedir, json_name), 'w') as file:
        json.dump(description, file)
    
    print(f'model description saved in a json file in directory: {json_savedir}')
    
# --------------- load model weights/description to a directory -------------- #
def load_model_weights(model, model_name, model_num, device, loadpath='saved_weights/'):
    '''load the saved state dict of a model'''
    model_name = f'{model_name}_{model_num}.pt'
    loadpath = os.path.join(loadpath, model_name)
    state_dict = torch.load(loadpath, map_location=device)
    model.load_state_dict(state_dict)
    print('model loaded!')
    return model
    
def load_model_description(model_name, model_num, json_load_dir='model_description/'):
    '''load the model description file'''
    model_name = f'{model_name}_{model_num}'
    params = json.load(open(os.path.join(json_load_dir, f'{model_name}.json')))
    return params


# ------------------------------ forward process ----------------------------- #
def forward_process(num_steps, device, schedule='sigmoid', start=1e-5, end=2e-2):
    betas = make_beta_schedule(schedule=schedule, n_timesteps=num_steps, start=start, end=end, device=device)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1], device=device).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_prod_log = torch.log(1 - alphas_prod)
    one_minus_alphas_prod_sqrt = torch.sqrt(1 - alphas_prod)
    return betas, alphas, alphas_prod, alphas_prod_p, alphas_bar_sqrt, one_minus_alphas_prod_log, one_minus_alphas_prod_sqrt


# --------------- count the number of parameters in your model --------------- #
def count_parameters(model): 
	return sum(p.numel() for p in model.parameters() if p.requires_grad)
