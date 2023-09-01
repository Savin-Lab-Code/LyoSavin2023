import submitit
import os
import json
import torch
from plot import *
from utils import *
from models import TimeAndClassEmbeddedNoiseEstimator
import torch.optim as optim
from utils import *
from tqdm import trange
import time
import dill
from generate_data import load_trimodal_data

def forward_process(num_steps, dataset, model_name, device):
    betas = make_beta_schedule(schedule='sigmoid', n_timesteps=num_steps, start=1e-5, end=2e-2)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_prod_log = torch.log(1 - alphas_prod)
    one_minus_alphas_prod_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        alphas_t = extract(alphas_bar_sqrt, t, x_0)
        alphas_1_m_t = extract(one_minus_alphas_prod_sqrt, t, x_0)
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    # plot the diffusion process
    fig, ax = plt.subplots(1, 10, figsize=(22, 3))
    lims = [-3,3]
    for i in range(10):
        q_i = q_x(dataset, torch.tensor([i * 10]))
        ax[i].scatter(q_i[:, 0], q_i[:, 1], color='white', edgecolor='peru', s=5)
        # ax[i].set_axis_off()
        ax[i].set(xlim=lims, ylim=lims)
        ax[i].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
        ax[i].set_aspect('equal')
    fig.tight_layout()
    plot_savedir = 'plots'
    figname = f'forward-process-{model_name}.png'
    plt.savefig(os.path.join(plot_savedir, figname))

    # larger image of dataset q(x_0)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    lims = [-1.5,1.5]
    q_i = q_x(dataset, torch.tensor([0]))
    ax.scatter(q_i[:, 0], q_i[:, 1], color='white', edgecolor='peru', s=5)
    ax.set(xlim=lims, ylim=lims)
    ax.set_title('$q(\mathbf{x}_{'+str(0)+'})$')
    ax.set_aspect('equal')
    figname = f'q0-{model_name}.png'
    plt.savefig(os.path.join(plot_savedir, figname))
    return betas, alphas, alphas_prod, alphas_prod_p, alphas_bar_sqrt, one_minus_alphas_prod_log, one_minus_alphas_prod_sqrt


def reverse_process(num_steps, num_hidden, num_classes, num_epochs, device, dataset, coefs):
    betas, alphas, alphas_prod, alphas_prod_p, alphas_bar_sqrt, one_minus_alphas_prod_log, one_minus_alphas_prod_sqrt = coefs
    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_prod_sqrt = one_minus_alphas_prod_sqrt.to(device)
    dataset = dataset.to(device)
    
    model = TimeAndClassEmbeddedNoiseEstimator(num_steps, num_hidden, num_classes)
    model.to(device)

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_size = 128
    start_time = time.time()
    
    model.train()
    for t in trange(num_epochs, desc='Training model', unit='epochs'):
        permutation = torch.randperm(dataset.size()[0], device=device)
    
        for i in range(0, dataset.size()[0], batch_size):
            # retrieve current batch
            indices = permutation[i:i+batch_size]
            batch_x = dataset[indices]
            
            # compute the loss
            loss = noise_estimation_loss(model, batch_x, num_steps, alphas_bar_sqrt, one_minus_alphas_prod_sqrt, device, has_class_label=True)
            # zero the gradients
            optimizer.zero_grad()
            # backward pass: compute the gradient of the loss wrt the parameters
            loss.backward()
            # call the step function to update the parameters
            optimizer.step()
        
        # print loss
        if (t % 100 == 0):
            print('t', t)
            print('loss', loss.item())

    end_time = time.time()
    duration = end_time - start_time
    duration_mins = duration / 60
    print(f'training took {duration:.0f} seconds, which is {duration_mins:.2f} minutes.')
    return model

def save_model(model, model_name):
    model_name = f'{model_name}.pt'
    savepath = os.path.join('.', 'saved_weights')
    torch.save(model.state_dict(), os.path.join(savepath, model_name))
    print('model saved!')
    
    

def train_model(model_name, num_steps, num_hidden, num_classes, num_samples, epochs, manifold_offsets):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ------------------------------ define dataset ------------------------------ #
    dataset = load_trimodal_data(num_samples, manifold_offsets)

    # ------------------------- forward diffusion process ------------------------ #
    coefs = forward_process(num_steps, dataset, model_name, device)

    # -------------------- TRAINING - reverse diffusion process ------------------ #
    model = reverse_process(num_steps, num_hidden, num_classes, epochs, device, dataset, coefs)
    save_model(model, model_name)
    


def main():
    print('we are running!')

    # --------------------------- set global parameters -------------------------- #
    model_name = f'class_condtl'
    model_number = 13
    num_steps = 100
    num_hidden = 16
    num_classes = 3
    num_samples = int(1e2)
    num_epochs = int(1000 * int(1e4)/num_samples)  # epochs scales inversely with num of samples
    # manifold_offsets = [[0,0], [4,0], [2,4]]  # original offset. manifolds are separated
    manifold_offsets = [[0,0], [1,0], [1,1]]
    manifold_separated = True


    # ----------------------------- other parameters ----------------------------- #
    mode = 'train'
    log_folder = os.path.join('cluster', 'logs/%j')

    # -------------------------- save model description -------------------------- #
    description = {
        'model_number': model_number,
        'model_name': model_name,
        'num_steps': num_steps,
        'num_hidden': num_hidden,
        'num_classes': num_classes,
        'num_samples': f'{num_samples:.0e}',
        'num_epochs': f'{num_epochs:.0e}',
        'manifold_offsets': manifold_offsets,
        'manifold_separated': manifold_separated,
    }
    json_savedir = 'model_description'
    model_name = f'{model_name}_{model_number}'
    json_name = f'{model_name}.json'
    with open(os.path.join(json_savedir, json_name), 'w') as file:
        json.dump(description, file)

    # ------------------------- submitit cluster executor ------------------------ #
    ex = submitit.AutoExecutor(folder=log_folder)

    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f'!!! Slurm executable `srun` not found. Will execute jobs on "{ex.cluster}"')

    # slurm parameters
    ex.update_parameters(
        slurm_job_name = model_name,
        nodes = 1,
        slurm_partition = 'gpu',
        slurm_gpus_per_task=1,
        slurm_constraint='v100-32gb',
        cpus_per_task=12,
        mem_gb=8,
        timeout_min=60,
    )

    jobs = []
    with ex.batch():
        if mode=='train':
            job = ex.submit(train_model, model_name, num_steps, num_hidden, num_classes, num_samples, num_epochs, manifold_offsets)
        jobs.append(job)
    print('all jobs submitted!')

    idx = 0
    print(f'Job {jobs[idx].job_id}')
    # idx += 1


if __name__ == '__main__':
    main()