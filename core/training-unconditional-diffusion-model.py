import submitit
import os
import json
import torch
from plot import *
from utils import *
from models import NoiseConditionalEstimatorConcat, UnbiasedNoiseConditionalEstimatorConcat4Layers
from models import  VariableDendriticCircuit, VariableDendriticCircuitSomaBias
import torch.optim as optim
from utils import *
from tqdm import trange
import time
import dill
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataset_utils import load_trimodal_data, load_unimodal_data, load_unimodal_data_3d, load_unimodal_data_nd

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
    plot_savedir = 'plots/archive'
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


def reverse_process(model_name, num_steps, num_hidden, num_dims, num_epochs, batch_size, lr, device, dataset, coefs, pretrained_model):
    betas, alphas, alphas_prod, alphas_prod_p, alphas_bar_sqrt, one_minus_alphas_prod_log, one_minus_alphas_prod_sqrt = coefs
    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_prod_sqrt = one_minus_alphas_prod_sqrt.to(device)
    dataset = dataset.to(device)
    
    model = NoiseConditionalEstimatorConcat(num_hidden)
    # model = VariableDendriticCircuit(hidden_cfg=num_hidden, num_in=num_dims, num_out=num_dims, bias=True)
    # model = VariableDendriticCircuit(hidden_cfg=num_hidden, num_in=num_dims, num_out=num_dims, bias=False)
    # model = UnbiasedNoiseConditionalEstimatorConcat4Layers(num_hidden)
    # model = UnbiasedNoiseConditionalEstimatorConcat4Layers(num_hidden, num_in=3, num_out=3, bias=False)
    # model = VariableDendriticCircuitSomaBias(hidden_cfg=num_hidden, num_in=num_dims, num_out=num_dims, bias=False)

    
    if pretrained_model['use_pretrained_model_weights'] == True:
        from utils import load_model_weights
        pretrained_model_name = pretrained_model['model_name']
        pretrained_model_num = pretrained_model['model_num']
        model = load_model_weights(model, pretrained_model_name, pretrained_model_num, device)
    
    model.to(device)

    # lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # batch_size = 256

    tb = SummaryWriter(f'runs/{model_name}')

    start_time = time.time()
    
    model.train()
    for t in trange(num_epochs, desc='Training model', unit='epochs'):
        permutation = torch.randperm(dataset.size()[0], device=device)
    
        for i in range(0, dataset.size()[0], batch_size):
            # retrieve current batch
            indices = permutation[i:i+batch_size]
            batch_x = dataset[indices]
            
            # compute the loss
            loss = noise_estimation_loss(model, batch_x, num_steps, alphas_bar_sqrt, one_minus_alphas_prod_sqrt, device, has_class_label=False)
            # zero the gradients
            optimizer.zero_grad()
            # backward pass: compute the gradient of the loss wrt the parameters
            loss.backward()
            # call the step function to update the parameters
            optimizer.step()
        
        # write to tensorboard
        tb.add_scalar('Loss', loss.item(), t)

        # print loss
        if (t % 1000 == 0):
            print('t', t)
            print('loss', loss.item())
    tb.flush()


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
    
    

def train_model(model_name, num_steps, num_hidden, num_samples, epochs, batch_size, lr, manifold_type, manifold_offsets, num_ambient_dims, manifold_rotation_angle, pretrained_model):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ------------------------------ define dataset ------------------------------ #
    dataset = load_trimodal_data(num_samples, manifold_offsets, train_test_split=False, add_class_label=False, plot=False, noise=0)
    
    # dataset = load_unimodal_data(num_samples, manifold_type=manifold_type, offset=manifold_offsets, train_test_split=False, add_class_label=False)
    # dataset = make_nd_dataset(num_samples, manifold_type, n_dims=num_ambient_dims, theta=manifold_rotation_angle)
    # dataset = load_unimodal_data_nd(num_samples, manifold_type, dim_amb=num_ambient_dims, train_test_split=False)
    # dataset = load_unimodal_data_nd(num_samples, manifold_type=manifold_type, dim_amb=num_ambient_dims, rotation_angle=manifold_rotation_angle)

    # ------------------------- forward diffusion process ------------------------ #
    coefs = forward_process(num_steps, dataset, model_name, device)

    # -------------------- TRAINING - reverse diffusion process ------------------ #
    model = reverse_process(model_name, num_steps, num_hidden, num_ambient_dims, epochs, batch_size, lr, device, dataset, coefs, pretrained_model)
    save_model(model, model_name)



def main():
    print('we are running!')

    # --------------------------- set global parameters -------------------------- #
    # model_name = f'unbiased-unconditional-4layers'  # `unconditional-concat` or `unconditional-dendritic` or `unbiased-unconditional-3layers` or `unbiased-unconditional-4layers`
    # model_name = f'unconditional-dendritic-3d-manifold'  # `unconditional-concat` or `unconditional-dendritic` or `unbiased-unconditional-3layers` or `unbiased-unconditional-4layers`
    model_name = f'unconditional-concat'  # `unconditional-concat` or `unconditional-dendritic` or `unbiased-unconditional-3layers` or `unbiased-unconditional-4layers`
    model_number = 15
    num_steps = 100
    # hidden_cfg = {'3Layer': [25, 25], '4Layer': [64, 25, 10]} 
    
    # list of ints for the unconditional-dendritic model
    # num_hidden = [64,25,10]  # v1
    # num_hidden = [128, 64]  # v2
    # num_hidden = [128, 64, 16]  # v3
    # num_hidden = [64, 32, 16, 8]  # v4
    # num_hidden = [64, 25, 10]  # v6, v8
    # num_hidden = [25, 25, 25]  # v9
    # num_hidden = [50, 50]  # v10
    # num_hidden = [64, 10]  # v11
    # num_hidden = [128, 5]  # v12
    # num_hidden = [128, 20]  # v13
    # num_hidden = [128, 32, 16]  # v14
    # num_hidden = [5, 5, 5, 5, 5, 5] # with relu again
    # num_hidden = [4, 4, 4, 4, 3, 3, 2] # with relu again
    # num_hidden = [2, 2, 2, 2, 2, 2, 2] 
    # num_hidden = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    
    # num_hidden = [3, 3, 3, 3, 3, 3, 4]
    
    # integer for the unconditional-concat model
    num_hidden = 256

    # num_samples = int(1e3)
    num_samples = int(4e3)  # for the unconditional-concat model
    # num_epochs = int(6000 * int(1e4)/num_samples)  # epochs scales inversely with num of samples
    # num_epochs = int(5e4)  # epochs scales inversely with num of samples
    # num_epochs = int(2e5)  # this is for the unconditional-concat model, doesn't need more than this
    num_epochs = int(2e4)  # this is for the unconditional-concat model, doesn't need more than this
    batch_size = 128
    learning_rate = 3e-4
    # manifold_type = 'line'  # swiss_roll_3d
    manifold_type = 'trimodal'  # swiss_roll_3d
    # manifold_offsets = [0, 0]
    # manifold_offsets = [[0,0], [4,0], [2,4]]  # original offset. manifolds are separated
    # manifold_offsets = [[0,0], [1,0], [1,1]]  # manifolds lie on top of each other
    manifold_offsets = [[0.5,0.5], [0,1.5], [2,1]]
    num_manifold_dims = 1
    num_ambient_dims = 2
    manifold_rotation_angle = np.pi/6
    
    # load the weights of a previously trained model
    pretrained_model = {
        'use_pretrained_model_weights': False, 
        'model_name': 'unconditional-concat', 
        'model_num': 13
    }


    # ----------------------------- other parameters ----------------------------- #
    mode = 'train'
    log_folder = os.path.join('cluster', 'logs/%j')

    # -------------------------- save model description -------------------------- #
    description = {
        'model_number': model_number,
        'model_name': model_name,
        'num_steps': num_steps,
        'num_hidden': num_hidden,
        # 'num_hidden': num_hidden,
        # 'num_classes': num_classes,
        'num_samples': f'{num_samples:.0e}',
        'num_epochs': f'{num_epochs:.0e}',
        # 'manifold_type': manifold_type,
        'manifold_offsets': manifold_offsets,
        # 'num_manifold_dims': num_manifold_dims,
        'num_ambient_dims': num_ambient_dims,
        # 'manifold_rotation_angle': manifold_rotation_angle,
        'batch_size': batch_size,
        'learning_rate': f'{learning_rate:.0e}',
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
        mem_gb=32,
        timeout_min=120,
    )

    jobs = []
    with ex.batch():
        if mode=='train':
            job = ex.submit(train_model, 
                            model_name, 
                            num_steps, 
                            num_hidden, 
                            num_samples, 
                            num_epochs, 
                            batch_size, 
                            learning_rate, 
                            manifold_type,
                            manifold_offsets,
                            num_ambient_dims,
                            manifold_rotation_angle,
                            pretrained_model
            )
        jobs.append(job)
    print('all jobs submitted!')

    idx = 0
    print(f'Job {jobs[idx].job_id}')
    # idx += 1


if __name__ == '__main__':
    main()
