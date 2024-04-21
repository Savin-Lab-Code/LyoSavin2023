import submitit
import os, sys
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import time

project_root = os.path.abspath("")  # alternative
if project_root[-12:] == 'LyoSavin2023':
    base_dir = project_root
else:
    base_dir = os.path.dirname(project_root)


def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, loss, model_name, model_number):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss
    }
    save_path = os.path.join(base_dir, 'core/saved_weights', f'{model_name}_{model_number}', f'epoch={epoch}')
    from pathlib import Path
    Path(save_path).mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_path, 'checkpoint.pt'))

def reverse_process(model, 
                    model_name, 
                    model_number, 
                    num_steps, 
                    forward_schedule,
                    num_hidden, 
                    num_dims,
                    num_epochs,
                    batch_size,
                    optimizer_type,
                    lr,
                    device,
                    dataset,
                    l1_reg,
                    l1_on_basal_only,
                    pretrained_model):
    
    # beta-related parameters
    from prior_utils import forward_process
    from utils import noise_estimation_loss
    
    coefs = forward_process(num_steps, device, forward_schedule)
    betas, alphas, alphas_prod, alphas_prod_p, alphas_bar_sqrt, one_minus_alphas_prod_log, one_minus_alphas_prod_sqrt = coefs
    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_prod_sqrt = one_minus_alphas_prod_sqrt.to(device)
    
    # training set
    dataset = dataset.to(device)
    
    print('model_name:', model_name)
    print('model_number:', model_number)
    print('num_steps:', num_steps)
    print('forward_schedule:', forward_schedule)
    print('num_hidden:', num_hidden)
    print('num_epochs:', num_epochs)
    print('dataset shape:', dataset.shape)
    print('l1_on_basal_only', l1_on_basal_only)
    
    # define model
    if pretrained_model['use_pretrained_model_weights']:
        if pretrained_model['use_checkpoint_weights']==False:
            from utils import load_model_weights
            pretrained_model_name = pretrained_model['model_name']
            pretrained_model_num = pretrained_model['model_num']
            print(f'taking weights from pretrained model {pretrained_model_name}_{pretrained_model_num}!')
            model = load_model_weights(model, pretrained_model_name, pretrained_model_num, device)
        elif pretrained_model['use_checkpoint_weights']==True:
            from utils import load_model_weights_from_chkpt
            model, num_steps, ambient_dims = load_model_weights_from_chkpt(pretrained_model['model_name'], pretrained_model['model_num'], epoch_number=pretrained_model['checkpoint_epoch'], device=device)
            print('model weights loaded from checkpoint!', flush=True)
            
    model.to(device)

    # training parameteres
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
    if pretrained_model['use_pretrained_model_weights'] and pretrained_model['use_checkpoint_weights']==True:
        from utils import load_optimizer_state_dict
        optimizer = load_optimizer_state_dict(optimizer, pretrained_model['model_name'], pretrained_model['model_num'], epoch_number=pretrained_model['checkpoint_epoch'], device=device)
        print('optimizer state dict loaded from checkpoint!', flush=True)

    run_dir = os.path.join(base_dir, 'demos/runs', f'{model_name}_{model_number}')
    tb = SummaryWriter(run_dir)
    start_time = time.time()
    
    # start training
    model.train()
    for t in tqdm(range(int(num_epochs)), total=int(num_epochs), desc='Training model', unit='epochs', miniters=int(num_epochs)/1000, maxinterval=float("inf")):
        permutation = torch.randperm(dataset.size()[0], device=device)
    
        for i in range(0, dataset.size()[0], batch_size):
            # retrieve current batch
            indices = permutation[i:i+batch_size]
            batch_x = dataset[indices]
            
            # compute the loss
            loss = noise_estimation_loss(model, batch_x, num_steps, alphas_bar_sqrt, one_minus_alphas_prod_sqrt, device, norm='l2', l1_reg=l1_reg, l1_on_basal_only=l1_on_basal_only, has_class_label=False)
            # zero the gradients
            optimizer.zero_grad()
            # backward pass: compute the gradient of the loss wrt the parameters
            loss.backward()
            # call the step function to update the parameters
            optimizer.step()
        
        if t < int(5e5):
            if t % int(1e4) == 0:
                save_checkpoint(t, model.state_dict(), optimizer.state_dict(), loss.item(), model_name, model_number)
        else:
            if t % int(1e5) == 0:
                save_checkpoint(t, model.state_dict(), optimizer.state_dict(), loss.item(), model_name, model_number)
        
        # write to tensorboard
        tb.add_scalar('Loss', loss.item(), t)
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
    

def train_model(model_name, 
                model_number,
                has_skip_connections,
                num_steps, 
                forward_schedule,
                num_hidden, 
                dataset_size, 
                l1_reg,
                l1_on_basal_only,
                bias,
                epochs, 
                batch_size, 
                optimizer_type,
                lr,
                manifold_type, 
                num_ambient_dims, 
                pretrained_model,
                manifold_offsets=[], 
                manifold_rotation_angle=np.pi/4, 
                ):
    
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)
    
    sys.path.append(os.path.join(base_dir, 'core'))
    sys.path.append(os.path.join(base_dir, 'core/utils'))
    
    from utils import save_model_weights
    from models import NoiseConditionalEstimatorConcat, VariableDendriticCircuit, VariableDendriticCircuitWithSkipConnections
    from dataset_utils import load_trimodal_data, load_unimodal_data, load_unimodal_data_3d, load_unimodal_data_nd, generate_2d_swiss_roll

    # ------------------------------ define dataset ------------------------------ #
    if manifold_type == '2d_swiss_roll' and num_ambient_dims == 2:
        dataset = generate_2d_swiss_roll(dataset_size, rescaled=True, return_as_tensor=True)[1]
    elif manifold_type == "3d_swiss_roll" and num_ambient_dims == 10:
        # dataset = load_unimodal_data_nd(dataset_size, 'swiss_roll_3d', 10, rotation_angle=np.pi/4, noise=0, shrink_y_axis=True)
        dataset = load_unimodal_data_nd(dataset_size, 'swiss_roll_3d', num_ambient_dims, rotation_angle=manifold_rotation_angle, noise=0, shrink_y_axis=True)

    else:
        raise ValueError('Invalid manifold type or number of ambient dimensions!')

    # -------------------------------- load model -------------------------------- #
    print('has_skip_connections', has_skip_connections)
    if has_skip_connections:
        model = VariableDendriticCircuitWithSkipConnections(hidden_cfg=num_hidden, num_in=num_ambient_dims, num_out=num_ambient_dims, bias=bias)
    else:
        model = VariableDendriticCircuit(hidden_cfg=num_hidden, num_in=num_ambient_dims, num_out=num_ambient_dims, bias=bias)
    # model = NoiseConditionalEstimatorConcat(num_hidden)
    
    # -------------------- TRAINING - reverse diffusion process ------------------ #
    model = reverse_process(
        model, 
        model_name, 
        model_number, 
        num_steps, 
        forward_schedule, 
        num_hidden, 
        num_ambient_dims, 
        epochs, 
        batch_size, 
        optimizer_type, 
        lr, 
        device, 
        dataset, 
        l1_reg, 
        l1_on_basal_only, 
        pretrained_model
    )
    
    save_model_weights(model, model_name, model_number)
    
def construct_model_name(manifold_type, num_hidden, has_skip_connections, l1_lambda, l1_on_basal_only):
    model_name = f'unconditional-dendritic'
    
    if manifold_type == '2d_swiss_roll':
        model_name = model_name
    elif manifold_type == '3d_swiss_roll':
        model_name = f'{model_name}-3d_manifold-10d_ambient'
    
    model_name = f'{model_name}-{num_hidden}-layers'
    
    if has_skip_connections:
        model_name = f'{model_name}-with-skip-conns'
    
    model_name = f'{model_name}-l1-reg={l1_lambda}'
    model_name = f'{model_name}-l1_on_basal_only={l1_on_basal_only}'
    return model_name
    
def main():
    print('we are running!')

    # -------------------------- set model parameters -------------------------- #
    model_versions = [2]
    l1_regs = [1e-4, 1e-5, 1e-6, 0]  # how much to penalize the L1 norm of the weights of the Fully connected layer
    # l1_reg_on_phase_weights = False
    l1_on_basal_only = True
    # model_name = 'unconditional-concat'
    # model_version = 18
    num_steps = 100
    forward_schedule = 'sigmoid'
    has_skip_connections = False
    
    num_hiddens = [
        [2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
        [3, 3, 3, 3, 3, 3, 4],
        [4, 4, 4, 4, 4, 3],
        [5, 5, 5, 5, 5],
        [8, 8, 7, 7],
        [59, 59]
    ]
    
    bias = True
    num_epochs = 2e6+1
    # num_hidden = 128
    num_ambient_dims = 10
    # manifold_type = '2d_swiss_roll'
    manifold_type = '3d_swiss_roll'
    manifold_noise_amount = 0
    # manifold_rotation_angle = 'np.pi/4'
    dataset_size = int(2e3)
    batch_size = 512
    optimizer_type = 'adam'
    learning_rate = 3e-3
    
    pretrained_model_name = 'unconditional-concat'
    pretrained_model = {
        'use_pretrained_model_weights': False,
        'use_checkpoint_weights': False,
        'checkpoint_epoch': 1490000,
        'model_name': pretrained_model_name,
        'model_num': 1
    }

    # -------------------------- save model description -------------------------- #
    description = {
        'num_steps': num_steps,
        'forward_schedule': forward_schedule,
        'num_ambient_dims': num_ambient_dims,
        'num_epochs': f'{num_epochs:.1e}',
        'manifold_type': manifold_type,
        'manifold_noise_amount': manifold_noise_amount,
        # 'manifold_rotation_angle': manifold_rotation_angle,
        'dataset_size': f'{dataset_size:.0e}',
        'bias': bias,
        'batch_size': batch_size,
        'optimizer_type': optimizer_type,
        'learning_rate': f'{learning_rate:.0e}',
        'use_pretrained_model': pretrained_model['use_pretrained_model_weights'],
    }
    if pretrained_model['use_pretrained_model_weights']:
        description['pretrained_model_name'] = pretrained_model['model_name']
        description['pretrained_model_num'] = pretrained_model['model_num']
        if pretrained_model['use_checkpoint_weights']:
            description['pretrained_checkpoint_epoch'] = pretrained_model['checkpoint_epoch']

    # ------------------------- submitit cluster executor ------------------------ #
    log_folder = os.path.join(base_dir, 'core/cluster/logs/training_models', '%j')
    ex = submitit.AutoExecutor(folder=log_folder)
    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f'!!! Slurm executable `srun` not found. Will execute jobs on "{ex.cluster}"')
    
    # slurm parameters
    ex.update_parameters(
        slurm_job_name = 'training',
        nodes = 1,
        slurm_partition = 'gpu',
        slurm_constraint = 'a100',
        slurm_gpus_per_task = 1,
        slurm_cpus_per_task = 16,
        slurm_ntasks_per_node = 1,
        mem_gb = 128,
        timeout_min = 2400,  # 40 hours
    )

    jobs = []
    json_savedir = os.path.join(base_dir, 'core/model_description')
    with ex.batch():
        for num_hidden in num_hiddens:
            for l1_lambda in l1_regs:
                for model_version in model_versions:
                    
                    assert isinstance(model_versions, list)
                    model_name = construct_model_name(manifold_type, len(num_hidden), has_skip_connections, l1_lambda, l1_on_basal_only)

                    description['model_name'] = model_name
                    description['model_number'] = model_version
                    description['skip_connections'] = has_skip_connections
                    description['num_hidden'] = num_hidden
                    description['l1_regularization'] = l1_lambda
                    description['l1_on_basal_only'] = l1_on_basal_only
                    model_name_and_number = f'{model_name}_{model_version}'
                    json_name = f'{model_name_and_number}.json'
                    with open(os.path.join(json_savedir, json_name), 'w') as file:
                        json.dump(description, file)
                    
                    job = ex.submit(train_model, 
                                    model_name, 
                                    model_version,
                                    has_skip_connections,
                                    num_steps, 
                                    forward_schedule,
                                    num_hidden, 
                                    dataset_size, 
                                    l1_lambda,
                                    l1_on_basal_only,
                                    bias,
                                    num_epochs, 
                                    batch_size, 
                                    optimizer_type,
                                    learning_rate, 
                                    manifold_type,
                                    num_ambient_dims,
                                    pretrained_model,
                                    # manifold_offsets,
                                    # manifold_rotation_angle,
                    )
                    jobs.append(job)
    print('all jobs submitted!')

    idx = 0
    print(f'Job {jobs[idx].job_id}')
    # idx += 1

if __name__ == '__main__':
    main()
