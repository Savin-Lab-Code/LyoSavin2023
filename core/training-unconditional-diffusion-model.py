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
                    lr,
                    device,
                    dataset,
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
    
    # define model
    if pretrained_model['use_pretrained_model_weights']:
        from utils import load_model_weights
        pretrained_model_name = pretrained_model['model_name']
        pretrained_model_num = pretrained_model['model_num']
        print(f'taking weights from pretrained model {pretrained_model_name}_{pretrained_model_num}!')
        model = load_model_weights(model, pretrained_model_name, pretrained_model_num, device)
    model.to(device)

    # training parameteres
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
            loss = noise_estimation_loss(model, batch_x, num_steps, alphas_bar_sqrt, one_minus_alphas_prod_sqrt, device, norm='l2', has_class_label=False)
            # zero the gradients
            optimizer.zero_grad()
            # backward pass: compute the gradient of the loss wrt the parameters
            loss.backward()
            # call the step function to update the parameters
            optimizer.step()
            
        if t % int(1e4) == 0:
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
                num_steps, 
                forward_schedule,
                num_hidden, 
                dataset_size, 
                epochs, 
                batch_size, 
                lr, 
                manifold_type, 
                num_ambient_dims, 
                pretrained_model,
                manifold_offsets=[], 
                manifold_rotation_angle=0, 
                ):
    
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)
    sys.path.append(os.path.join(base_dir, 'core'))
    sys.path.append(os.path.join(base_dir, 'core/utils'))
    
    from utils import save_model_weights
    from models import NoiseConditionalEstimatorConcat, VariableDendriticCircuit
    from dataset_utils import load_trimodal_data, load_unimodal_data, load_unimodal_data_3d, load_unimodal_data_nd, generate_2d_swiss_roll

    # ------------------------------ define dataset ------------------------------ #
    dataset = generate_2d_swiss_roll(dataset_size, rescaled=True, return_as_tensor=True)[1]

    # -------------------------------- load model -------------------------------- #
    model = VariableDendriticCircuit(hidden_cfg=num_hidden, num_in=num_ambient_dims, num_out=num_ambient_dims, bias=True)
    # model = NoiseConditionalEstimatorConcat(num_hidden)
    
    # -------------------- TRAINING - reverse diffusion process ------------------ #
    model = reverse_process(model, model_name, model_number, num_steps, forward_schedule, num_hidden, num_ambient_dims, epochs, batch_size, lr, device, dataset, pretrained_model)
    
    save_model_weights(model, model_name, model_number)

def main():
    print('we are running!')

    # -------------------------- set model parameters -------------------------- #
    model_name = 'unconditional-dendritic'
    model_number = 67
    # model_name = 'unconditional-concat'
    # model_number = 17
    num_steps = 2000
    forward_schedule = 'sigmoid'
    # num_hidden = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3]
    num_hidden = [3, 3, 3, 3, 3, 3, 4]
    # num_hidden = [8, 8, 7, 7]
    # num_hidden = [59, 59]
    # num_hidden = 128
    num_ambient_dims = 2
    num_epochs = 15e5
    manifold_type = 'swiss_roll'
    manifold_noise_amount = 0
    dataset_size = int(2e3)
    batch_size = 128
    learning_rate = 3e-4
    pretrained_model = {
        'use_pretrained_model_weights': True,
        'model_name': 'unconditional-dendritic',
        'model_num': 62
    }

    # -------------------------- save model description -------------------------- #
    description = {
        'model_name': model_name,
        'model_number': model_number,
        'num_steps': num_steps,
        'forward_schedule': forward_schedule,
        'num_hidden': num_hidden,
        'num_ambient_dims': num_ambient_dims,
        'num_epochs': f'{num_epochs:.0e}',
        'manifold_type': manifold_type,
        'manifold_noise_amount': manifold_noise_amount,
        'dataset_size': f'{dataset_size:.0e}',
        'batch_size': batch_size,
        'learning_rate': f'{learning_rate:.0e}',
        'use_pretrained_model': pretrained_model['use_pretrained_model_weights'],
    }
    if pretrained_model['use_pretrained_model_weights']:
        description['pretrained_model_name'] = pretrained_model['model_name']
        description['pretrained_model_num'] = pretrained_model['model_num']

    json_savedir = os.path.join(base_dir, 'core/model_description')
    model_name_and_number = f'{model_name}_{model_number}'
    json_name = f'{model_name_and_number}.json'
    with open(os.path.join(json_savedir, json_name), 'w') as file:
        json.dump(description, file)

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
        mem_gb = 64,
        timeout_min = 2000,  # 33 hours
    )

    jobs = []
    with ex.batch():
        job = ex.submit(train_model, 
                        model_name, 
                        model_number,
                        num_steps, 
                        forward_schedule,
                        num_hidden, 
                        dataset_size, 
                        num_epochs, 
                        batch_size, 
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
