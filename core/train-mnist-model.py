import submitit
import os, sys
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import time

project_root = os.path.abspath("")  # alternative
if project_root[-12:] == 'LyoSavin2023':
    base_dir = project_root
else:
    base_dir = os.path.dirname(project_root)


def train(num_epochs, model, train_loader, description, optimizer, tb, device):
    from image_utils import calculate_loss
    
    for epoch in tqdm(range(1, int(num_epochs) + 1), total=int(num_epochs), desc='Training model', unit='epochs', miniters=int(num_epochs)/100, maxinterval=float("inf")):
        # start training    
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            
            # compute the loss
            loss = calculate_loss(model, data, n_steps=description['num_steps'], forward_schedule=description['forward_schedule'], norm='l2', device=device)
            
            # zero the gradients
            optimizer.zero_grad()
            # backward pass: compute the gradient of the loss wrt the parameters
            loss.backward()
            # call the step function to update the parameters
            optimizer.step()
            
            # write to tensorboard
            tb.add_scalar('Loss', loss.item(), epoch+batch_idx)
    return model


def train_model():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)
    
    sys.path.append(os.path.join(base_dir, 'core'))
    sys.path.append(os.path.join(base_dir, 'core/utils'))
    
    # -------------------------------- description ------------------------------- #
    
    description = {
        # 'model_name': 'fc-mnist',
        'model_name': 'dendritic-mnist',
        'model_number': 33,
        'batch_size': 512,
        'lr': 3e-4,
        'num_epochs': 5e3,
        'forward_schedule': 'sigmoid',
        'hidden_cfg': [150],
        'num_steps': 100,
        'num_ambient_dims': 28 * 28,
        'manifold_type': 'mnist',
        'optimizer': 'Adam',
        'classes': 'all',
    }
    
    pretrained = True
    if pretrained:
        description['pretrained'] = True
        description['pretrained_model_name'] = 'dendritic-mnist'
        description['pretrained_model_num'] = 26
    else:
        description['pretrained'] = False
        
    # save description
    with open(os.path.join(base_dir, 'core', 'model_description', f'{description["model_name"]}_{description["model_number"]}.json'), 'w') as f:
        json.dump(description, f)


    # ---------------------------------- dataset --------------------------------- #
    mnist_dir = os.path.join(base_dir, 'core', 'datasets', 'mnist')
    mnist_train = datasets.MNIST(mnist_dir, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]))

    from image_utils import rescale_to_neg_one_to_one   
    test = rescale_to_neg_one_to_one(mnist_train[0][0])

    # select digits
    if description['classes'] != 'all':
        indices = (mnist_train.targets == 0) | (mnist_train.targets == 1) | (mnist_train.targets == 2)
        mnist_train.data, mnist_train.targets = mnist_train.data[indices], mnist_train.targets[indices]

    train_loader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=description['batch_size'], shuffle=True)

    
    # ------------------------------- define model ------------------------------- #
    from models import VariableDendriticCircuit, FullyConnectedNetwork
    if description['model_name'] == 'dendritic-mnist':
        model = VariableDendriticCircuit(hidden_cfg=description['hidden_cfg'], 
                                        num_in=description['num_ambient_dims'], 
                                        num_out=description['num_ambient_dims'], 
                                        bias=True)
    elif description['model_name'] == 'fc-mnist':
        model = FullyConnectedNetwork(n_dim_data=description['num_ambient_dims'], num_hidden=description['hidden_cfg'], num_hidden_layers=description['num_hidden_layers'])
    
    model = model.to(device)
    
    from utils import count_parameters
    c = count_parameters(model)
    print(f'{c:.3g}')
    description['num_params'] = f'{c:.3g}'
    
    if description['pretrained']==True:
        from utils import load_model_weights
        pretrained_model_name = description['pretrained_model_name']
        pretrained_model_num = description['pretrained_model_num']
        print(f'taking weights from pretrained model {pretrained_model_name}_{pretrained_model_num}!')
        model = load_model_weights(model, pretrained_model_name, pretrained_model_num, device)


    # ------------------------------- train model ------------------------------- #
    optimizer = optim.Adam(model.parameters(), lr=description['lr'])
    # optimizer = optim.SGD(model.parameters(), lr=description['lr'])
    
    run_dir = os.path.join(base_dir, 'demos/runs', f'{description["model_name"]}_{description["model_number"]}')
    tb = SummaryWriter(run_dir, flush_secs=1)
    start_time = time.time()

    model.train()
    model = train(description['num_epochs'], model, train_loader, description, optimizer, tb, device)

    end_time = time.time()
    total_time = end_time-start_time
    print(f'training took {total_time/60:.0f} minutes or {total_time/60:.1f} hours', flush=True)
    tb.flush()

    from utils import save_model_weights
    save_model_weights(model, description['model_name'], description['model_number'])
    
    
    

def main():
    # ------------------------- submitit cluster executor ------------------------ #
    log_folder = os.path.join(base_dir, 'core/cluster/logs/train-mnist', '%j')
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
        mem_gb = 160,
        timeout_min = 60*20,  # 10 hours
    )

    jobs = []
    with ex.batch():
        job = ex.submit(train_model)
        jobs.append(job)
    print('all jobs submitted!')

    idx = 0
    print(f'Job {jobs[idx].job_id}')

if __name__ == '__main__':
    main()
