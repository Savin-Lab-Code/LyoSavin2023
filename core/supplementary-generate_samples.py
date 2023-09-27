import submitit
import os, sys
import numpy as np
import torch
import time
from tqdm import trange
import zarr

project_root = os.path.abspath("")  # alternative
if project_root[-12:] == 'LyoSavin2023':
    base_dir = project_root
else:
    base_dir = os.path.dirname(project_root)
    
def generate_samples(save_dir, batch_idx, num_runs, sampling_method, distribution_type, sample_size, manifold_type='unimodal', model_name=None, model_num=None, v=None, eval_method=None, s_bu=None, s_td=None, posterior_type=None, label=2, eval_epoch=None):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)
    sys.path.append(os.path.join(base_dir, 'core'))
    sys.path.append(os.path.join(base_dir, 'core/utils'))
    from pathlib import Path
    
    print('sample_size is:', sample_size)

    # ------------------------------- select model ------------------------------- #
    from utils import select_model
    if model_name==None or model_num==None:
        print('no model name or num provided, using default model')
        if manifold_type == 'unimodal':
            prior_sampler, num_steps, ambient_dims = select_model('unconditional-dendritic', 47)
            normalized_beta_schedule = True
            schedule='sigmoid'
            save_dir = os.path.join(save_dir, 'unconditional-dendritic-47')
        if manifold_type == 'trimodal':
            prior_sampler, num_steps, ambient_dims = select_model('unconditional-dendritic', 42)
            normalized_beta_schedule = True
            schedule='sigmoid'
            save_dir = os.path.join(save_dir, 'unconditional-dendritic_42')
    else:
        if eval_epoch==None:
            print(f'using specific model: {model_name}_{model_num}')
            prior_sampler, num_steps, ambient_dims = select_model(model_name, model_num)
            normalized_beta_schedule = False
            schedule='sine'
            save_dir = os.path.join(save_dir, f'{model_name}_{model_num}')
        else:
            print(f'using specific model: {model_name}_{model_num}, from epoch={eval_epoch}')
            from utils import load_model_weights_from_chkpt
            prior_sampler, num_steps, ambient_dims = load_model_weights_from_chkpt(model_name, model_num, eval_epoch)
            normalized_beta_schedule = False
            schedule = 'sine'
            save_dir = os.path.join(save_dir, f'{model_name}_{model_num}', 'by_checkpoints', f'{eval_epoch}')
            

    prior_sampler.to(device)
    prior_sampler.eval()
    
    # ----------------------- define continuous likelihood ----------------------- #
    if distribution_type == 'posterior':
        if manifold_type == 'unimodal':
            v = np.array([[4, 1]]).T
            likelihood_sigma = 0.4
        elif manifold_type == 'trimodal':
            v = np.array([[2, -5]]).T
            likelihood_sigma = 0.5
        if s_bu == None:
            s_bu = 1
        if s_td == None:
            s_td = 0.1
    
    # -------- initial datapoint on the manifold (for sequential sampling) ------- #
    if sampling_method == 'seq':
        if manifold_type == 'unimodal':
            print('unimodal manifold.')
            from dataset_utils import generate_2d_swiss_roll
            dataset = generate_2d_swiss_roll(sample_size, rescaled=True, return_as_tensor=True)[1]
            manifold_initial_point = dataset[np.random.randint(sample_size)].reshape(1, -1)
            manifold_initial_point = manifold_initial_point.to(device)
        if manifold_type == 'trimodal':
            print('trimodal manifold.')
            from dataset_utils import load_trimodal_data
            offsets = [[0,0], [4,0], [2,4]]
            dataset = load_trimodal_data(sample_size, offsets, noise=0)[:, :2]
            manifold_initial_point = dataset[np.random.randint(sample_size)].reshape(1, -1)
            manifold_initial_point = manifold_initial_point.to(device)
            
    # -------------------------------- eval method ------------------------------- #
    if eval_method == 'xt':
        print('evaluating likelihood score g_t at x_t.')
        eval_at_mean = False
    elif eval_method == 'mu':
        print('evaluating likelihood score g_t at mu.')
        eval_at_mean = True

    # ----------------------------- generate prior samples ----------------------------- #
    if distribution_type == 'prior':
        print('sampling from the prior.')
        if sampling_method == 'iid':
            print('sampling method is iid.')
            from prior_utils import p_sample_loop
            save_dir = os.path.join(save_dir, f'{distribution_type}-{sampling_method}-{manifold_type}/num_samples={sample_size:.0g}-num_runs={num_runs:.0g}-batch_idx={int(batch_idx)}')
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            for run_idx in trange(num_runs, desc='run number'):
                x_rev = p_sample_loop(prior_sampler, (sample_size, ambient_dims), num_steps, device, normalized_beta_schedule=normalized_beta_schedule, schedule=schedule)
                x_rev = x_rev.numpy()
                zarr.save(os.path.join(save_dir, f'x_revs-run_num={run_idx}.zarr'), x_rev)
            
        elif sampling_method == 'seq':
            print('sampling method is seq.')
            from prior_utils import sequential_prior_sampler
            save_dir = os.path.join(save_dir, f'{distribution_type}-{sampling_method}-{manifold_type}/num_samples={sample_size:.0g}-num_runs={num_runs:.0g}-batch_idx={int(batch_idx)}')
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            for run_idx in trange(num_runs, desc='run number'):
                _, x_fwd, x_rev = sequential_prior_sampler(prior_sampler, manifold_initial_point, sample_size, num_steps, disable_tqdm=True, normalized_beta_schedule=normalized_beta_schedule, schedule=schedule)
                x_fwd = x_fwd.numpy()
                x_rev = x_rev.numpy()
                zarr.save(os.path.join(save_dir, f'x_rev-run_num={run_idx}.zarr'), x_rev)
                zarr.save(os.path.join(save_dir, f'x_fwd-run_num={run_idx}.zarr'), x_fwd)

        
    # ----------------------------- generate posterior samples ----------------------------- #
    elif distribution_type == 'posterior':
        print(f'sampling from the posterior. Posterior type is {posterior_type}')
        classifier = None
        if posterior_type=='td' or posterior_type=='both':
            classifier = select_model('noisy-image-classifier-with-noise-info', 2)[0]
            classifier.to(device)
            classifier.eval()
        
        if sampling_method == 'iid':
            print('sampling method is iid.')
            from likelihood_utils import perform_variable_inference
            save_dir = os.path.join(save_dir, f'{distribution_type}-{sampling_method}-{manifold_type}-{posterior_type}-{eval_method}/num_samples={sample_size:.0g}-num_runs={num_runs:.0g}-batch_idx={int(batch_idx)}')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
            for run_idx in trange(num_runs, desc='run number'):
                x_rev, label = perform_variable_inference(prior_sampler, classifier, v, posterior_type, label, likelihood_sigma, s_bu, s_td, num_steps, sample_size, device, normalized_beta_schedule, eval_at_mean, schedule=schedule)
                zarr.save(os.path.join(save_dir, f'x_rev-run_num={run_idx}.zarr'), x_rev)
            
        elif sampling_method == 'seq':
            print('sampling method is seq.')
            from likelihood_utils import variable_neural_inference
            save_dir = os.path.join(save_dir, f'{distribution_type}-{sampling_method}-{manifold_type}-{posterior_type}-{eval_method}/num_samples={sample_size:.0g}-num_runs={num_runs:.0g}-batch_idx={int(batch_idx)}')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            for run_idx in trange(num_runs, desc='run number'):
                _, x_fwd, x_rev, label = variable_neural_inference(prior_sampler, classifier, v, manifold_initial_point, posterior_type, label, likelihood_sigma, s_bu, s_td, num_steps, sample_size, device, normalized_beta_schedule, eval_at_mean, disable_tqdm=False, schedule=schedule)
                zarr.save(os.path.join(save_dir, f'x_rev-run_num={run_idx}.zarr'), x_rev)
                zarr.save(os.path.join(save_dir, f'x_fwd-run_num={run_idx}.zarr'), x_fwd)

def main():
    log_folder = os.path.join(base_dir, 'core/cluster/logs/generate_samples', '%j')
    # device = 'cpu'

    # ------------------------- submitit cluster executor ------------------------ #
    ex = submitit.AutoExecutor(folder=log_folder)
    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f'!!! Slurm executable `srun` not found. Will execute jobs on "{ex.cluster}"')

    # ex.update_parameters(
    #     slurm_job_name = 'gen_samples',
    #     nodes = 1,
    #     slurm_partition = 'ccn',
    #     slurm_cpus_per_task = 4,
    #     slurm_ntasks_per_node = 1,
    #     mem_gb = 512,
    #     timeout_min = 1440
    # )

    # slurm parameters
    ex.update_parameters(
        slurm_job_name = 'generate',
        nodes = 1,
        slurm_partition = 'gpu',
        slurm_gpus_per_task=1,
        slurm_ntasks_per_node=1,
        slurm_constraint='a100',
        cpus_per_task=12,
        mem_gb=512,
        timeout_min=3000,
    )
    
    # ----------------------------- model parameters ----------------------------- #
    # sample_size = int(1e3)
    sample_size = int(5e3)
    batch_size = 10  # the number of jobs
    num_runs = 1  # how many repeats of the data collection per job

    print('batch_size is:', batch_size)
    print('num runs is:', num_runs)


    
    distribution_types = ['posterior']  # ['prior', 'posterior']
    sampling_methods = ['iid', 'seq']  # ['iid', 'seq']
    
    # only for posterior distribution
    manifold_types = ['unimodal']  # ['unimodal', 'trimodal']
    posterior_types = ['bu']  # ['bu', 'td', 'both', 'neither']
    eval_methods = ['xt', 'mu']  # ['xt', 'mu']
    

    # # specify models
    # eval_epochs = [i for i in range(0, int(15e5), int(1e5))]
    # eval_epochs.append(int(15e5-1e4))

    # layers = [2, 4, 5, 6, 7, 10]
    # model_names = []
    # for l in layers:
    #     model_name = f'unconditional-dendritic-{l}-layers'
    #     model_names.append(model_name)

    # model_nums = [1, 2, 3, 4]

    model_name = 'unconditional-dendrtici-4-layers'
    model_num = 5

    
    # ------------------------------- save location ------------------------------- #
    save_dir = os.path.join(base_dir, 'core', 'saved_arrays', 'samples')
    from pathlib import Path
    Path(save_dir).mkdir(parents=False, exist_ok=True)

    # ----------------------------------- jobs ----------------------------------- #
    jobs = []
    with ex.batch():
        for distribution_type in distribution_types:
            if distribution_type == 'prior':
                for sampling_method in sampling_methods:
                    for batch_idx in range(batch_size):
                        for model_name in model_names:
                            for model_num in model_nums:
                                for eval_epoch in eval_epochs:
                                    job = ex.submit(generate_samples, 
                                                    save_dir,
                                                    batch_idx,
                                                    num_runs, 
                                                    sampling_method, 
                                                    distribution_type, 
                                                    sample_size, 
                                                    model_name=model_name,
                                                    model_num=model_num,
                                                    eval_epoch=eval_epoch
                                                    )
                                    jobs.append(job)
            elif distribution_type == 'posterior':
                for sampling_method in sampling_methods:
                    for manifold_type in manifold_types:
                        for eval_method in eval_methods:
                            for posterior_type in posterior_types:
                                for batch_idx in range(batch_size):
                                    job = ex.submit(generate_samples, 
                                                    save_dir,
                                                    batch_idx,
                                                    num_runs, 
                                                    sampling_method, 
                                                    distribution_type, 
                                                    sample_size, 
                                                    manifold_type=manifold_type, 
                                                    model_name=model_name,
                                                    model_num=model_num,
                                                    eval_method=eval_method, 
                                                    posterior_type=posterior_type
                                                    )
                                    jobs.append(job)
    for idx in range(len(jobs)):
        print(f'Job {jobs[idx].job_id}')
    print('all jobs submitted!')

if __name__ == '__main__':
    main()