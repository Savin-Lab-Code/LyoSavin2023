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
    
def generate_samples(save_dir, batch_idx, num_runs, sampling_method, distribution_type, sample_size, manifold_type='unimodal', v=None, eval_method=None, s_bu=None, s_td=None, posterior_type=None, label=2):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)
    sys.path.append(os.path.join(base_dir, 'core'))
    sys.path.append(os.path.join(base_dir, 'core/utils'))
    from pathlib import Path
    
    # ------------------------------- select model ------------------------------- #
    from utils import select_model
    if manifold_type == 'unimodal':
        prior_sampler, num_steps, ambient_dims = select_model('unconditional-dendritic', 47)
        normalized_beta_schedule = True
    if manifold_type == 'trimodal':
        prior_sampler, num_steps, ambient_dims = select_model('unconditional-dendritic', 42)
        normalized_beta_schedule = True
    prior_sampler.to(device)
    
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
            from dataset_utils import generate_2d_swiss_roll
            dataset = generate_2d_swiss_roll(sample_size, rescaled=True, return_as_tensor=True)[1]
            manifold_initial_point = dataset[np.random.randint(sample_size)].reshape(1, -1)
            manifold_initial_point = manifold_initial_point.to(device)
        if manifold_type == 'trimodal':
            from dataset_utils import load_trimodal_data
            offsets = [[0,0], [4,0], [2,4]]
            dataset = load_trimodal_data(sample_size, offsets, noise=0)[:, :2]
            manifold_initial_point = dataset[np.random.randint(sample_size)].reshape(1, -1)
            manifold_initial_point = manifold_initial_point.to(device)
            
    # -------------------------------- eval method ------------------------------- #
    if eval_method == 'xt':
        eval_at_mean = False
    elif eval_method == 'mu':
        eval_at_mean = True

    # ----------------------------- generate prior samples ----------------------------- #
    if distribution_type == 'prior':    
        if sampling_method == 'iid':
            x_rev = p_sample_loop(model, (sample_size, ambient_dims), num_steps, device, normalized_beta_schedule=normalized_beta_schedule)
            
            save_dir = os.path.join(save_dir, f'{distribution_type}-{sampling_method}-{manifold_type}/num_samples={sample_size:.0g}-num_runs={num_runs:.0g}')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            zarr.save(os.path.join(save_dir, f'x_revs.zarr'), x_rev)
            
        elif sampling_method == 'seq':
            from prior_utils import sequential_prior_sampler
            _, x_fwd, x_rev = sequential_prior_sampler(model, manifold_initial_point, sample_size, num_steps, disable_tqdm=True, normalized_beta_schedule=normalized_beta_schedule)
            
            save_dir = os.path.join(save_dir, f'{distribution_type}-{sampling_method}-{manifold_type}/num_samples={sample_size:.0g}-num_runs={num_runs:.0g}-batch_idx={int(batch_idx)}')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            zarr.save(os.path.join(save_dir, f'x_revs.zarr'), x_rev)
            zarr.save(os.path.join(save_dir, f'x_fwds.zarr'), x_fwd)
        
        
    # ----------------------------- generate posterior samples ----------------------------- #
    elif distribution_type == 'posterior':
        classifier = select_model('noisy-image-classifier-with-noise-info', 2)[0]
        
        if sampling_method == 'iid':
            from likelihood_utils import perform_variable_inference
            save_dir = os.path.join(save_dir, f'{distribution_type}-{sampling_method}-{manifold_type}-{posterior_type}-{eval_method}/num_samples={sample_size:.0g}-num_runs={num_runs:.0g}-batch_idx={int(batch_idx)}')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            x_revs = []
            for run_idx in trange(num_runs, desc='run number'):
                x_rev, label = perform_variable_inference(prior_sampler, classifier, v, posterior_type, label, likelihood_sigma, s_bu, s_td, num_steps, sample_size, device, normalized_beta_schedule, eval_at_mean)
                x_revs.append(x_rev)
            x_revs = np.stack(x_revs, axis=0)
            zarr.save(os.path.join(save_dir, f'x_revs.zarr'), x_revs)
            
        elif sampling_method == 'seq':
            from likelihood_utils import variable_neural_inference
            save_dir = os.path.join(save_dir, f'{distribution_type}-{sampling_method}-{manifold_type}-{posterior_type}-{eval_method}/num_samples={sample_size:.0g}-num_runs={num_runs:.0g}-batch_idx={int(batch_idx)}')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            x_fwds = []; x_revs = []
            for run_idx in trange(num_runs, desc='run number'):
                _, x_fwd, x_rev, label = variable_neural_inference(prior_sampler, classifier, v, manifold_initial_point, posterior_type, label, likelihood_sigma, s_bu, s_td, num_steps, sample_size, device, normalized_beta_schedule, eval_at_mean, disable_tqdm=True)
                x_fwds.append(x_fwd)
                x_revs.append(x_rev)
            x_fwds = np.stack(x_fwds, axis=0)
            x_revs = np.stack(x_revs, axis=0)
            zarr.save(os.path.join(save_dir, f'x_revs.zarr'), x_revs)
            zarr.save(os.path.join(save_dir, f'x_fwds.zarr'), x_fwds)

def main():
    log_folder = os.path.join(base_dir, 'core/cluster/logs/generate_samples', '%j')
    # device = 'cpu'

    # ------------------------- submitit cluster executor ------------------------ #
    ex = submitit.AutoExecutor(folder=log_folder)
    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f'!!! Slurm executable `srun` not found. Will execute jobs on "{ex.cluster}"')

    ex.update_parameters(
        slurm_job_name = 'gen_samples',
        nodes = 1,
        slurm_partition = 'ccn',
        slurm_cpus_per_task = 4,
        slurm_ntasks_per_node = 1,
        mem_gb = 32,
        timeout_min = 300,
    )
    
    # ----------------------------- model parameters ----------------------------- #
    # sample_size = int(1e3)
    batch_size = 20 # 20
    sample_size = 50
    num_runs = 200 # 200
    
    '''
    distribution_types = ['prior', 'posterior']
    sampling_methods = ['iid', 'seq']
    
    # only for posterior distribution
    manifold_types = ['unimodal', 'trimodal']
    eval_methods = ['xt', 'mu']
    posterior_types = ['bu', 'td', 'both', 'neither']
    '''
    
    distribution_types = ['posterior']
    sampling_methods = ['iid', 'seq']
    
    # only for posterior distribution
    manifold_types = ['unimodal']
    eval_methods = ['xt']
    posterior_types = ['bu']
    
        
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
                        job = ex.submit(generate_samples, 
                                        save_dir,
                                        batch_idx,
                                        num_runs, 
                                        sampling_method, 
                                        distribution_type, 
                                        sample_size, 
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
                                                    eval_method=eval_method, 
                                                    posterior_type=posterior_type
                                                    )
                                    jobs.append(job)
    for idx in range(len(jobs)):
        print(f'Job {jobs[idx].job_id}')
    print('all jobs submitted!')

if __name__ == '__main__':
    main()