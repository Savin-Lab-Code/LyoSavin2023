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
    
def generate_samples(sampling_method, eval_method, sample_size, M, likelihood_sigma, s, num_runs, run_idx):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    sys.path.append(os.path.join(base_dir, 'core'))
    sys.path.append(os.path.join(base_dir, 'core/utils'))
    
    from utils import select_model
    prior_sampler, num_steps, ambient_dims = select_model('unconditional-dendritic', 47)
    prior_sampler.to(device)
    
    if eval_method == 'xt':
        eval_at_mean = False
    elif eval_method == 'mu':
        eval_at_mean = True
        
    if sampling_method == 'iid':
        from likelihood_utils import posterior_sample_loop_occlusion
        samples_iid_reverse = posterior_sample_loop_occlusion(prior_sampler, M, likelihood_sigma, s, (sample_size, ambient_dims), num_steps, eval_at_mean=eval_at_mean, device=device)
        zarr.save(os.path.join(base_dir, f'core/saved_arrays/eval_loc/sample_size={sample_size:.0g}_runs={num_runs:.0g}', f'{sampling_method}_{eval_method}s_{int(run_idx)}.zarr'), samples_iid_reverse)
        
    elif sampling_method == 'seq':
        from likelihood_utils import sequential_posterior_sampler
        from dataset_utils import generate_2d_swiss_roll
        dataset = generate_2d_swiss_roll(sample_size, rescaled=True, return_as_tensor=True)[1]
        manifold_initial_point = dataset[np.random.randint(sample_size)].reshape(1, -1)
        manifold_initial_point = manifold_initial_point.to(device)
        samples_seq_reverse = sequential_posterior_sampler(prior_sampler, manifold_initial_point, M, likelihood_sigma, s, num_cycles=sample_size, eval_at_mean=eval_at_mean, status_bar=False, device=device)[1]
        zarr.save(os.path.join(base_dir, f'core/saved_arrays/eval_loc/sample_size={sample_size:.0g}_runs={num_runs:.0g}', f'{sampling_method}_{eval_method}s_{int(run_idx)}.zarr'), samples_seq_reverse)

def main():
    log_folder = os.path.join(base_dir, 'core/cluster/logs', '%j')
    # device = 'cpu'
    
    # ----------------------------- model parameters ----------------------------- #
    v = np.array([[4, 1]]).T
    M = v / np.linalg.norm(v)
    likelihood_sigma = 0.1
    s = 0.1
    # sample_size = int(5e2)
    sample_size = int(1e4)
    num_runs = 200
    
    # eval_methods = ['xt', 'mu']
    eval_methods = ['xt']
    # sampling_methods = ['iid', 'seq']
    sampling_methods = ['seq']

    # ------------------------- submitit cluster executor ------------------------ #
    ex = submitit.AutoExecutor(folder=log_folder)
    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f'!!! Slurm executable `srun` not found. Will execute jobs on "{ex.cluster}"')
        
    # slurm parameters
    # ex.update_parameters(
    #     slurm_job_name = 'eval_methods',
    #     nodes = 1,
    #     slurm_partition = 'gpu',
    #     slurm_gpus_per_task=1,
    #     slurm_ntasks_per_node=1,
    #     slurm_constraint='v100-32gb',
    #     cpus_per_task=12,
    #     mem_gb=32,
    #     timeout_min=3000,
    # )

    ex.update_parameters(
        slurm_job_name = 'eval_methods',
        nodes = 1,
        slurm_partition = 'ccn',
        slurm_cpus_per_task = 1,
        slurm_ntasks_per_node = 1,
        cpus_per_task = 4,
        mem_gb = 32,
        timeout_min = 300,
    )

    save_dir = f'sample_size={sample_size:.0g}_runs={num_runs:.0g}'
    save_dir = os.path.join(base_dir, 'core', 'saved_arrays', 'eval_loc', save_dir)
    from pathlib import Path
    Path(save_dir).mkdir(parents=False, exist_ok=True)

    # ----------------------------------- jobs ----------------------------------- #
    jobs = []
    with ex.batch():
        for eval_method in eval_methods:
            for sampling_method in sampling_methods:
                for run_idx in range(num_runs):
                    job = ex.submit(generate_samples, 
                                    sampling_method, eval_method, sample_size, M, likelihood_sigma, s, num_runs, run_idx
                                    )
                    jobs.append(job)
    for idx in range(len(jobs)):
        print(f'Job {jobs[idx].job_id}')
    print('all jobs submitted!')

if __name__ == '__main__':
    main()
