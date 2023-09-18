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
    
def generate_samples(sampling_method, eval_method, sample_size, num_runs, M, likelihood_sigma, s):
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
        samples_iid_reverse_array = []
        for i in range(num_runs):
            samples_iid_reverse = posterior_sample_loop_occlusion(prior_sampler, M, likelihood_sigma, s, (sample_size, ambient_dims), num_steps, eval_at_mean=eval_at_mean, device=device)
            samples_iid_reverse_array.append(samples_iid_reverse)
        samples_iid_reverse_array = np.stack(samples_iid_reverse_array)
        
        zarr.save(os.path.join(base_dir, 'core/saved_arrays/eval_loc/sample_size=1e3', f'{sampling_method}_{eval_method}s.zarr'), samples_iid_reverse_array)
        
    elif sampling_method == 'seq':
        from likelihood_utils import sequential_posterior_sampler
        samples_seq_reverse_array = []
        start_time = time.time()
        for i in range(num_runs):
            end_time = time.time()
            duration = end_time - start_time
            duration_mins = duration / 60
            duration_hrs = duration_mins / 60
            print(f'so far, this process took {duration_mins:.2f} minutes, which is {duration_hrs:.2f} hours.')
            print(f'run number: {int(i+1)}/{int(num_runs)}', flush=True)

            from dataset_utils import generate_2d_swiss_roll
            dataset = generate_2d_swiss_roll(sample_size, rescaled=True, return_as_tensor=True)[1]
            manifold_initial_point = dataset[np.random.randint(sample_size)].reshape(1, -1)
            manifold_initial_point = manifold_initial_point.to(device)
            
            samples_seq_reverse = sequential_posterior_sampler(prior_sampler, manifold_initial_point, M, likelihood_sigma, s, num_cycles=sample_size, eval_at_mean=eval_at_mean, status_bar=False, device=device)[1]
            samples_seq_reverse_array.append(samples_seq_reverse)
        samples_seq_reverse_array = np.stack(samples_seq_reverse_array)

        zarr.save(os.path.join(base_dir, 'core/saved_arrays/eval_loc/sample_size=1e3_runs=200', f'{sampling_method}_{eval_method}s.zarr'), samples_seq_reverse_array)

def main():
    log_folder = os.path.join(base_dir, 'core/cluster/logs', '%j')
    # device = 'cpu'
    
    # ----------------------------- model parameters ----------------------------- #
    v = np.array([[4, 1]]).T
    M = v / np.linalg.norm(v)
    likelihood_sigma = 0.1
    s = 0.1
    # sample_size = int(5e2)
    sample_size = int(1e3)
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
    ex.update_parameters(
        slurm_job_name = 'eval_methods',
        nodes = 1,
        # slurm_partition = 'ccn',
        slurm_partition = 'gpu',
        slurm_gpus_per_task=1,
        slurm_ntasks_per_node=1,
        slurm_constraint='v100-32gb',
        cpus_per_task=12,
        mem_gb=32,
        timeout_min=3000,
    )

    # ----------------------------------- jobs ----------------------------------- #
    jobs = []
    with ex.batch():
        for eval_method in eval_methods:
            for sampling_method in sampling_methods:
                job = ex.submit(generate_samples, 
                                sampling_method, eval_method, sample_size, num_runs, M, likelihood_sigma, s
                                )
                jobs.append(job)
    for idx in range(len(jobs)):
        print(f'Job {jobs[idx].job_id}')
    print('all jobs submitted!')

if __name__ == '__main__':
    main()
