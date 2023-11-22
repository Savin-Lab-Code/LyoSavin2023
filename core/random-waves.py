import submitit
import os, sys
import json
import numpy as np
import torch
from tqdm import tqdm

project_root = os.path.abspath("")  # alternative
if project_root[-12:] == 'LyoSavin2023':
    base_dir = project_root
else:
    base_dir = os.path.dirname(project_root)


def generate_samples(mode, variance, iter_num, num_samples=2e3):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)
    
    sys.path.append(os.path.join(base_dir, 'core'))
    sys.path.append(os.path.join(base_dir, 'core/utils'))
    
    from utils import select_model, make_beta_schedule
    from dataset_utils import generate_2d_swiss_roll
    from random_wave_utils import repeat_scalar, neural_sampling_custom_betas
    
    # -------------------------------- description ------------------------------- #
    model = select_model('unconditional-dendritic-4-layers', 1)[0]
    model.to(device)
    
    # dataset
    dataset = generate_2d_swiss_roll(int(2e3), True, True)[1]
    init_x = dataset[np.random.randint(0, len(dataset))].reshape(-1, 2).to(device)

    # sample a random datapoint from the dataset
    n_timesteps = 100
    original_betas = make_beta_schedule('sine', n_timesteps=n_timesteps)

    # num_samples = 2e3
    # variance = [0, .05, .1, .15, .2, .25, .3]
    deterministic_value = repeat_scalar(1, num_samples)
    periods = torch.normal(1, variance, size=(int(num_samples),))
    amplitudes = torch.normal(1, variance, size=(int(num_samples),))
    
    while periods.min() <= 0:
        # generate another sample to replace the negative ones
        periods[periods <= 0] = torch.normal(1, variance, size=(int(num_samples),))[periods <= 0]
        
    while amplitudes.min() <= 0:
        # generate another sample to replace the negative ones
        amplitudes[amplitudes <= 0] = torch.normal(1, variance, size=(int(num_samples),))[amplitudes <= 0]
        

    # generate the samples
    print('generating samples...', flush=True)
    if mode == 'period':
        x_seq_random_period = neural_sampling_custom_betas(model, init_x, num_samples, original_betas, periods, deterministic_value, disable_tqdm=True, device=device)
    elif mode == 'amplitude':
        x_seq_random_amplitude = neural_sampling_custom_betas(model, init_x, num_samples, original_betas, deterministic_value, amplitudes, disable_tqdm=True, device=device)
    elif mode == 'both':
        x_seq_random_both = neural_sampling_custom_betas(model, init_x, num_samples, original_betas, periods, amplitudes, disable_tqdm=True, device=device)
    else:
        raise ValueError('mode must be one of: period, amplitude, both')

    # save the data
    save_dir = os.path.join(base_dir, 'core', 'saved_arrays', 'random_waves')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print('saving data...', flush=True)
    if mode == 'period':
        np.save(os.path.join(save_dir, f'x_seq_random_period-variance={variance}-v{iter_num}.npy'), x_seq_random_period)
    elif mode == 'amplitude':
        np.save(os.path.join(save_dir, f'x_seq_random_amplitude-variance={variance}-v{iter_num}.npy'), x_seq_random_amplitude)
    elif mode == 'both':
        np.save(os.path.join(save_dir, f'x_seq_random_both-variance={variance}-v{iter_num}.npy'), x_seq_random_both)

    
    print('done!', flush=True)
    

def main():
    # ------------------------- submitit cluster executor ------------------------ #
    log_folder = os.path.join(base_dir, 'core/cluster/logs/random-waves', '%j')
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
        timeout_min = 60*1,  # 10 hours
    )
    
    
    # ------------------------------- parameters -------------------------------- #
    modes = ['period', 'amplitude', 'both']
    # modes = ['both']
    variances = [0, .05, .1, .15, .2, .25]
    num_repeats = np.arange(20)
    num_samples = 5e3
    

    jobs = []
    with ex.batch():
        for mode in modes:
            for variance in variances:
                for iter_num in num_repeats:
                    job = ex.submit(generate_samples, mode, variance, iter_num, num_samples)
                    jobs.append(job)
    print('all jobs submitted!')

    idx = 0
    print(f'Job {jobs[idx].job_id}')

if __name__ == '__main__':
    main()
