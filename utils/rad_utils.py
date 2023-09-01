import torch
import matplotlib.pyplot as plt
import numpy as np
# from tqdm import trange
from tqdm.auto import trange
device = 'cpu'

# -------------------- select and load weights into model -------------------- #

def select_model(model_name, model_version_number):
    from models import NoiseConditionalEstimatorConcat, UnbiasedNoiseConditionalEstimatorConcat3Layers, UnbiasedNoiseConditionalEstimatorConcat4Layers
    from models import VariableDendriticCircuit, VariableDendriticCircuitSomaBias
    from utils import load_model_description, load_model_weights, count_parameters

    # diffusion model parameters
    diffuser_details = load_model_description(model_name, model_version_number)

    model_name = diffuser_details['model_name']
    model_num = diffuser_details['model_number']
    num_steps = diffuser_details['num_steps']
    num_hidden = diffuser_details['num_hidden']

    # dim_amb = diffuser_details['num_manifold_dims']
    if 'num_ambient_dims' in diffuser_details.keys():
        dim_amb = diffuser_details['num_ambient_dims']
    else: 
        if model_num == 47:
            dim_amb = 2
    
    dim_amb = int(dim_amb)

    print('embedding dimension is', dim_amb)
    # print('manifold dimension is', diffuser_details['num_manifold_dims'])

    # initialize model
    if model_name == 'unconditional-concat':
        model = NoiseConditionalEstimatorConcat(num_hidden)
    elif model_name == 'unconditional-dendritic' or model_name == 'unconditional-dendritic-3d-manifold':
        model = VariableDendriticCircuit(hidden_cfg=num_hidden, num_in=dim_amb, num_out=dim_amb, bias=True)

    # load model state dict
    model = load_model_weights(model, model_name, model_num, device)
    print(f'the model has {count_parameters(model)} parameters')
    
    model.eval()
    
    return model, num_steps, dim_amb



# --------------------- sampling from the forward process -------------------- #
def diffused_sample(num_steps, t, x_0):
    from utils import forward_process
    from utils import extract
    _, _, alphas_prod, _, alphas_bar_sqrt, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device)
    batch_size = x_0.shape[0]
    
    t = torch.tensor([t], dtype=torch.long)

    scale_factor_t = extract(alphas_bar_sqrt, t, x_0)
    
    # Generate z
    z = torch.randn_like(x_0, device=device)

    # Fixed sigma
    sigma_t = extract(one_minus_alphas_prod_sqrt, t, x_0)
    
    x_t = scale_factor_t * x_0 + sigma_t * z
    return x_t



# ----------------- sampling from the vanilla reverse process ---------------- #
def plot_reverse_samples_10_steps(model, sample_size, embedding_dims, num_steps):
    from generate_data import p_sample_loop
    from utils import forward_process

    # num_dim_data = 2
    num_dim_data = embedding_dims

    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device)

    sample_size = int(6e2)
    lims = [-2,2]
    x_seq = p_sample_loop(model, (sample_size, num_dim_data), num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device)

    fig, axes = plt.subplots(1, 10, figsize=(12, 2), sharey=True)
    for i in range(1, 11):
        cur_x = x_seq[i * 10].detach().cpu()
        
        ax = axes[i-1]
        # ax.scatter(cur_x[:, 0], cur_x[:, 1],color='white',edgecolor='indianred', s=1);
        ax.scatter(cur_x[:, 0], cur_x[:, 1], s=1);
        ax.set_aspect('equal')
        ax.set(xlim=lims, ylim=lims)
    return alphas, betas, one_minus_alphas_prod_sqrt, x_seq


def prior_sampling(num_steps, model, num_dim_data, dim1=0, dim2=1, sample_size=int(1e3), plot=True):
    '''
    performs the vanilla reverse process and returns the sequence of x_t's as the t varies.  
    '''
    from generate_data import p_sample_loop
    from utils import forward_process
    
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device)
    x_seq = p_sample_loop(model, (sample_size, num_dim_data), num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device)

    if plot:
        lims = [-1.5, 1.5]
        fig, axes = plt.subplots(1, 5, figsize=(10, 3), sharey=True)
        
        for i in range(1, 6):
            cur_x = x_seq[i * 20].detach().cpu()
            ax = axes[i-1]
            
            # x-y plane
            ax.scatter(cur_x[:, dim1], cur_x[:, dim2],color='white',edgecolor='indianred', s=0.5);
            
            # y-z plane
            ax.set_aspect('equal')
            ax.set(xlim=lims, ylim=lims)
            ax.set_yticks([-1, 0, 1])
            
            fig.suptitle(f'x=dim {dim1}, y=dim {dim2}')
        fig.tight_layout()
    else: 
        x_seq = torch.stack(x_seq)
        return x_seq


def rad_ad_various_temps(data_2d, model, alphas, betas, one_minus_alphas_prod_sqrt, temps = [10, 30, 50, 100], lims=[-2, 2]):
    fig, ax = plt.subplots(1, 4, figsize=(11,5))

    x_init = data_2d
    idx = 0
    for t in temps:
        for j in range(6):
            x_rad, _ = p_rev_loop(model, x_init, x_init.shape, t, alphas, betas, one_minus_alphas_prod_sqrt, device)
            x_rad = torch.stack(x_rad).detach().cpu()
            ax[idx].scatter(x_rad[-1,:,0], x_rad[-1,:,1], color='orange', s=5)
        
        # plot the underlying distribution
        ax[idx].scatter(data_2d[:,0], data_2d[:,1], color='white', edgecolor='grey', s=5, label="initial points")
        
        # plot details
        ax[idx].set_aspect('equal')
        ax[idx].set(xlim=lims, ylim=lims)
        ax[idx].legend()
        ax[idx].set_title(f't={t}')
        
        idx += 1

    # add figure details
    fig.suptitle(f'Distribution of datapoints during RAD process\n at different temperatures')
    fig.tight_layout()


# line dataset generation function:
def generate_line(num_samples, rescaled=True):
    '''
    generates the `x, y` coordinates of a 2d line, as well as the `t` coords
    '''
    n_points = int(n_points)
    r = np.linspace(-5, 5, n_points).reshape(-1, 1)
    x = r * np.cos(angle)
    y = r * np.sin(angle)

    x += np.random.rand(n_points,1) * noise
    y += np.random.rand(n_points,1) * noise
    
    coords = np.hstack((x, y))
    
    if rescaled:
        '''
        rescale the 2D data points to lie within (-1, 1). Preserves aspect ratio
        '''
        from utils import rescale_samples_to_pm1
        dataset = rescale_samples_to_pm1(coords)
        
    
    return r, np.array(coords)


# swiss roll generation function:
def generate_2d_swiss_roll(num_samples, rescaled=True, return_as_tensor=False):
    '''
    generates the `x, y` coordinates of a 2d swiss roll, as well as the `t` coords
    '''
    num_samples = int(num_samples)
    t = 1.5 * np.pi * (1 + 2 * np.linspace(0, 1, num_samples))
    x = t * np.cos(t) / 10
    y = t * np.sin(t) / 10
    
    # need to normalize
    if rescaled:
        '''
        rescale the 2D data points to lie within (-1, 1). Preserves aspect ratio
        '''
        data = np.vstack([x, y]).T
        min = np.min(data)
        max = np.max(data)
        
        rescaled_data = ((data - min) / (max - min))*2 -1

        x = rescaled_data[:, 0]
        y = rescaled_data[:, 1]
        clean_manifold = np.vstack([x, y]).T
        
        if return_as_tensor:
            print('returning as tensor')
            t = torch.tensor(t, dtype=torch.float)
            clean_manifold = torch.tensor(clean_manifold, dtype=torch.float)
        
        return t, clean_manifold, min, max
    
    clean_manifold = np.vstack([x, y]).T
    if return_as_tensor:
        print('returning as tensor')
        t = torch.tensor(t, dtype=torch.float)
        clean_manifold = torch.tensor(clean_manifold, dtype=torch.float)
    print(type(clean_manifold))

    return t, clean_manifold

def generate_2d_swiss_roll_uniformly_distributed(num_samples, rescaled=True):
    '''
    generates the `x, y` coordinates of a 2d swiss roll, as well as the `t` coords. 
    Unlike the above function, this one ensures that the xy data points are uniformly distributed on the latent line
    '''
    num_samples = int(num_samples)
    


# first define the reverse anti-diffusion function
from utils import extract
@torch.no_grad()
def p_sample_rev(model, x, t, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device):
    """
    returns one step of the reversed anti-diffusion process (i.e. the results at one temperature)
    """
    t = torch.tensor([t], device=device)
    
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_prod_sqrt, t, x))
    
    # Model output
    T = t.repeat(x.shape[0], 1) / n_steps
    eps_theta = model(x, T)
    
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    
    # Generate z
    z = torch.randn_like(x, device=device)
    
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return sample, mean

@torch.no_grad()
def p_rev_loop(model, x_0, shape, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device):
    """
    for each initial datapoint, this function returns the datapoints corresponding to every step of the reversed anti-diffusion process
    """
    cur_x = x_0
    x_seq = [cur_x]
    mean_seq = [cur_x]

    for i in range(n_steps):
        cur_x, cur_mean = p_sample_rev(model, cur_x, i, n_steps, alphas, betas, one_minus_alphas_prod_sqrt, device)
        x_seq.append(cur_x)
        mean_seq.append(cur_mean)
    return x_seq, mean_seq



def rad_ad_cycle(xi_rad, model, num_steps, alphas, betas, one_minus_alphas_prod_sqrt):
    '''
    diffuses the datapoints using the RAD process, and then anti-diffuses using the AD process.
    '''
    from generate_data import p_sample_loop
    import numpy
    
    # assert xi_rad!=None
    # if xi_rad==None:
        # xi_rad = data_2d
        
    if type(xi_rad) == numpy.ndarray:
        xi_rad = torch.tensor(xi_rad, dtype=torch.float)
    else: 
        xi_rad = xi_rad
    
    # RAD
    x_rad_seq, _ = p_rev_loop(model, xi_rad, xi_rad.shape, num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device)
    x_rad_seq = torch.stack(x_rad_seq).detach().cpu()
    xf_rad = x_rad_seq[-1, :, :]
    
    # AD
    xi_ad = xf_rad
    x_ad_seq = p_sample_loop(model, xi_ad.shape, num_steps, alphas, betas, one_minus_alphas_prod_sqrt, device, init_x = xi_ad)
    x_ad_seq = torch.stack(x_ad_seq).detach().cpu()
    xf_ad = x_ad_seq[-1, :, :]

    return xf_ad, x_rad_seq, x_ad_seq


def sequential_prior_sampler(model, init_x, num_cycles, num_steps=100, disable_tqdm=False):
    import numpy
    from utils import forward_process
    betas, alphas, _, _, _, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device)
    
    if type(init_x) == numpy.ndarray:
        init_x = torch.tensor(init_x, dtype=torch.float)
    else: 
        init_x = init_x

    
    # go through the rad-ad cycle multiple times
    x_ad, x_fwd, x_rev = rad_ad_cycle(init_x, model, num_steps, alphas, betas, one_minus_alphas_prod_sqrt)  # burn the first sample
    
    # print(x_ad)
    
    seq_x = []
    seq_fwd_x = []
    seq_rev_x = []
    for i in trange(num_cycles, disable=disable_tqdm):
        x_ad, x_fwd, x_rev = rad_ad_cycle(x_ad, model, num_steps, alphas, betas, one_minus_alphas_prod_sqrt)
        seq_x.append(x_ad)
        seq_fwd_x.append(x_fwd)
        seq_rev_x.append(x_rev)
    seq_x = torch.stack(seq_x).detach().numpy().reshape(num_cycles, -1)
    seq_fwd_x = torch.stack(seq_fwd_x).detach().numpy().reshape(num_cycles, num_steps+1, -1)
    seq_rev_x = torch.stack(seq_rev_x).detach().numpy().reshape(num_cycles, num_steps+1, -1)

    return seq_x, seq_fwd_x, seq_rev_x


# average of many independent sequential runs
def generate_sequential_samples_line(ground_truth_manifold, model_line, num_steps, num_cycles, alphas, betas, one_minus_alphas_prod_sqrt):
    manifold_initial_point = ground_truth_manifold[np.random.randint(ground_truth_manifold.shape[0])].reshape(1, -1)
    xf_ad = rad_ad_cycle(manifold_initial_point, model_line, num_steps, alphas, betas, one_minus_alphas_prod_sqrt)
    xfs_line = []
    # for i in range(num_iters-1):
    # for i in trange(num_cycles-1, desc='sequential sampling', unit='cycles'):
    for i in range(num_cycles):
        xf_ad = rad_ad_cycle(xf_ad, model_line, num_steps, alphas, betas, one_minus_alphas_prod_sqrt)
        xfs_line.append(xf_ad)
    return torch.stack(xfs_line)



def project_onto_clean_2d_roll_manifold(datapoints, clean_manifold, min, max, clean_manifold_t, normalized=True):
    '''
    "stick" the noisy data to the closest point on the clean manifold
    '''
    if type(datapoints) == torch.Tensor:
        datapoints = datapoints.detach().numpy()
    # calculate distances to every point on clean manifold
    if datapoints.ndim == 1:
        [x, y] = datapoints  # this is for one set of coords, but not for a whole list of them
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
    elif datapoints.ndim == 2:
        x = datapoints[:, 0]
        y = datapoints[:, 1]
    clean_manifold_t_prev = clean_manifold_t
    t = np.repeat(clean_manifold_t.reshape(-1,1), len(x), axis=1)
    
    if normalized:
        dist = np.sqrt(((2*(1/10 *t*np.cos(t) - min))/(max - min) - 1 - x.T)**2 + ((2*(1/10 *t*np.sin(t) - min))/(max - min) - 1 - y.T)**2)
    else:
        dist_unscaled = np.sqrt(((t * np.cos(t))/10 - x)**2 + ((t * np.sin(t))/10 - y)**2)
    # has shape (num_points_in_clean_manifold, num_points_in_x)

    # for each x, identify index with smallest distance (there should only be one for small noise)
    manifold_pts_xy = []
    manifold_pts_t = []
    for di in dist.T:
        min_index = np.argmin(di)
        
        # identify the closest point on manifold 
        manifold_pt = clean_manifold[min_index, :]
        manifold_pts_xy.append(manifold_pt)
        
        # identify the t value associated with each x,y coord
        t_val = manifold_pts_t.append(clean_manifold_t_prev[min_index])
        
    manifold_pts_xy = np.vstack(manifold_pts_xy)
    manifold_pts_t = np.vstack(manifold_pts_t)
    return manifold_pts_xy, manifold_pts_t


def project_onto_clean_line_manifold(datapoints, clean_manifold, t, angle): 
    if type(datapoints) == torch.Tensor:
        datapoints = datapoints.detach().numpy()
    
    # calculate distances to every point on clean manifold
    if datapoints.ndim == 1:
        [x, y] = datapoints  # this is for one set of coords, but not for a whole list of them
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
    elif datapoints.ndim == 2:
        x = datapoints[:, 0].reshape(-1, 1)
        y = datapoints[:, 1].reshape(-1, 1)
    else:
        print('datapoints have wrong shape')
        return
    
    clean_manifold = np.repeat(clean_manifold.reshape(1, -1,2), len(x), axis=0)
    
    dist = np.sqrt((clean_manifold[:, :, 0] - x)**2 + (clean_manifold[:,:, 1] - y)**2)
    
    # has shape (num_points_in_clean_manifold, num_points_in_x)

    # for each x, identify index with smallest distance (there should only be one for small noise)
    manifold_pts_xy = []
    manifold_pts_t = []
    for di in dist:
        min_index = np.argmin(di)
        
        # identify the closest point on manifold 
        manifold_pt = clean_manifold[:, min_index, :]
        manifold_pts_xy.append(manifold_pt)
        
        # identify the t value associated with each x,y coord
        t_val = manifold_pts_t.append(t[min_index])
        
    manifold_pts_xy = np.vstack(manifold_pts_xy)
    manifold_pts_t = np.vstack(manifold_pts_t).reshape(-1,)
    return manifold_pts_xy, manifold_pts_t


def project_multiple_datasets_onto_ground_truth_line_manifold(xfs_line, clean_manifold, num_gt_points=5000):
    # we need to map the noisy points to the closest point on the clean manifold
    from rad_utils import project_onto_clean_line_manifold

    # project points onto the ground truth manifold
    t = np.linspace(-1, 1, num_gt_points)
    manifold_pts_ts_seq = []
    for j in range(xfs_line.shape[0]):
        _, manifold_pts_t = project_onto_clean_line_manifold(xfs_line[j], clean_manifold, t)
        manifold_pts_ts_seq.append(manifold_pts_t)

    manifold_pts_ts_seq = np.stack(manifold_pts_ts_seq)
    manifold_pts_ts_seq = manifold_pts_ts_seq.reshape(-1,)

    return manifold_pts_ts_seq

def project_multiple_datasets_onto_ground_truth_roll_manifold(seq_sampled_points, clean_manifold_params):
    '''
    takes in a sequence of datasets, and projects each dataset onto the ground truth manifold
    returns 1d array of t values (1d variable parameterizing the manifold)
    '''
    from rad_utils import project_onto_clean_2d_roll_manifold, generate_2d_swiss_roll
    
    clean_manifold_t, data_roll_gt, min, max = clean_manifold_params
    assert seq_sampled_points.shape[0] < clean_manifold_t.shape[0], 'number of samples in ground truth manifold must be greater than number of samples in each dataset'

    # project points onto the ground truth manifold
    seq_sampled_projected_pts = []
    for j in range(seq_sampled_points.shape[0]):
        _, manifold_pts_t = project_onto_clean_2d_roll_manifold(seq_sampled_points[j], data_roll_gt, min, max, clean_manifold_t)
        seq_sampled_projected_pts.append(manifold_pts_t)

    seq_sampled_projected_pts = np.stack(seq_sampled_projected_pts)
    return seq_sampled_projected_pts.reshape(-1,)
    

def calculate_autocorrelation(data_1d):
    # Calculate autocorrelation
    mean = np.mean(data_1d)
    var = np.var(data_1d)
    ndata = data_1d - mean

    # correlate
    acorr_seq = np.correlate(ndata, ndata, 'full')[len(ndata)-1:]
    return acorr_seq / var / len(ndata)


def remove_all_ticks_and_labels(ax, include_z_axis=False):
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        labelleft=False)
    if include_z_axis:
        ax.tick_params(
            axis='z',          # changes apply to the z-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False)
    
    return ax