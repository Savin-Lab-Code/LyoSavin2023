import os, sys
project_root = os.path.abspath("")
if project_root[-12:] == 'LyoSavin2023':
    base_dir = project_root
else:
    base_dir = os.path.dirname(project_root)
sys.path.append(os.path.join(base_dir, 'core'))

import numpy as np
import torch
from tqdm.auto import trange

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


def project_onto_manifolds_in_trimodal_dataset(model_output, dataset_size:int = int(1e3), offsets:list = [[0,0], [4,0], [2,4]], noise:float = 0):
    '''
    determine the closest point on the swiss roll manifold (which is part of a trimodal dataset) and return the t value and (x,y) values
    '''
    from dataset_utils import generate_2d_trimodal_distribution
    dataset = generate_2d_trimodal_distribution(dataset_size, offsets)
    swiss_roll = dataset[:int(dataset_size), :].numpy()
    s_curve = dataset[-int(dataset_size):, :].numpy()
    swiss_roll_and_s_curve = np.concatenate((swiss_roll, s_curve), axis=0)
    # print(swiss_roll.shape)
    # print(s_curve.shape)
    # print(swiss_roll_and_s_curve.shape)

    calculate_distance = lambda x: np.linalg.norm(x - swiss_roll_and_s_curve, ord=2, axis=1)
    assert model_output.shape[1] == 2, "2nd dimension of `data` should be 2"
    
    min_idxs = np.zeros(model_output.shape[0], dtype=int)

    for idx in range(model_output.shape[0]):
        distances = calculate_distance(model_output[idx])
        min_idx = np.argmin(distances)
        min_idxs[idx] = min_idx
    
    return swiss_roll_and_s_curve, min_idxs, swiss_roll_and_s_curve[min_idxs]



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
    from dataset_utils import generate_2d_swiss_roll
    
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


def calculate_KL_divergence(histogram_samples, histogram_dataset, epsilon):
    from scipy.stats import entropy
    p = histogram_samples + epsilon  # distribution of model generated samples 
    q = histogram_dataset + epsilon  # distribution to compare against

    p = p.flatten() / np.sum(p)  # turn into a vector and then normalize
    q = q.flatten() / np.sum(q)  # turn into a vector and then normalize
    return entropy(p, q)