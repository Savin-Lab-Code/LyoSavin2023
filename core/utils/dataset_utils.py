import os, sys
project_root = os.path.abspath("")
if project_root[-12:] == 'LyoSavin2023':
    base_dir = project_root
else:
    base_dir = os.path.dirname(project_root)
sys.path.append(os.path.join(base_dir, 'core'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from utils import rescale_samples_to_pm1


def sample_gaussian(mean, std, sample_size):
    samples = torch.normal(mean=mean, std=std, out=torch.Tensor(sample_size,2))
    return samples

def bimodal_gaussian(mean1, mean2, std1, std2, num1, num2):
    '''
     returns a 2D mixture of Gaussians 
    '''
    samples1 = sample_gaussian(mean1, std1, num1)
    samples2 = sample_gaussian(mean2, std2, num2)
    samples = torch.cat((samples1, samples2), 0)
    return samples

def two_spirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    
    # coords = np.vstack((np.hstack((d1x,d1y)), np.hstack((-d1x,-d1y))))
    coords = np.hstack((d1x, d1y))
    # labels = np.hstack((np.zeros(n_points), np.ones(n_points)))
    # return (torch.tensor(coords), torch.tensor(labels))
    return torch.tensor(coords).float()

def make_line(n_points, angle=np.pi/6, noise=0.1):
    """
    returns points lying on a line, centred around the origin
    """
    n_points = int(n_points)
    r = np.linspace(-5, 5, n_points).reshape(-1, 1)
    x = r * np.cos(angle)
    y = r * np.sin(angle)

    x += np.random.rand(n_points,1) * noise
    y += np.random.rand(n_points,1) * noise
    
    coords = np.hstack((x, y))
    
    return np.array(coords)

def make_circle(n_points, radius=1.5, noise=0.1):
    """
     returns points lying on a circle, centered around the origin
    """
    angle = np.sqrt(np.random.rand(n_points, 1)) * 780 * 2 * np.pi / 360
    x = radius * np.cos(angle) + np.random.rand(n_points, 1) * noise
    y = radius * np.sin(angle) + np.random.rand(n_points, 1) * noise
    
    coords = np.hstack((x, y))
    
    return np.array(coords)

def make_cross(n_points, length=1.5, noise=0.1):
    """
    returns a cross
    """
    line1_x = np.random.uniform(low=-length, high=length, size=(n_points//2, 1))
    line1_y = (np.random.rand(n_points//2, 1) * 2 - 1) * noise * 0.5
    coords1 = np.hstack((line1_x, line1_y))
    
    line2_x = (np.random.rand(n_points//2, 1) * 2 - 1) * noise * 0.5
    line2_y = np.random.uniform(low=-length, high=length, size=(n_points//2, 1))
    coords2 = np.hstack((line2_x, line2_y))
    
    coords = np.vstack((coords1, coords2))
    
    return np.array(coords)

def make_cross_3d(n_points, length=1.5, noise=0.1):
    """
    returns a cross dataset inside a 3D ambient space
    """
    line1_x = np.random.uniform(low=-length, high=length, size=(n_points, 1))
    line1_y = (np.random.rand(n_points, 1) * 2 - 1) * noise * 0.5
    line1_z = np.zeros((n_points, 1))
    coords1 = np.hstack((line1_x, line1_y, line1_z))
    
    line2_x = (np.random.rand(n_points, 1) * 2 - 1) * noise * 0.5
    line2_y = np.random.uniform(low=-length, high=length, size=(n_points, 1))
    coords2 = np.hstack((line2_x, line2_y, line1_z))
    
    coords = np.vstack((coords1, coords2))
    return np.array(coords)

def make_circle_3d(n_points, radius=1.5, noise=0.1):
    """
     returns points lying on a circle, centered around the origin. in 3d ambient space
    """
    angle = np.sqrt(np.random.rand(n_points, 1)) * 780 * 2 * np.pi / 360
    x = radius * np.cos(angle) + np.random.rand(n_points, 1) * noise
    y = radius * np.sin(angle) + np.random.rand(n_points, 1) * noise
    z = np.zeros((n_points, 1))
    
    coords = np.hstack((x, y, z))
    
    return np.array(coords)



# ------------------ create datasets of arbitrary dimension ------------------ #

def rotate_manifold(z, num_dims, i, j, theta):
    """
    takes a vector z and performs a right-hand rotation from the i-axis towards the j-axis, by angle theta. 
    """
    assert i < num_dims and j < num_dims, "indices should be smaller than the ambient dimension"

    # generator matrix
    x_hat = np.zeros(num_dims)
    x_hat[i] = 1
    y_hat = np.zeros(num_dims)
    y_hat[j] = 1
    gen = np.outer(y_hat, x_hat) - np.outer(x_hat, y_hat)

    # form rotation matrix from generator
    rot = expm(theta * gen)
    
    # apply the rotation matrix to the N-dim datapoint
    # if the batch size is >1, apply the matrix transformation to every datapoint
    x = []
    for z_i in z:
        x_i = np.matmul(rot, z_i)
        x.append(x_i)
    x = np.asarray(x)
    
    return x


def select_data_distribution(type, sample_size=12000):
    if type=='bimodal':
        # bimodal distribution
        mean1, mean2 = torch.Tensor([0., 0.]), torch.Tensor([20., 20.])
        std1, std2 = 2., 4.
        num1, num2 = round(sample_size*0.75), round(sample_size*0.25)
        data_samples = bimodal_gaussian(mean1, mean2, std1, std2, num1, num2)
        sample_size = num1+num2
    elif type=='twospiral':
        # two spiral distribution
        data_samples = two_spirals(sample_size, noise=.5)
        
    rescaled_data = rescale_samples_to_pm1(data_samples)
    return data_samples, rescaled_data


def set_noise_levels(num_noise_levels, start=0.0001, end=0.02):
    return torch.linspace(start, end, num_noise_levels)


from utils import add_noise_to_dataset
def create_train_test_split(N, noise_min, noise_max, rescaled_data, sample_size, test_fraction=0.15):
    '''
    this splits the dataset into the train and test portions, with proportional 
    representation of each noise level
    '''
    noise_levels = set_noise_levels(num_noise_levels=N, start=noise_min, end=noise_max)

    noisy_data = torch.empty((N, rescaled_data.shape[0], rescaled_data.shape[1]))
    for n in range(N):
        noisy_data_samples = add_noise_to_dataset(rescaled_data, noise_levels[n])
        noisy_data[n] = noisy_data_samples
        # noisy_data[n] = rescale_samples_to_pm1(noisy_data_samples)

    target_data = rescaled_data.repeat(N, 1, 1)
    target_labels = torch.swapaxes(noise_levels.repeat(sample_size,1), 1, 0).reshape(sample_size*N, -1)

    # flatten the tensors
    noisy_data_r = noisy_data.reshape(-1, 2)
    target_data_r = target_data.reshape(-1, 2)

    from sklearn.model_selection import train_test_split

    Y_train, Y_test, X_train, X_test = train_test_split(
        noisy_data_r,  # data
        target_data_r,  # target
        test_size=test_fraction,
        random_state=42,
        stratify=target_labels
    )
    return Y_train, Y_test, X_train, X_test


# ------------------------------- trimodal data ------------------------------ #
from sklearn.datasets import make_swiss_roll, make_moons, make_s_curve
from plot import plot_data_distribution, plot_trimodal_data_distribution_separate_colors
from utils import remove_all_ticks_and_labels
def load_trimodal_data(sample_size_per_manifold, offsets=[[0,0], [4,0], [2,4]], train_test_split=False, add_class_label=True, plot=False, noise=0.1):
    '''
    Generate a dataset comprising three 2D manifolds. 
    With the default offset value, the manifolds are not overlaid. 
    Whether the dataset returns a held out test set can be specified.
    '''
    sample_size_per_manifold = int(sample_size_per_manifold)
    
    # first datset: swiss roll
    swiss_roll, _ = make_swiss_roll(sample_size_per_manifold, noise=noise)
    swiss_roll = swiss_roll[:, [0, 2]]/10.0

    # second dataset: moons
    moons, _ = make_moons(sample_size_per_manifold, noise=noise/10)

    # third manifold: s_curve
    s_curve, _ = make_s_curve(sample_size_per_manifold, noise=noise/10)
    s_curve = s_curve[:, [0, 2]]/1.5
    
    # split into train and test sets
    def split_into_train_test(dataset, dataset_size):
        train_set = dataset[:dataset_size]
        test_set = dataset[dataset_size:]
        return train_set, test_set

    if train_test_split:
        test_size_per_manifold = int(sample_size_per_manifold*0.15)
        sample_size_per_manifold = int(sample_size_per_manifold*0.85)
        swiss_roll, swiss_roll_test = split_into_train_test(swiss_roll, sample_size_per_manifold)
        moons, moons_test = split_into_train_test(moons, sample_size_per_manifold)
        s_curve, s_curve_test = split_into_train_test(s_curve, sample_size_per_manifold)

    # show the three manifolds
    if plot:
        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
        ax[0].scatter(*swiss_roll.T, s=5)
        ax[2].scatter(*s_curve.T, s=5)
        ax[1].scatter(*moons.T, s=5)
        for a in range(3):
            ax[a].set_aspect('equal')
            remove_all_ticks_and_labels(ax[a])

    # offset each manifold
    def offset_manifolds(manifolds, offsets):
        manifold1 = manifolds[0] + offsets[0]
        manifold2 = manifolds[1] + offsets[1]
        manifold3 = manifolds[2] + offsets[2]
        return manifold1, manifold2, manifold3
    
    swiss_roll, moons, s_curve = offset_manifolds([swiss_roll, moons, s_curve], offsets)

    # convert to torch tensors and float
    swiss_roll = torch.Tensor(swiss_roll).float()
    moons = torch.Tensor(moons).float()
    s_curve = torch.Tensor(s_curve).float()

    # combine the three manifolds
    combined_dataset = torch.cat((swiss_roll, moons, s_curve), dim=0)

    # plot the data
    dataset = rescale_samples_to_pm1(combined_dataset)
    if plot:
        # plot_data_distribution(dataset)
        
        # plot the three manifolds but with separate colors
        plot_trimodal_data_distribution_separate_colors(dataset, sample_size_per_manifold)

    # add the class label to the data
    def add_label_to_data(data, labels, dataset_size):
        label_vector0 = torch.ones((dataset_size, 1)) * labels[0]
        label_vector1 = torch.ones((dataset_size, 1)) * labels[1]
        label_vector2 = torch.ones((dataset_size, 1)) * labels[2]
        label_vector = torch.cat((label_vector0, label_vector1, label_vector2), dim=0)
        data = torch.cat((data, label_vector), dim=1)
        return data
    
    print(f'size of the training set is {sample_size_per_manifold*3}')
    if add_class_label:
        dataset = add_label_to_data(dataset, [0,1,2], sample_size_per_manifold)
    
    # if train_test_split, do the same thing for the test dataset
    if train_test_split:
        swiss_roll_test, moons_test, s_curve_test = offset_manifolds([swiss_roll_test, moons_test, s_curve_test], offsets)
        
        swiss_roll_test = torch.Tensor(swiss_roll_test).float()
        moons_test = torch.Tensor(moons_test).float()
        s_curve_test = torch.Tensor(s_curve_test).float()
        
        combined_dataset_test = torch.cat((swiss_roll_test, moons_test, s_curve_test), dim=0)
        
        test_dataset = rescale_samples_to_pm1(combined_dataset_test)
        if plot:
            plot_data_distribution(test_dataset)
        
        print(f'size of the test set is {test_size_per_manifold*3}')
        if add_class_label:
            test_dataset = add_label_to_data(test_dataset, [0,1,2], test_size_per_manifold)
    
    if train_test_split:
        return dataset, test_dataset
    else:
        return dataset

def load_unimodal_data_3d(sample_size_per_manifold, manifold_type, offset=[0,0, 0], train_test_split=False, plot=True, noise=0.1):
    '''
    Generate a dataset comprising one 2D manifold in a 3D ambient space
    Whether the dataset returns a held out test set can be specified.
    '''
    sample_size_per_manifold = int(sample_size_per_manifold)
    
    if manifold_type == 'circle3d':
        dataset = make_circle_3d(sample_size_per_manifold, radius=1.5, noise=noise)
    elif manifold_type == 'cross3d':
        dataset = make_cross_3d(sample_size_per_manifold, length=1.5, noise=noise)
    else:
        print('enter `swiss_roll`, `moons`, `s_curve`, `circle`, or `cross`.')
        return

    # split into train and test sets
    def split_into_train_test(dataset, dataset_size):
        train_set = dataset[:dataset_size]
        test_set = dataset[dataset_size:]
        return train_set, test_set
    
    if train_test_split:
        test_size_per_manifold = int(sample_size_per_manifold*0.15)
        sample_size_per_manifold = int(sample_size_per_manifold*0.85)
        dataset, dataset_test = split_into_train_test(dataset, sample_size_per_manifold)
        
    # show the manifold pre-resizing
    if plot:
        fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
        dataset_2d = dataset.T[:2, :]
        print('dataset_2d.shape', dataset_2d.shape)
        ax.scatter(*dataset_2d, s=5)
        ax.set_aspect('equal')
    
    # offset each manifold
    def offset_manifold(manifold, offset):
        return manifold + offset
    
    dataset = offset_manifold(dataset, offset)
    
    # convert to torch tensors and float
    dataset = torch.Tensor(dataset).float()
    
    # plot the data
    dataset = rescale_samples_to_pm1(dataset)
    if plot:
        plot_data_distribution(dataset)
    
    print(f'size of the training set is {sample_size_per_manifold}')

    # if train_test_split, do the same thing for the test dataset
    if train_test_split:
        dataset_test = offset_manifolds(dataset_test, offset)
        
        dataset_test = torch.Tensor(dataset_test).float()
        
        test_dataset = rescale_samples_to_pm1(dataset_test)
        if plot:
            plot_data_distribution(test_dataset)
        
        print(f'size of the test set is {test_size_per_manifold}')
        if add_class_label:
            test_dataset = add_label_to_data(test_dataset, 0, test_size_per_manifold)
    
    if train_test_split:
        return dataset, test_dataset
    else:
        return dataset



def make_nd_dataset(n_points, manifold_type, radius=1.5, noise=0.1, n_dims=10, theta=np.pi/4, shrink_y_axis=False, return_as_tensor=False):
    '''
    take a manifold lying in a 2D plane and embeds it in a N-dimensional ambient space 
    using a rotation matrix A.  
    '''
    # circle dataset
    if manifold_type == 'circle':
        z = make_circle(n_points)
    elif manifold_type == 'swiss_roll':
        swiss_roll, _ = make_swiss_roll(n_points, noise=noise)
        z = swiss_roll[:, [0, 2]]/10.0
    elif manifold_type == 'cross':
        z = make_cross(n_points)
    elif manifold_type == 'swiss_roll_3d':
        swiss_roll, colors = make_swiss_roll(n_points, noise=noise)
        z = swiss_roll / 10.0
        z[:, 1] -= 1.0  # mean along the y axis should be 0
        if shrink_y_axis:
            z[:, 1] = z[:, 1] * 0.5  # shrink along the y axis by a factor of 0.5
    else:
        print('enter `swiss_roll(_3d)`, `circle`, or `cross`.')
    
    # embed this manifold trivially in a n_dim space
    emb_dim = z.shape[1]
    z_zeros = np.zeros((n_points, n_dims-emb_dim))
    # print(z.shape)
    # print(z_zeros.shape)
    
    z = np.hstack((z, z_zeros))
    
    # for every axis pair, rotate the vector by angle theta
    for i in range(0, n_dims-1):
        for j in range(i+1, n_dims):
            z = rotate_manifold(z, n_dims, i, j, theta)
    z = np.array(z)
    
    if return_as_tensor:
        z = torch.tensor(z, dtype=torch.float)
    
    # include colors for swiss_roll_3d
    if manifold_type == 'swiss_roll_3d':
        return z, colors
    else:
        return z


def load_unimodal_data_nd(sample_size_per_manifold, manifold_type, dim_amb, rotation_angle=np.pi/4, train_test_split=False, plot=True, noise=0.1, shrink_y_axis=False):
    '''
    Generate a dataset comprising one 2D latent manifold in an arbitrary dimensional ambient space
    Whether the dataset returns a held out test set can be specified.
    
    dim_amb specifies the dimension of the ambient space
    rotation_matrix specifies the transformation applied to the latent manifold to get the observation, x = Az
    '''
    sample_size_per_manifold = int(sample_size_per_manifold)
    
    dataset = make_nd_dataset(sample_size_per_manifold, manifold_type, radius=1.5, noise=noise, n_dims=dim_amb, theta=rotation_angle, shrink_y_axis=shrink_y_axis)

    # colors are included for swiss_roll_3d dataset
    if manifold_type=='swiss_roll_3d':
        dataset, colors = dataset

    # split into train and test sets
    def split_into_train_test(dataset, dataset_size):
        train_set = dataset[:dataset_size]
        test_set = dataset[dataset_size:]
        return train_set, test_set
    
    if train_test_split:
        test_size_per_manifold = int(sample_size_per_manifold*0.15)
        sample_size_per_manifold = int(sample_size_per_manifold*0.85)
        dataset, dataset_test = split_into_train_test(dataset, sample_size_per_manifold)
        
    # # show the manifold pre-resizing
    # fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
    # dataset_2d = dataset.T[:2, :]
    # print('dataset_2d.shape', dataset_2d.shape)
    # ax.scatter(*dataset_2d, s=5)
    # ax.set_aspect('equal')
    
    # convert to torch tensors and float
    dataset = torch.Tensor(dataset).float()
    
    # plot the data
    dataset = rescale_samples_to_pm1(dataset)
    # plot_data_distribution(dataset)
    
    print(f'size of the training set is {sample_size_per_manifold}')

    # if train_test_split, do the same thing for the test dataset
    if train_test_split:
        dataset_test = offset_manifolds(dataset_test, offset)
        
        dataset_test = torch.Tensor(dataset_test).float()
        
        test_dataset = rescale_samples_to_pm1(dataset_test)
        if plot:
            plot_data_distribution(test_dataset)
        
        print(f'size of the test set is {test_size_per_manifold}')
        if add_class_label:
            test_dataset = add_label_to_data(test_dataset, 0, test_size_per_manifold)
    
    if train_test_split:
        return dataset, test_dataset
    else:
        return dataset


from utils import rescale_samples_to_pos_quad

def load_unimodal_data(
    sample_size_per_manifold, 
    manifold_type, 
    offset=[0,0], 
    train_test_split=False, 
    normalize=True, 
    positive_quadrant=False, 
    add_class_label=False, 
    plot=True,
    noise=0.1,
    ):
    '''
    Generate a dataset comprising one 2D manifold
    Whether the dataset returns a held out test set can be specified.
    '''
    sample_size_per_manifold = int(sample_size_per_manifold)
    
    if manifold_type == 'swiss_roll':
        swiss_roll, _ = make_swiss_roll(sample_size_per_manifold, noise=noise)
        dataset = swiss_roll[:, [0, 2]]/10.0
    elif manifold_type == 'moons':
        dataset, _ = make_moons(sample_size_per_manifold, noise=noise/10)
    elif manifold_type == 's_curve':
        s_curve, _ = make_s_curve(sample_size_per_manifold, noise=noise/10)
        dataset = s_curve[:, [0, 2]]/1.5
    elif manifold_type == 'line':
        dataset = make_line(sample_size_per_manifold, noise=noise)
    elif manifold_type == 'circle':
        dataset = make_circle(sample_size_per_manifold, noise=noise)
    elif manifold_type == 'cross':
        dataset = make_cross(sample_size_per_manifold, length=1.5, noise=noise)
    elif manifold_type == 'circle3d':
        dataset = make_circle_3d(sample_size_per_manifold, radius=1.5, noise=noise)
    elif manifold_type == 'cross3d':
        dataset = make_cross_3d(sample_size_per_manifold, length=1.5, noise=noise)
    else:
        print('enter `swiss_roll`, `moons`, `s_curve`, `circle`, or `cross`.')
        return
    
    # split into train and test sets
    def split_into_train_test(dataset, dataset_size):
        train_set = dataset[:dataset_size]
        test_set = dataset[dataset_size:]
        return train_set, test_set
    
    if train_test_split:
        test_size_per_manifold = int(sample_size_per_manifold*0.15)
        sample_size_per_manifold = int(sample_size_per_manifold*0.85)
        dataset, dataset_test = split_into_train_test(dataset, sample_size_per_manifold)
        
    # show the manifold pre-resizing
    if plot:
        fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
        ax.scatter(*dataset.T, s=1)
        ax.set_aspect('equal')
        ax.set_title('pre-resizing')
    
    # offset each manifold
    def offset_manifold(manifold, offset):
        return manifold + offset
    
    dataset = offset_manifold(dataset, offset)
    
    # convert to torch tensors and float
    dataset = torch.Tensor(dataset).float()
    
    # plot the data
    if normalize:
        dataset = rescale_samples_to_pm1(dataset)
        if positive_quadrant:
            dataset = rescale_samples_to_pos_quad(dataset)
    if plot:
        plot_data_distribution(dataset)
        
    
    # add the class label to the data
    def add_label_to_data(data, label, dataset_size):
        label_vector = torch.ones((dataset_size, 1)) * label
        data = torch.cat((data, label_vector), dim=1)
        return data
    
    print(f'size of the training set is {sample_size_per_manifold}')
    if add_class_label:
        dataset = add_label_to_data(dataset, 0, sample_size_per_manifold)

    # if train_test_split, do the same thing for the test dataset
    if train_test_split:
        dataset_test = offset_manifolds(dataset_test, offset)
        
        dataset_test = torch.Tensor(dataset_test).float()
        
        test_dataset = rescale_samples_to_pm1(dataset_test)
        if plot:
            plot_data_distribution(test_dataset)
        
        print(f'size of the test set is {test_size_per_manifold}')
        if add_class_label:
            test_dataset = add_label_to_data(test_dataset, 0, test_size_per_manifold)
    
    if train_test_split:
        return dataset, test_dataset
    else:
        return dataset    


def construct_noisy_dataset(dataset, num_noise_levels, alphas_bar_sqrt, one_minus_alphas_prod_sqrt, add_noise_info=True, device='cpu'):
    '''
    construct a noisy version of a dataset. 
    Assumes the first two numbers are the values of the 2 pixel image, and the third is the class label. 
    '''
    noisy_dataset = []
    for s in range(num_noise_levels):
        s = torch.tensor([s])
        
        noisy_dataset_t = q_x(dataset[:, :2], s, alphas_bar_sqrt, one_minus_alphas_prod_sqrt)
        
        class_label = dataset[:, 2].view(len(dataset), 1)
        noisy_dataset_t = torch.cat((noisy_dataset_t, class_label), dim=1)
        
        if add_noise_info:
            noise_level = s.repeat([len(dataset), 1])
            noisy_dataset_t = torch.cat((noisy_dataset_t, noise_level), dim=1)
        
        noisy_dataset.append(noisy_dataset_t)
        
    noisy_dataset = torch.stack(noisy_dataset, 0).view(-1, noisy_dataset_t.shape[1])  # shape = (num_datapoints, 2 + 1 (+ 1))
    return noisy_dataset


# ----------------------- generate scaled noisy images ----------------------- #
def q_x(x_0, t, alphas_bar_sqrt, one_minus_alphas_prod_sqrt, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_prod_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)





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
            # print('returning as tensor')
            t = torch.tensor(t, dtype=torch.float)
            clean_manifold = torch.tensor(clean_manifold, dtype=torch.float)
        
        return t, clean_manifold, min, max
    
    clean_manifold = np.vstack([x, y]).T
    if return_as_tensor:
        # print('returning as tensor')
        t = torch.tensor(t, dtype=torch.float)
        clean_manifold = torch.tensor(clean_manifold, dtype=torch.float)
    # print(type(clean_manifold))
    return t, clean_manifold

def generate_2d_s_curve(num_samples, return_as_tensor=False):
    num_samples = int(num_samples)
    t = 3 * np.pi * (np.linspace(0, 1, num_samples) - 0.5)
    x = np.sin(t)
    # y = 2.0 * generator.rand(1, n_samples)
    y = np.sign(t) * (np.cos(t) - 1)
    clean_manifold = np.vstack([x, y]).T
    
    if return_as_tensor:
        t = torch.tensor(t, dtype=torch.float)
        clean_manifold = torch.tensor(clean_manifold, dtype=torch.float)
    return t, clean_manifold

def generate_2d_moons(n_samples, return_as_tensor=False):
    if isinstance(n_samples, int):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError as e:
            raise ValueError(
                "`n_samples` can be either an int or a two-element tuple."
            ) from e
    
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    clean_manifold = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    
    if return_as_tensor:
        clean_manifold = torch.tensor(clean_manifold, dtype=torch.float)

    return clean_manifold

def generate_2d_trimodal_distribution(dataset_size, offsets:list = [[0,0], [4,0], [2,4]]):
    from dataset_utils import generate_2d_swiss_roll, generate_2d_s_curve, generate_2d_moons
    
    # first dataset: swiss roll
    swiss_roll = generate_2d_swiss_roll(dataset_size, rescaled=False, return_as_tensor=False)[1]
    # second dataset: moons
    moons = generate_2d_moons(dataset_size)
    # third manifold: s_curve
    _, s_curve = generate_2d_s_curve(dataset_size)
    s_curve = s_curve/1.5
    
    def offset_manifolds(manifolds, offsets):
        manifold1 = manifolds[0] + offsets[0]
        manifold2 = manifolds[1] + offsets[1]
        manifold3 = manifolds[2] + offsets[2]
        return manifold1, manifold2, manifold3

    swiss_roll, moons, s_curve = offset_manifolds([swiss_roll, moons, s_curve], offsets)

    # convert to torch tensors and float
    swiss_roll = torch.Tensor(swiss_roll).float()
    moons = torch.Tensor(moons).float()
    s_curve = torch.Tensor(s_curve).float()

    # combine the three manifolds
    combined_dataset = torch.cat((swiss_roll, moons, s_curve), dim=0)

    from utils import rescale_samples_to_pm1
    dataset = rescale_samples_to_pm1(combined_dataset)
    
    return dataset


def generate_2d_swiss_roll_uniformly_distributed(num_samples, rescaled=True):
    '''
    generates the `x, y` coordinates of a 2d swiss roll, as well as the `t` coords. 
    Unlike the above function, this one ensures that the xy data points are uniformly distributed on the latent line
    '''
    num_samples = int(num_samples)
    
