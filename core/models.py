import os, sys
project_root = os.path.abspath("")
if project_root[-12:] == 'LyoSavin2023':
    base_dir = project_root
else:
    base_dir = os.path.dirname(project_root)
sys.path.append(os.path.join(base_dir, 'core'))
sys.path.append(os.path.join(base_dir, 'core/utils'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------- #
#           simple noise estimator with concatenated noise level info          #
# ---------------------------------------------------------------------------- #

class NoiseConditionalLinearConcat(nn.Module):
    '''
    A custom module. Defines a linear matrix and an embedding matrix. 
    '''
    def __init__(self, num_in, num_out):
        super(NoiseConditionalLinearConcat, self).__init__()
        self.num_in = num_in  # 2 for a 2 pixel image
        self.num_out = num_out  # size of the hidden layer, e.g. 32
        self.linear = nn.Linear(num_in + 1, num_out)  # A normal linear layer, 3x32
        
    def forward(self, x, t):
        t = t.float().view(len(t), 1)
        
        out = torch.cat((x, t), dim=1)
        out = self.linear(out)
        return out    
    

class NoiseConditionalEstimatorConcat(nn.Module):
    def __init__(self, num_hidden):
        super(NoiseConditionalEstimatorConcat, self).__init__()
        # num_hidden = 32
        self.condlin1 = NoiseConditionalLinearConcat(2, num_hidden)
        self.condlin2 = NoiseConditionalLinearConcat(num_hidden, num_hidden)
        self.condlin3 = NoiseConditionalLinearConcat(num_hidden, num_hidden)
        # self.condlin4 = NoiseConditionalLinearConcat(num_hidden, num_hidden)
        self.linear = nn.Linear(num_hidden, 2)
        self.nonlin = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.nonlin = nn.Softplus()
        
    def forward(self, x, t):
        '''
        x is the image
        t is the timestep
        '''
        x = self.condlin1(x, t)
        x = self.nonlin(x)

        x = self.condlin2(x, t)
        x = self.nonlin(x)

        x = self.condlin3(x, t)
        x = self.nonlin(x)

        # x = self.condlin4(x, t)
        # x = self.nonlin(x)
        
        # pre_nonlin = self.linear(x)
        # x = self.sigmoid(pre_nonlin)

        x = self.linear(x)
        
        return x

class FullyConnectedNetwork(nn.Module):
    '''
    Same as the NoiseConditionalEstimatorConcat but with arbitrary number of input/output dims
    '''
    def __init__(self, n_dim_data, num_hidden, num_hidden_layers=3):
        super(FullyConnectedNetwork, self).__init__()
        self.condlin1 = NoiseConditionalLinearConcat(n_dim_data, num_hidden)
        self.condlin = NoiseConditionalLinearConcat(num_hidden, num_hidden)
        self.linear = nn.Linear(num_hidden, n_dim_data)
        self.nonlin = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hidden_layers = self._make_layers(num_hidden, num_hidden_layers)
        # self.nonlin = nn.Softplus()
        
    def _make_layers(self, num_hidden, num_hidden_layers):
        layers = []
        for i in range(num_hidden_layers):
            layers += [self.condlin, self.nonlin]
        return MySequential(*layers)
    
    def forward(self, x, t):
        '''
        x is the image
        t is the timestep
        '''
        x = self.condlin1(x, t)
        x = self.nonlin(x)
        
        x = self.hidden_layers(x, t)
        x = self.linear(x)
        
        return x



class UnbiasedNoiseConditionalLinearConcat(nn.Module):
    '''
    A custom module. Defines a linear matrix and an embedding matrix. 
    '''
    def __init__(self, num_in, num_out):
        super(UnbiasedNoiseConditionalLinearConcat, self).__init__()
        self.num_in = num_in  # 2 for a 2 pixel image
        self.num_out = num_out  # size of the hidden layer, e.g. 32
        self.linear = nn.Linear(num_in + 1, num_out, bias=False)  # A normal linear layer, 3x32
        
    def forward(self, x, t):
        t = t.float().view(len(t), 1)
        
        out = torch.cat((x, t), dim=1)
        out = self.linear(out)
        return out    
    

# ---------------------------------------------------------------------------- #
#           unbiased unconditional denoiser model                              #
# ---------------------------------------------------------------------------- #

class UnbiasedNoiseConditionalLinearConcat(nn.Module):
    '''
    A custom module. Defines a linear matrix and an embedding matrix. 
    '''
    def __init__(self, num_in, num_out, bias):
        super(UnbiasedNoiseConditionalLinearConcat, self).__init__()
        self.num_in = num_in  # 2 for a 2 pixel image
        self.num_out = num_out  # size of the hidden layer, e.g. 32
        self.linear = nn.Linear(num_in + 1, num_out, bias=bias)  # A normal linear layer, 3x32
        
    def forward(self, x, t):
        t = t.float().view(len(t), 1)
        
        out = torch.cat((x, t), dim=1)
        out = self.linear(out)
        return out    
    
class UnbiasedNoiseConditionalEstimatorConcat3Layers(nn.Module):
    def __init__(self, num_hidden, bias=True):
        super(UnbiasedNoiseConditionalEstimatorConcat3Layers, self).__init__()
        # num_hidden = 32
        self.condlin1 = UnbiasedNoiseConditionalLinearConcat(2, num_hidden, bias=bias)
        self.condlin2 = UnbiasedNoiseConditionalLinearConcat(num_hidden, num_hidden, bias=bias)
        self.condlin3 = UnbiasedNoiseConditionalLinearConcat(num_hidden, num_hidden, bias=bias)

        self.linear = nn.Linear(num_hidden, 2, bias=bias)
        self.nonlin = nn.ReLU()
        
    def forward(self, x, t):
        '''
        x is the image
        t is the timestep
        '''
        x = self.condlin1(x, t)
        x = self.nonlin(x)

        x = self.condlin2(x, t)
        x = self.nonlin(x)

        x = self.condlin3(x, t)
        x = self.nonlin(x)
        
        x = self.linear(x)
        
        return x

class UnbiasedNoiseConditionalEstimatorConcat4Layers(nn.Module):
    def __init__(self, num_hidden, num_in=2, num_out=2, bias=False):
        super(UnbiasedNoiseConditionalEstimatorConcat4Layers, self).__init__()
        # num_hidden = 32
        self.condlin1 = UnbiasedNoiseConditionalLinearConcat(num_in, num_hidden, bias)
        self.condlin2 = UnbiasedNoiseConditionalLinearConcat(num_hidden, num_hidden, bias)
        self.condlin3 = UnbiasedNoiseConditionalLinearConcat(num_hidden, num_hidden, bias)
        self.condlin4 = UnbiasedNoiseConditionalLinearConcat(num_hidden, num_hidden, bias)

        self.linear = nn.Linear(num_hidden, num_out, bias=bias)
        self.nonlin = nn.ReLU()
        
    def forward(self, x, t):
        '''
        x is the image
        t is the timestep
        '''
        x = self.condlin1(x, t)
        x = self.nonlin(x)

        x = self.condlin2(x, t)
        x = self.nonlin(x)

        x = self.condlin3(x, t)
        x = self.nonlin(x)
        
        x = self.condlin4(x, t)
        x = self.nonlin(x)

        x = self.linear(x)
        
        return x
    



# ---------------------------------------------------------------------------- #
#            a modifiable noise concatenated dendritic neural network          #
# ---------------------------------------------------------------------------- #
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                x, t = inputs
                inputs = module(*inputs)
            elif isinstance(module, nn.ReLU) or isinstance(module, nn.Sigmoid) or isinstance(module, nn.Softplus):
                inputs = module(inputs)
            else:
                inputs = module(inputs, t)
        return inputs
    
class NoiseConditionalLinearConcat1(nn.Module):
    '''
    A custom module. Defines a linear matrix and an embedding matrix. 
    '''
    def __init__(self, num_in, num_out, bias=True):
        super(NoiseConditionalLinearConcat1, self).__init__()
        # self.num_in = num_in if bias else num_in-1 # 2 for a 2 pixel image
        self.num_in = num_in # 2 for a 2 pixel image
        self.num_out = num_out  # size of the hidden layer, e.g. 32
        self.linear = nn.Linear(self.num_in + 1, self.num_out, bias=bias)  # A normal linear layer, 3x32
        self.linear.weight.data.uniform_(-1,1)
        
    def forward(self, x, t):
        t = t.float().view(len(t), 1)
        
        out = torch.cat((x, t), dim=1)
        out = self.linear(out)
        return out
    

class DendriticBranchLayer(nn.Module):
    def __init__(self, num_in, num_out, branch_factor):
        super(DendriticBranchLayer, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.branch_factor = branch_factor
        assert self.branch_factor > 1

        # self.layer_weights = torch.randn(self.num_out, self.branch_factor + 1)
        self.layer_weights = nn.Parameter(data=torch.ones(self.num_out, self.branch_factor + 1), requires_grad=True)
        self.layer_weights.data.uniform_(-1, 1)
        # self.layer_weights = nn.Parameter(data=torch.ones(self.num_out, self.branch_factor + 1, requires_grad=True))
        # # self.layer_weights.data.uniform_(-1, 1)

    def forward(self, x, t):
        '''
        x is the image
        t is the index corresponding to the noise level
        Each layer's units have one to many connections to nodes of the previous layer.
        No two units have connections to the same unit in the previous layer.
        '''
        t = t.float().view(len(t), 1)
        batch_size = x.size()[0]

        # Append t to the end of each row
        x = x.view(batch_size, -1, self.branch_factor)
        t = t.repeat_interleave(self.num_out, 1).view(batch_size, -1, 1)
        x_t = torch.cat((x, t), dim=2)
        
        # Each neuron has its own dendritic branch.
        # This is equivalent to a diagonal block weight matrix
        out = x_t * self.layer_weights
        out = torch.sum(out, dim=2)
        return out
    
class DendriticBranchLayerSomaBias(nn.Module):
    def __init__(self, num_in, num_out, branch_factor):
        super(DendriticBranchLayerSomaBias, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.branch_factor = branch_factor
        assert self.branch_factor > 1

        self.layer_weights = nn.Parameter(data=torch.ones(self.num_out, self.branch_factor + 1), requires_grad=True)
        # self.layer_bias = nn.Parameter(data=torch.ones(self.num_out, self.branch_factor + 1), requires_grad=True)
        self.layer_bias = nn.Parameter(data=torch.ones(self.num_out,), requires_grad=True)
        self.layer_weights.data.uniform_(-1, 1)
        self.layer_bias.data.uniform_(-1, 1)

    def forward(self, x, t):
        '''
        x is the image
        t is the index corresponding to the noise level
        Each layer's units have one to many connections to nodes of the previous layer.
        No two units have connections to the same unit in the previous layer.
        '''
        t = t.float().view(len(t), 1)
        batch_size = x.size()[0]

        # Append t to the end of each row
        x = x.view(batch_size, -1, self.branch_factor)
        t = t.repeat_interleave(self.num_out, 1).view(batch_size, -1, 1)
        x_t = torch.cat((x, t), dim=2)
        
        # Each neuron has its own dendritic branch.
        # This is equivalent to a diagonal block weight matrix 
        out = x_t * self.layer_weights
        # out = x_t * self.layer_weights + self.layer_bias
        out = torch.sum(out, dim=2) + self.layer_bias
        return out


class DendriticBranchLayerNoNoiseInfo(nn.Module):
    def __init__(self, num_in, num_out, branch_factor):
        super(DendriticBranchLayerNoNoiseInfo, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.branch_factor = branch_factor
        assert self.branch_factor > 1

        # self.layer_weights = torch.randn(self.num_out, self.branch_factor + 1)
        self.layer_weights = nn.Parameter(data=torch.ones(self.num_out, self.branch_factor + 1), requires_grad=True)
        self.layer_weights.data.uniform_(-1, 1)
        # self.layer_weights = nn.Parameter(data=torch.ones(self.num_out, self.branch_factor + 1, requires_grad=True))
        # # self.layer_weights.data.uniform_(-1, 1)

    def forward(self, x, t):
        '''
        x is the image
        t is the index corresponding to the noise level
        Each layer's units have one to many connections to nodes of the previous layer.
        No two units have connections to the same unit in the previous layer.
        '''
        t = t.float().view(len(t), 1)
        batch_size = x.size()[0]

        # Append t to the end of each row
        x = x.view(batch_size, -1, self.branch_factor)
        # t = t.repeat_interleave(self.num_out, 1).view(batch_size, -1, 1)
        # x_t = torch.cat((x, t), dim=2)
        
        # Each neuron has its own dendritic branch.
        # This is equivalent to a diagonal block weight matrix
        out = x * self.layer_weights
        out = torch.sum(out, dim=2)
        return out
    
class DendriticBranchLayerSeparateT(nn.Module):
    def __init__(self, num_in, num_out, branch_factor):
        super(DendriticBranchLayerSeparateT, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.branch_factor = branch_factor
        assert self.branch_factor > 1

        # self.layer_weights = torch.randn(self.num_out, self.branch_factor + 1)
        self.layer_weights = nn.Parameter(data=torch.ones(self.num_out, self.branch_factor), requires_grad=True)
        self.layer_weights.data.uniform_(-1, 1)
        # self.layer_weights = nn.Parameter(data=torch.ones(self.num_out, self.branch_factor + 1, requires_grad=True))
        # # self.layer_weights.data.uniform_(-1, 1)
        
        self.t_weights = nn.Parameter(torch.randn(1, self.num_out), requires_grad=True)

    def forward(self, x, t):
        '''
        x is the image
        t is the index corresponding to the noise level
        Each layer's units have one to many connections to nodes of the previous layer.
        No two units have connections to the same unit in the previous layer.
        '''
        t = t.float().view(len(t), 1)
        batch_size = x.size()[0]
        
        x = x.view(batch_size, -1, self.branch_factor)
        out = self.layer_weights * x
        out = torch.sum(out, dim=2)
        t_out = t @ self.t_weights
        out = torch.add(out, t_out)
        return out


class DendriticBranchLayerSparse(nn.Module):
    '''
    dendritic branch layer using sparse weight matrices
    '''
    def __init__(self, num_in, num_out, branch_factor):
        super(DendriticBranchLayerSparse, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.branch_factor = branch_factor
        assert self.branch_factor > 1

        self.weight_vals = nn.Parameter(torch.randn((self.branch_factor) * self.num_out,), requires_grad=True).cuda()
        # self.layer_weights = self._create_sparse_weight_matrix(self.num_out, self.branch_factor, self.weight_vals)

        self.t_weights = nn.Parameter(torch.randn(self.num_out, 1), requires_grad=True)
    
    def _create_sparse_weight_matrix(self, input_dims, branch_factor, values):
        '''
        create tree tensor with sparse COO encoding
        '''        
        row_indices = np.repeat(np.arange(input_dims), branch_factor)
        col_indices = np.arange(input_dims * branch_factor)
        return torch.sparse_coo_tensor(indices=[row_indices, col_indices], values=values)


    def forward(self, x, t):
        '''
        x is the image
        t is the index corresponding to the noise level
        Each layer's units have one to many connections to nodes of the previous layer.
        No two units have connections to the same unit in the previous layer.
        '''
        t = t.float().view(len(t), 1)
        batch_size = x.size()[0]
        
        # print('num in:', self.num_in)
        # print('num out:', self.num_out)
        # print('t:', t.shape)
        # print('x:', x.shape)
        # print('t weights:', self.t_weights.shape)
        
        layer_weights = self._create_sparse_weight_matrix(self.num_out, self.branch_factor, self.weight_vals)
        
        # print('layer weights:', layer_weights.shape)
        
        out = torch.sparse.mm(layer_weights, x.T)
        t_out = self.t_weights @ t.T

        # print('t_out:', t_out.shape)
        # print('out:', out.shape)

        out = torch.add(out, t_out).T
        # print('final output:', out.shape)
        # print('---------')
        return out
    

class VariableDendriticCircuit(nn.Module):
    def __init__(self, hidden_cfg, num_in, num_out, bias=True):
        super(VariableDendriticCircuit, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.nonlin = nn.ReLU()
        # self.nonlin = nn.Sigmoid()
        self.features = self._make_layers(hidden_cfg, bias)

    def _make_layers(self, cfg_layers, bias):
        layers = []
        cfg_layers.append(self.num_out)
        num_units_in_layer = np.cumprod(cfg_layers[::-1])[::-1]
        layers += [
            NoiseConditionalLinearConcat1(self.num_in, num_units_in_layer[0], bias),
        ]

        for i in range(len(cfg_layers)-1):
            layers += [
                self.nonlin,
                DendriticBranchLayer(num_units_in_layer[i], num_units_in_layer[i+1], cfg_layers[i]),
                # DendriticBranchLayerSparse(num_units_in_layer[i], num_units_in_layer[i+1], cfg_layers[i]),
                # DendriticBranchLayerSeparateT(num_units_in_layer[i], num_units_in_layer[i+1], cfg_layers[i]),
            ]
        return MySequential(*layers)

    def forward(self, x, t):
        out = self.features(x, t)
        return out
    
    
class VariableDendriticCircuitSomaBias(nn.Module):
    def __init__(self, hidden_cfg, num_in, num_out, bias=False):
        super(VariableDendriticCircuitSomaBias, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.nonlin = nn.ReLU()
        # self.nonlin = nn.Sigmoid()
        self.features = self._make_layers(hidden_cfg, bias)

    def _make_layers(self, cfg_layers, bias):
        layers = []
        cfg_layers.append(self.num_out)
        num_units_in_layer = np.cumprod(cfg_layers[::-1])[::-1]
        layers += [
            NoiseConditionalLinearConcat1(self.num_in, num_units_in_layer[0], bias),
        ]

        # intermediate layers have no bias
        for i in range(len(cfg_layers)-2):
            layers += [
                self.nonlin,
                DendriticBranchLayer(num_units_in_layer[i], num_units_in_layer[i+1], cfg_layers[i]),
            ]
        
        # final layer has bias
        ii = len(cfg_layers)-2
        layers += [
            self.nonlin, 
            DendriticBranchLayerSomaBias(num_units_in_layer[ii], num_units_in_layer[ii+1], cfg_layers[ii])
        ]
        return MySequential(*layers)

    def forward(self, x, t):
        out = self.features(x, t)
        return out
        


# ---------------------------------------------------------------------------- #
#                       class conditional diffusion model                      #
# ---------------------------------------------------------------------------- #
class ClassConditionalLinear1(nn.Module):
    '''
    A custom linear layer embedding with time and class information.
    '''
    def __init__(self, num_in, num_out, n_steps, n_classes):
        super(ClassConditionalLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.n_steps = n_steps
        self.n_classes = n_classes

        self.linear = nn.Linear(self.num_in, self.num_out)
        self.time_embed = nn.Embedding(self.n_steps, self.num_out//2)
        self.class_embed = nn.Embedding(self.n_classes, self.num_out//2)

        self.time_embed.weight.data.uniform_()
        self.class_embed.weight.data.uniform_()

    def forward(self, x, t, c):
        gamma = self.time_embed(t)
        kappa = self.class_embed(c)
        out = self.linear(x)

        embeddings = torch.cat((gamma, kappa), dim=1)
        out = embeddings * out

        return out



class ClassConditionalLinear(nn.Module):
    '''
    A custom linear layer embedding with time and class information.
    '''
    def __init__(self, num_in, num_out, n_steps, n_classes):
        super(ClassConditionalLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.n_steps = n_steps
        self.n_classes = n_classes

        self.linear = nn.Linear(self.num_in, self.num_out)
        self.time_embed = nn.Embedding(self.n_steps, self.num_out//2)
        self.class_embed = nn.Embedding(self.n_classes, self.num_out//2)

        self.time_embed.weight.data.uniform_()
        self.class_embed.weight.data.uniform_()

    def forward(self, x, t, c):
        gamma = self.time_embed(t)
        kappa = self.class_embed(c)
        out = self.linear(x)

        embeddings = torch.cat((gamma, kappa), dim=1)
        out = embeddings * out

        return out


class TimeAndClassEmbeddedNoiseEstimator(nn.Module):
    def __init__(self, n_steps, num_hidden, n_classes):
        super(TimeAndClassEmbeddedNoiseEstimator, self).__init__()
        self.condlin1 = ClassConditionalLinear(2, num_hidden, n_steps, n_classes)
        self.condlin2 = ClassConditionalLinear(num_hidden, num_hidden, n_steps, n_classes)
        self.condlin3 = ClassConditionalLinear(num_hidden, num_hidden, n_steps, n_classes)
        self.linear = nn.Linear(num_hidden, 2)
        self.nonlin = nn.ReLU()

        # self.class_embed = nn.Embedding(n_classes, self.num_out)
        
    def forward(self, x, t, c):
        '''
        x is the image
        t is the timestep
        c is the class label
        '''
        x = self.condlin1(x, t, c)
        x = self.nonlin(x)

        x = self.condlin2(x, t, c)
        x = self.nonlin(x)

        x = self.condlin3(x, t, c)
        x = self.nonlin(x)

        x = self.linear(x)

        return x
    

# ---------------------------------------------------------------------------- #
#    simple noise estimator with one-hot class label and concat noise level    #
# ---------------------------------------------------------------------------- #

class NoiseAndClassConditionalLinearConcat(nn.Module):
    '''
    A custom module. Defines a linear matrix and an embedding matrix. 
    '''
    def __init__(self, num_in, num_out, num_classes):
        super(NoiseAndClassConditionalLinearConcat, self).__init__()
        self.num_in = num_in  # 2 for a 2 pixel image
        self.num_out = num_out  # size of the hidden layer, e.g. 32
        self.num_classes = num_classes
        
        self.class_embed = nn.Embedding(self.n_classes, self.num_out//2)
        self.class_embed.weight.data.uniform_()
        
        self.linear = nn.Linear(num_in + 1, num_out)  # A normal linear layer, 3x32
        
    def forward(self, x, t, c):
        t = t.float().view(len(t), 1)
        
        out = torch.cat((x, t), dim=1)
        out = self.linear(out)
        return out    
    

class NoiseAndClassConditionalEstimatorConcat(nn.Module):
    def __init__(self, num_hidden):
        super(NoiseAndClassConditionalEstimatorConcat, self).__init__()
        # num_hidden = 32
        self.condlin1 = NoiseAndClassConditionalLinearConcat(2, num_hidden)
        self.condlin2 = NoiseAndClassConditionalLinearConcat(num_hidden, num_hidden)
        self.condlin3 = NoiseAndClassConditionalLinearConcat(num_hidden, num_hidden)
        # self.condlin4 = NoiseConditionalLinearConcat(num_hidden, num_hidden)
        self.linear = nn.Linear(num_hidden, 2)
        self.nonlin = nn.ReLU()
        # self.nonlin = nn.Softplus()
        
    def forward(self, x, t):
        '''
        x is the image
        t is the timestep
        '''
        x = self.condlin1(x, t, c)
        x = self.nonlin(x)

        x = self.condlin2(x, t, c)
        x = self.nonlin(x)

        x = self.condlin3(x, t, c)
        x = self.nonlin(x)
        
        x = self.linear(x)

        return x
    



# ---------------------------------------------------------------------------- #
#              A classifier model for the simple manifold dataset              #
# ---------------------------------------------------------------------------- #
class Classifier(nn.Module):
    '''
    A simple 2 layer feedforward classifier model for classifying 2D manifolds
    '''
    def __init__(self, num_in, num_hidden, num_classes):
        super(Classifier, self).__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        
        self.linear1 = nn.Linear(self.num_in, self.num_hidden)
        self.linear2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.linear3 = nn.Linear(self.num_hidden, self.num_classes)
        self.nonlin = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)        
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.nonlin(out)
        
        out = self.linear2(out)
        out = self.nonlin(out)
        
        out = self.linear3(out)
        out = self.softmax(out)
        return out


class NoisyImageClassifierWithTimeEmbedding(nn.Module):
    '''
    A simple 2 layer feedforward classifier model for classifying noisy images from 2D manifolds.
    Takes in explicit noise level information in the forward method
    '''
    def __init__(self, num_in, num_hidden, num_classes, num_steps):
        super(NoisyImageClassifierWithTimeEmbedding, self).__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        
        self.linear1 = nn.Linear(self.num_in, self.num_hidden)
        self.linear2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.linear3 = nn.Linear(self.num_hidden, self.num_classes)
        self.nonlin = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self.embed = nn.Embedding(num_steps, num_hidden)  # the embedding layer is a lookup table that encodes side information like `t`
        self.embed.weight.data.uniform_()
        
    def forward(self, x, t): 
        out = self.linear1(x)
        time_embedding = self.embed(t)
        out = time_embedding * out
        out = self.nonlin(out)
        
        out = self.linear2(out)
        out = self.nonlin(out)
        
        out = self.linear3(out)
        out = self.softmax(out)
        return out


class NoisyImageClassifierNoTimeEmbedding(nn.Module):
    '''
    A simple 2 layer feedforward classifier model for classifying noisy images from 2D manifolds.
    There is no noise level information fed into the model. 
    '''
    def __init__(self, num_in, num_hidden, num_classes, num_steps):
        super(NoisyImageClassifierNoTimeEmbedding, self).__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        
        self.linear1 = nn.Linear(self.num_in, self.num_hidden)
        self.linear2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.linear3 = nn.Linear(self.num_hidden, self.num_classes)
        self.nonlin = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):    
        out = self.linear1(x)
        out = self.nonlin(out)
        
        out = self.linear2(out)
        out = self.nonlin(out)
        
        out = self.linear3(out)
        out = self.softmax(out)
        return out
    


class LargeNoisyImageClassifier(nn.Module):
    '''
    A simple 2 layer feedforward classifier model for classifying noisy images from 2D manifolds.
    Takes in explicit noise level information in the forward method
    '''
    def __init__(self, num_in, num_hidden, num_classes, num_steps):
        super(LargeNoisyImageClassifier, self).__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        
        self.linear1 = nn.Linear(self.num_in, self.num_hidden)
        self.linear2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.linear3 = nn.Linear(self.num_hidden, self.num_hidden)
        self.linear4 = nn.Linear(self.num_hidden, self.num_classes)
        self.nonlin = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self.embed = nn.Embedding(num_steps, num_hidden)  # the embedding layer is a lookup table that encodes side information like `t`
        self.embed.weight.data.uniform_()
        
    def forward(self, x, t): 
        out = self.linear1(x)
        time_embedding = self.embed(t)
        out = time_embedding * out
        out = self.nonlin(out)
        
        out = self.linear2(out)
        out = self.nonlin(out)
        
        out = self.linear3(out)
        out = self.nonlin(out)
        
        out = self.linear4(out)
        out = self.softmax(out)
        return out
    
    
    
# %%

# ------------------------- Stochastic Neural Network ------------------------ #
# betas = forward_process(num_steps, device, schedule)[0]

# define the model
from utils import extract
class StochasticLayer(nn.Module):
    '''
    A custom module. Defines a linear matrix and an embedding matrix. 
    '''
    def __init__(self, num_in, num_out, betas):
        super(StochasticLayer, self).__init__()
        self.num_in = num_in  # 2 for a 2 pixel image
        self.num_out = num_out  # size of the hidden layer, e.g. 32
        self.linear = nn.Linear(num_in + 1, num_out)  # A normal linear layer, 3x32
        self.nonlin = nn.ReLU()
        self.betas = nn.Parameter(betas, requires_grad=False)
        
    def forward(self, x, t):        
        out = torch.cat((x, t), dim=1)
        out = self.linear(out)
        out = self.nonlin(out)
        # print(x.shape)  # 128 x 2 
        # print(out.shape)  # 128 x 42
        # print(t.shape)  # 128 x 1
        
        sigma_t = extract(self.betas.reshape(-1, 1), t.long(), t).sqrt()
        # print(sigma_t.shape)  # 128 x 1
        out = out + sigma_t * torch.randn_like(out)
        return out
    

class SNN(nn.Module):
    
    def __init__(self, num_hidden, betas):
        super(SNN, self).__init__()
        self.condlin1 = StochasticLayer(2, num_hidden, betas)
        self.condlin2 = StochasticLayer(num_hidden, num_hidden, betas)
        self.condlin3 = StochasticLayer(num_hidden, num_hidden, betas)
        self.linear = nn.Linear(num_hidden, 2)
        
    def forward(self, x, t):
        '''
        x is the image
        t is the timestep
        '''
        t = t.float().view(len(t), 1)
        
        x = self.condlin1(x, t)
        x = self.condlin2(x, t)
        x = self.condlin3(x, t)
        x = self.linear(x)
        
        return x
    
    
