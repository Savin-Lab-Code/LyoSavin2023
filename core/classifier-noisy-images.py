#%%
%load_ext autoreload
%autoreload 2

#%%
# -------------------- let's train and test the classifier ------------------- #
import matplotlib.pyplot as plt
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')

#%%
# ------------------------------- loading data ------------------------------- #
from generate_data import load_trimodal_data, q_x, construct_noisy_dataset
from utils import forward_process
sample_size = int(1e3)

# offsets = [[0,0], [1,0], [1,1]]
# offsets=[[0,0], [4,0], [2,4]]
offsets = [[0.5,0.5], [0,1.5], [2,1]]
dataset, test_dataset = load_trimodal_data(sample_size, offsets, train_test_split=True, add_class_label=True)

num_steps = 100
_, _, _, _, alphas_bar_sqrt, _, one_minus_alphas_prod_sqrt = forward_process(num_steps, device)

# this flag determines whether our classifier receives explicit information about the noise level (t)
add_noise_info = True

noisy_dataset = construct_noisy_dataset(dataset, num_steps, alphas_bar_sqrt, one_minus_alphas_prod_sqrt, add_noise_info=add_noise_info)
noisy_test_dataset = construct_noisy_dataset(test_dataset, num_steps, alphas_bar_sqrt, one_minus_alphas_prod_sqrt, add_noise_info=add_noise_info)

print('noisy_dataset', noisy_dataset.shape)
print('one example', noisy_dataset[0])
print('one example', noisy_dataset[2550])

#%%
# -------------------------------- load model -------------------------------- #
from models import NoisyImageClassifierWithTimeEmbedding, NoisyImageClassifierNoTimeEmbedding
import torch.optim as optim
from utils import calc_classifier_loss

dataset = noisy_dataset.to(device)
test_dataset = noisy_test_dataset.to(device)
num_in = 2
num_hidden = 20
num_classes = 3

if add_noise_info:
    model = NoisyImageClassifierWithTimeEmbedding(num_in, num_hidden, num_classes, num_steps)
else:
    model = NoisyImageClassifierNoTimeEmbedding(num_in, num_hidden, num_classes, num_steps)
model.to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
batch_size = 128

#%%
# --------------------------------- training --------------------------------- #
model.train()
num_epochs = 1000
print(f'The model has {num_hidden} hidden units.')
for t in range(num_epochs):
    permutation = torch.randperm(dataset.size()[0], device=device)
    # generates an array of integers (of dataset size) and randomly shuffles the order
    
    for i in range(0, dataset.size()[0], batch_size):
        indices = permutation[i:i+batch_size]  # random integer values
        batch_x = dataset[indices]  # selects random samples from the dataset
        
        loss = calc_classifier_loss(model, batch_x, num_classes)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    if (t % 100 == 0):
        print('t', t)
        print('loss', loss.item())

print('training is over at last')

# %%
# ------------------------------ test the model ------------------------------ #
from utils import get_classifier_accuracy

model.eval()
acc = get_classifier_accuracy(model, test_dataset)
print(f'The accuracy of the model is {acc * 100:.2f}%.')

# %%
# -------------------------- save the model weights -------------------------- #
from utils import save_model_weights, save_model_description
import os

if add_noise_info == True:
    model_name = 'noisy-image-classifier-with-noise-info'
elif add_noise_info == False:
    model_name = 'noisy-image-classifier-no-noise-info'

model_number = 3

# save the model details in separate file
description = {
    'model_name': model_name,
    'model_number': model_number,
    'num_in': num_in,
    'num_classes': num_classes,
    'num_hidden': num_hidden,
    'num_steps': num_steps,
    'sample_size': f'{sample_size:.0e}',
    'accuracy_on_test_data': f'{acc * 100:.2f}%',
}

# save the model state dict
save_model_weights(model, model_name, model_number)
save_model_description(model_name, model_number, description)


# %%
