This repo contains code that implements the model proposed in the paper ``Complex Priors and Flexible Inference in Recurrent Circuits with Dendritic Nonlinearities". 

Here is an overview of the different directories:

- the `demos` directory contains jupyter notebooks that generate figures found in the main text and the supplementary. 
- the `core` directory contains:
  - `saved_arrays` (data used for analysis)
  - `saved_weights` (the saved weights of the trained model)
  - `utils` (utilities used throughout the repo)
  - `models.py` (collection of model architectures)
  - `model_description` (descriptions of the parameters of every trained model)
  - a collection of python scripts for training models or generating samples from models on a high performance computing cluster.

