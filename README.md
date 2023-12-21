# STCS6701 Final Project: Beta Adversarial Autoencoders ($\beta$-AAE)

This repository contains code to implement $\beta$-AAE using PyTorch. \
\
The `code` folder contains four models: CVAE (`CVAE`), Prior-generating $\beta$-AAE (`betaAAE_prior`), Semi-supervised $\beta$-AAE (`betaAAE_semisupervised`) and Denoising $\beta$-AAE (`betaAAE_denoise`). 

## Installing the Dependencies
    pip3 install -r requirements.txt

## Dataset
The CIFAR-10 dataset will be downloaded automatically and will be made available in `./code/<model>/data` directory. 

## Training and Testing
For each model, use the `./code/<model>/run.ipynb` file to customize the hyperparameters (`Beta`, `Noise_Var` or both) by editing the parameters of `generate_config_beta_noise` function and run the training and testing process. 

- The train and test losses will be printed out after each epoch. 
- Five examples of reconstructed test images will be printed out along with the corresponding original images after all epochs are done.

## References
Paper: 
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
- [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644)


Some of the code is based on the following repositories:
- "[PyTorch-VAE]": https://github.com/AntixK/PyTorch-VAE
- "[adversarial-autoencoder]": https://github.com/musyoku/adversarial-autoencoder 



***Note:***
- Using your GPU during training is highly recommended. 
- Each run generates a required config file corresponding to the customized hyperparameters under `.code/<model>/configs/config_<hyperparameters>` directory.
- Each run generates a required tensorboard file under `./code/<model>/runs/<model_and_hyperparameters>` directory.
- Use  `tensorboard --logdir <tensorboard_dir>` to look at train and test losses.



