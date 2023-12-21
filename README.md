# STCS6701-Final-Project

The `code` folder contains four models: CVAE (`CVAE`), Prior-generating $\beta$-AAE (`betaAAE_prior`), Semi-supervised $\beta$-AAE (`betaAAE_semisupervised`) and Denoising $\beta$-AAE (`betaAAE_denoise`). \
\
For each model, use the `run.ipynb` file in the corresponding folder to customize the needed hyperparameters (`Beta`, `Noise_Var` or both) by editing the `generate_config_beta_noise` function, and then run the training and testing process.
