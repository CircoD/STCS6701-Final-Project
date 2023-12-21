# %%
import yaml
import os

# %%
config_file_path = 'configs/'
if not os.path.exists(config_file_path):
    os.makedirs(config_file_path)

def generate_config(config, config_name):
    with open(config_file_path + config_name, 'w') as f:
        yaml.dump(config, f)

# %%
TO_Override = {
    "model_config": {
        "in_channels": 3,
        "hidden_dims": [32, 64, 128, 256, 512],
        "latent_dim": 128,
        "num_classes": 10,
        "image_height": 32,
        "image_width": 32,

    },

    "training_config": {
        "seed": 42,
        "epochs": 100,
        "beta": 1,
        "learning_rate": 1e-4,
        'tb_verbose': True,
        'milestones': [150,200],
        'gamma': 0.1,
        "seed": 42
    }
}
TO_Override['training_config']['tb_log_dir'] = f"./runs/CVAE_beta={TO_Override['training_config']['beta']}."


config_name = "config.yaml"

# %%
generate_config(TO_Override, config_name)

# %%
def generate_config_beta_noise(Beta=1):
    TO_write = TO_Override.copy()
    # TO_write['beta'] = Beta
    # TO_write['tensorboard_log_dir'] = f"./drive/MyDrive/GR6701/GR6701 Project/runs/AAE_prior_beta={TO_write['training_config']['beta']}."
    TO_write['training_config']['beta'] = Beta
    TO_write['training_config']['tb_log_dir'] = f"./runs/CVAE_beta={TO_write['training_config']['beta']}."
    
    generate_config(TO_write, f"config_beta={Beta}.yaml")

# %%
def yaml_read_config(config_name):
    with open(config_name, 'r') as f:
        config = yaml.safe_load(f)
    return config

# %%



