# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
import yaml

#device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


# %%
# Batch Size
batch_size = 64

# Set random seed for reproducibility
torch.manual_seed(0)

# Transformations
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Set random seed for data loader
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


# %%
def calculate_conv_output_size(image_height, image_width, conv_layers):
    height, width = image_height, image_width

    for layer in conv_layers:
        if not isinstance(layer, nn.Conv2d):
            continue
        kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
        padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
        height = ((height + (2 * padding) - kernel_size) // stride) + 1
        width = ((width + (2 * padding) - kernel_size) // stride) + 1
    num_output_channels = conv_layers[-3].out_channels
    return height * width * num_output_channels


class Encoder(nn.Module):
    def __init__(self,in_channels, hidden_dims, latent_dim, Height=32, Width=32,seed=42):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        modules = []
        for hidden_dim in self.hidden_dims:
            modules.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
            modules.append(nn.LeakyReLU(negative_slope=0.2))
            in_channels = hidden_dim
        modules.append(nn.Flatten())
        self.output_shape = calculate_conv_output_size(Height, Width, modules)
        self.encoder = nn.Sequential(*modules)
        self.init_weights(seed)
    def forward(self, x):
        return self.encoder(x)
    def get_output_shape(self, image_height, image_width):
        return self.output_shape
    def init_weights(self, seed=42):
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, decoder_outchannels, seed=42):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append(nn.LeakyReLU(negative_slope=0.2))
            modules.append(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[i],
                    out_channels=hidden_dims[i+1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                )
            )
            modules.append(nn.LeakyReLU(negative_slope=0.2))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dims[-1],
                out_channels=hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                in_channels=hidden_dims[-1],out_channels=decoder_outchannels,kernel_size=3,padding=1
            ),
            nn.Tanh()
        )
        self.init_weights(seed)

    def forward(self, x):
        x = self.decoder(x)
        return self.final_layer(x)
    
    def init_weights(self, seed=42):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class Discriminator(nn.Module):
    def __init__(self, latent_dim=100, seed=42):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        modules = [
            nn.Linear(self.latent_dim, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ]
        self.discriminator = nn.Sequential(*modules)

    def forward(self, x):
        return self.discriminator(x)

    def init_weights(self):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                init.constant_(m.bias, 0)



class Conditional_Variational_AutoEncoder(nn.Module):
    def __init__(self, beta=1.0, in_channels=3, hidden_dims=[32, 64, 128, 256, 512], latent_dim=100, num_classes=10, image_height=32, image_width=32, seed=42):
        super(Conditional_Variational_AutoEncoder, self).__init__()
        self.beta = beta
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_height = image_height
        self.image_width = image_width

        self.encoder = Encoder(in_channels+1, hidden_dims, latent_dim,self.image_height,self.image_width) # +1 for the label
        encoder_output_shape = self.encoder.get_output_shape(image_height, image_width)
        self.fc_mu = nn.Linear(encoder_output_shape, latent_dim)
        self.fc_var = nn.Linear(encoder_output_shape, latent_dim)

        self.label_embedder = nn.Linear(num_classes, image_height*image_width)
        self.image_embedder = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim+num_classes,self.hidden_dims[0], encoder_output_shape)
        self.decoder = Decoder(latent_dim+num_classes, self.hidden_dims, in_channels)

        self.init_weights(seed)

    def forward(self, Images, Labels):
        # Concatenate the label embedding and the image
        one_hot_labels = F.one_hot(Labels.to(torch.int64), num_classes=self.num_classes).float()
        embedded_labels = self.label_embedder(one_hot_labels).view(-1, self.image_height, self.image_width).unsqueeze(1)
        embedded_images = self.image_embedder(Images)
        x = torch.cat((embedded_labels, embedded_images), dim=1)
        # Encode the concatenated image
        encoded = self.encoder(x)
        # Get the mean and variance
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        # Get the latent vector
        z = self.reparameterize(mu, log_var)
        # Concatenate the latent vector and the label
        labelled_z = torch.cat([z, one_hot_labels], dim=1)
        # Decode the concatenated latent vector and label
        embedded_decoder_input = self.decoder_input(labelled_z)
        embedded_decoder_input = embedded_decoder_input.view(-1, self.encoder.output_shape, 1, 1)
        decoded_images = self.decoder(embedded_decoder_input)
        return decoded_images, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + self.beta*KLD, BCE, KLD
    
    def sample(self, num_samples, label):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        one_hot_labels = F.one_hot(label.to(torch.int64), num_classes=self.num_classes).float()
        labelled_z = torch.cat([z, one_hot_labels], dim=1)
        embedded_decoder_input = self.decoder_input(labelled_z)
        embedded_decoder_input = embedded_decoder_input.view(-1, self.encoder.output_shape, 1, 1)
        decoded_images = self.decoder(embedded_decoder_input)
        return decoded_images
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def init_weights(self, seed=42):
        torch.manual_seed(seed)
        self.encoder.init_weights(seed)
        self.decoder.init_weights(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


# %%
def yaml_read_config(config_name):
    with open(config_name, 'r') as f:
        config = yaml.safe_load(f)
    return config

# %%
def run(config_name):
    config = yaml_read_config(config_name)
    model_config = config['model_config']
    training_config = config['training_config']
    Beta = training_config['beta']

    model = Conditional_Variational_AutoEncoder(beta=Beta, in_channels=model_config['in_channels'], hidden_dims=model_config['hidden_dims'], latent_dim=model_config['latent_dim'], num_classes=model_config['num_classes'], image_height=model_config['image_height'], image_width=model_config['image_width']).to(device)
    model.init_weights(training_config['seed'])

    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_config['milestones'], gamma=training_config['gamma'])
    
    Epochs = training_config['epochs']

    tb = None
    if training_config['tb_verbose']:
        tb = SummaryWriter(training_config['tb_log_dir'])


    train_reconstruction_loss = []
    train_kl_loss = []
    train_loss = []
    test_reconstruction_loss = []
    test_kl_loss = []
    test_loss = []

    '''
    init_test_loss = 0
    init_test_reconstruction_loss = 0
    init_test_kl_loss = 0
    for batch_idx, (Images, Labels) in enumerate(test_loader):
        Images, Labels = Images.to(device), Labels.to(device)
        decoded_images, mu, log_var = model(Images, Labels)
        loss, reconstruction_loss, kl_loss = model.loss_function(decoded_images, Images, mu, log_var)
        init_test_loss += loss.item()
        init_test_reconstruction_loss += reconstruction_loss.item()
        init_test_kl_loss += kl_loss.item()
    init_test_loss /= len(test_loader.dataset)
    init_test_reconstruction_loss /= len(test_loader.dataset)
    init_test_kl_loss /= len(test_loader.dataset)
    print("Initial Test Loss: ", init_test_loss)
    print("Initial Test Reconstruction Loss: ", init_test_reconstruction_loss)
    print("Initial Test KL Loss: ", init_test_kl_loss)
    '''

    for epoch in range(Epochs):
        print("Epoch: ", epoch + 1)
        model.train()
        train_loss_per_epoch = 0
        train_reconstruction_loss_per_epoch = 0
        train_kl_loss_per_epoch = 0
        for batch_idx, (Images, Labels) in enumerate(train_loader):
            Images, Labels = Images.to(device), Labels.to(device)
            optimizer.zero_grad()
            decoded_images, mu, log_var = model(Images, Labels)
            loss, reconstruction_loss, kl_loss = model.loss_function(decoded_images, Images, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss_per_epoch += loss.item()
            train_reconstruction_loss_per_epoch += reconstruction_loss.item()
            train_kl_loss_per_epoch += kl_loss.item()
        scheduler.step()
        train_loss_per_epoch /= len(train_loader.dataset)
        train_reconstruction_loss_per_epoch /= len(train_loader.dataset)
        train_kl_loss_per_epoch /= len(train_loader.dataset)
        train_loss.append(train_loss_per_epoch)
        train_reconstruction_loss.append(train_reconstruction_loss_per_epoch)
        train_kl_loss.append(train_kl_loss_per_epoch)
        model.eval()
        test_loss_per_epoch = 0
        test_reconstruction_loss_per_epoch = 0
        test_kl_loss_per_epoch = 0
        with torch.no_grad():
            for batch_idx, (Images, Labels) in enumerate(test_loader):
                Images, Labels = Images.to(device), Labels.to(device)
                decoded_images, mu, log_var = model(Images, Labels)
                loss, reconstruction_loss, kl_loss = model.loss_function(decoded_images, Images, mu, log_var)
                test_loss_per_epoch += loss.item()
                test_reconstruction_loss_per_epoch += reconstruction_loss.item()
                test_kl_loss_per_epoch += kl_loss.item()
        test_loss_per_epoch /= len(test_loader.dataset)
        test_reconstruction_loss_per_epoch /= len(test_loader.dataset)
        test_kl_loss_per_epoch /= len(test_loader.dataset)
        test_loss.append(test_loss_per_epoch)
        test_reconstruction_loss.append(test_reconstruction_loss_per_epoch)
        test_kl_loss.append(test_kl_loss_per_epoch)

        print("Test Loss: ", test_loss_per_epoch)
        print("Test Reconstruction Loss: ", test_reconstruction_loss_per_epoch)
        print("Test KL Loss: ", test_kl_loss_per_epoch)

        print("Train Loss: ", train_loss_per_epoch)
        print("Train Reconstruction Loss: ", train_reconstruction_loss_per_epoch)
        print("Train KL Loss: ", train_kl_loss_per_epoch)

        if tb is not None:
            tb.add_scalar("Train Loss", train_loss_per_epoch, epoch)
            tb.add_scalar("Train Reconstruction Loss", train_reconstruction_loss_per_epoch, epoch)
            tb.add_scalar("Train KL Loss", train_kl_loss_per_epoch, epoch)
            
            tb.add_scalar("Test Loss", test_loss_per_epoch, epoch)
            tb.add_scalar("Test Reconstruction Loss", test_reconstruction_loss_per_epoch, epoch)
            tb.add_scalar("Test KL Loss", test_kl_loss_per_epoch, epoch)
    
    
    model.save('CVAE_beta={}.pth'.format(Beta))

    for test_x, test_y in test_loader:
        test_x = test_x.to(device)
        test_y = test_y.to(device)

    img_gen, mu, logvar = model(Images = test_x, Labels = test_y)

    for i in range(5):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(test_x[i].cpu().detach().numpy().transpose(1, 2, 0))
        ax[0].set_title("Original Image")
        ax[1].imshow(img_gen[i].cpu().detach().numpy().transpose(1, 2, 0))
        ax[1].set_title("Generated Image")

        plt.show()


