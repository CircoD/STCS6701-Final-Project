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
#from model import Discriminator
import torch.nn.init as init
import itertools
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

'''
def make_fake_latent_vector(latent_vector, one_hot_labels):
    fake_latent_vector = torch.randn_like(latent_vector)
    labelled_fake_latent_vector = torch.cat([fake_latent_vector, one_hot_labels], dim=1)
    return labelled_fake_latent_vector
'''


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
        self.fc_mu = nn.Linear(self.output_shape, self.latent_dim)
        self.fc_var = nn.Linear(self.output_shape, self.latent_dim)
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        z = self.reparameterize(mu, log_var)
        return z
    def get_output_shape(self, image_height, image_width):
        return self.output_shape
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + (eps * std)

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

    def init_weights(self, ):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                init.constant_(m.bias, 0)


# %%
class Adversarial_Variational_AutoEncoder(nn.Module):
    def __init__(self, beta=1.0, in_channels=3, hidden_dims=[32, 64, 128, 256, 512], latent_dim=100, image_height=32, image_width=32, seed=42):
        super(Adversarial_Variational_AutoEncoder, self).__init__()
        self.beta = beta
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims

        self.latent_dim = latent_dim
        self.image_height = image_height
        self.image_width = image_width

        self.encoder = Encoder(in_channels, hidden_dims, latent_dim,self.image_height,self.image_width)
        encoder_output_shape = self.encoder.get_output_shape(image_height, image_width)
        # self.label_embedder = nn.Linear(num_classes, image_height*image_width)
        self.image_embedder = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim,self.hidden_dims[0], encoder_output_shape)
        self.decoder = Decoder(latent_dim, self.hidden_dims, in_channels)
        self.init_weights(seed)

    def forward(self, Images):
        # Concatenate the label embedding and the image
        # one_hot_labels = F.one_hot(Labels.to(torch.int64), num_classes=self.num_classes).float()
        z = self.encoder(Images)
        # Concatenate the latent vector and the label
        # labelled_z = torch.cat([z, one_hot_labels], dim=1)
        # Decode the concatenated latent vector and label
        embedded_decoder_input = self.decoder_input(z)
        embedded_decoder_input = embedded_decoder_input.view(-1, self.encoder.output_shape, 1, 1)
        decoded_images = self.decoder(embedded_decoder_input)
        return decoded_images, z
    
    
    
    #def reparameterize(self, mu, log_var):
        #std = torch.exp(0.5*log_var)
        #eps = torch.randn_like(std)
        #return mu + eps*std

    def sample(self, num_samples, label):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        # one_hot_labels = F.one_hot(label.to(torch.int64), num_classes=self.num_classes).float()
        # labelled_z = torch.cat([z, one_hot_labels], dim=1)
        embedded_decoder_input = self.decoder_input(z)
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
    noise_var = training_config['noise_var']

    autoencoder = Adversarial_Variational_AutoEncoder(beta=Beta, in_channels=model_config['in_channels'], hidden_dims=model_config['hidden_dims'], latent_dim=model_config['latent_dim'], image_height=model_config['image_height'], image_width=model_config['image_width']).to(device)
    discriminator = Discriminator(latent_dim=model_config['latent_dim']).to(device)
    autoencoder.init_weights(training_config['seed'])
    discriminator.init_weights()

    Generator_optimizer = optim.Adam(autoencoder.parameters(), lr=training_config['learning_rate'])
    Discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=training_config['learning_rate'])

    Generator_optimizer = optim.Adam(autoencoder.parameters(), lr=training_config['learning_rate'])
    Discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=training_config['learning_rate'])

    Generator_scheduler = optim.lr_scheduler.MultiStepLR(Generator_optimizer, milestones=training_config['milestones'], gamma=training_config['gamma'])
    Discriminator_scheduler = optim.lr_scheduler.MultiStepLR(Discriminator_optimizer, milestones=training_config['milestones'], gamma=training_config['gamma'])

    Epochs = training_config['epochs']

    tb = None
    if training_config['tb_verbose']:
        tb = SummaryWriter(training_config['tb_log_dir'])
    

    train_reconstruction_loss = []
    train_validity_loss = []
    train_generator_loss = []
    train_true_loss = []
    train_fake_loss = []
    train_discriminator_loss = []

    test_reconstruction_loss = []
    test_validity_loss = []
    test_generator_loss = []
    test_true_loss = []
    test_fake_loss = []
    test_discriminator_loss = []


    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(Epochs):
            print("Epoch: ", epoch + 1)

            train_reconstruction_loss_per_epoch = 0
            train_validity_loss_per_epoch = 0
            train_generator_loss_per_epoch = 0
            train_true_loss_per_epoch = 0
            train_fake_loss_per_epoch = 0
            train_discriminator_loss_per_epoch = 0

            discriminator.eval()
            autoencoder.train()

            for batch_idx, (Images, Labels) in enumerate(train_loader):
                Images, Labels = Images.to(device), Labels.to(device)
                decoded_images, latent_vector = autoencoder(Images)
                fake_images = Images + noise_var*torch.randn_like(Images)
                fake_decoded_images, fake_latent_vector = autoencoder(fake_images)
                reconstruction_loss = F.mse_loss(decoded_images, Images, reduction='sum')
                discriminator_validity = discriminator(latent_vector)
                fake_discriminator_validity = discriminator(fake_latent_vector)
                real_validity = torch.ones_like(discriminator_validity)
                fake_validity = torch.zeros_like(fake_discriminator_validity)
                real_Validity_loss = F.binary_cross_entropy(discriminator_validity, real_validity, reduction='sum')
                fake_Validity_loss = F.binary_cross_entropy(fake_discriminator_validity, fake_validity, reduction='sum')
                Validity_loss = 0.5*(real_Validity_loss + fake_Validity_loss)

                Generator_loss = reconstruction_loss - Beta*Validity_loss
                Generator_optimizer.zero_grad()
                Generator_loss.backward(retain_graph=True)
                Generator_optimizer.step()

                train_reconstruction_loss_per_epoch += reconstruction_loss.item()
                train_validity_loss_per_epoch += Validity_loss.item()
                train_generator_loss_per_epoch += Generator_loss.item()

            autoencoder.eval()
            discriminator.train()

            for batch_idx, (Images, Labels) in enumerate(train_loader):
                Images, Labels = Images.to(device), Labels.to(device)
                decoded_images, latent_vector = autoencoder(Images)
                fake_images = Images + noise_var*torch.randn_like(Images)
                fake_decoded_images, fake_latent_vector = autoencoder(fake_images)
                real_discr_output = discriminator(latent_vector)
                fake_discr_output = discriminator(fake_latent_vector)
                real_labels = torch.ones_like(real_discr_output)
                fake_labels = torch.zeros_like(fake_discr_output)
                real_loss = F.binary_cross_entropy(real_discr_output, real_labels, reduction='sum')
                fake_loss = F.binary_cross_entropy(fake_discr_output, fake_labels, reduction='sum')
                Discriminator_loss = 0.5*(real_loss + fake_loss)

                Discriminator_optimizer.zero_grad()
                Discriminator_loss.backward()
                Discriminator_optimizer.step()

                #log the losses
                train_true_loss_per_epoch += real_loss.item()
                train_fake_loss_per_epoch += fake_loss.item()
                train_discriminator_loss_per_epoch += Discriminator_loss.item()

            train_reconstruction_loss_per_epoch /= len(train_loader.dataset)
            train_validity_loss_per_epoch /= len(train_loader.dataset)
            train_generator_loss_per_epoch /= len(train_loader.dataset)
            train_true_loss_per_epoch /= len(train_loader.dataset)
            train_fake_loss_per_epoch /= len(train_loader.dataset)
            train_discriminator_loss_per_epoch /= len(train_loader.dataset)
            
            test_reconstruction_loss_per_epoch = 0
            test_validity_loss_per_epoch = 0
            test_generator_loss_per_epoch = 0
            test_true_loss_per_epoch = 0
            test_fake_loss_per_epoch = 0
            test_discriminator_loss_per_epoch = 0

            Generator_scheduler.step()
            Discriminator_scheduler.step()
                
            with torch.no_grad():

                autoencoder.eval()

                for batch_idx, (Images, Labels) in enumerate(test_loader):
                    Images, Labels = Images.to(device), Labels.to(device)
                    decoded_images, latent_vector = autoencoder(Images)
                    fake_images = Images + noise_var*torch.randn_like(Images)
                    fake_decoded_images, fake_latent_vector = autoencoder(fake_images)
                    reconstruction_loss = F.mse_loss(decoded_images, Images, reduction='sum')
                    discriminator_validity = discriminator(latent_vector)
                    fake_discriminator_validity = discriminator(fake_latent_vector)
                    real_validity = torch.ones_like(discriminator_validity)
                    fake_validity = torch.zeros_like(fake_discriminator_validity)
                    real_Validity_loss = F.binary_cross_entropy(discriminator_validity, real_validity, reduction='sum')
                    fake_Validity_loss = F.binary_cross_entropy(fake_discriminator_validity, fake_validity, reduction='sum')
                    Validity_loss = 0.5*(real_Validity_loss + fake_Validity_loss)
                    Generator_loss = reconstruction_loss - Beta*Validity_loss

                    test_reconstruction_loss_per_epoch += reconstruction_loss.item()
                    test_validity_loss_per_epoch += Validity_loss.item()
                    test_generator_loss_per_epoch += Generator_loss.item()

                discriminator.eval()

                for batch_idx, (Images, Labels) in enumerate(test_loader):
                    Images, Labels = Images.to(device), Labels.to(device)
                    decoded_images, latent_vector = autoencoder(Images)
                    fake_images = Images + noise_var*torch.randn_like(Images)
                    fake_decoded_images, fake_latent_vector = autoencoder(fake_images)
                    real_discr_output = discriminator(latent_vector)
                    fake_discr_output = discriminator(fake_latent_vector)
                    real_labels = torch.ones_like(real_discr_output)
                    fake_labels = torch.zeros_like(fake_discr_output)
                    real_loss = F.binary_cross_entropy(real_discr_output, real_labels, reduction='sum')
                    fake_loss = F.binary_cross_entropy(fake_discr_output, fake_labels, reduction='sum')
                    Discriminator_loss = 0.5*(real_loss + fake_loss)
                    
                    #log the losses
                    test_true_loss_per_epoch += real_loss.item()
                    test_fake_loss_per_epoch += fake_loss.item()
                    test_discriminator_loss_per_epoch += Discriminator_loss.item()
                    
            test_reconstruction_loss_per_epoch /= len(test_loader.dataset)
            test_validity_loss_per_epoch /= len(test_loader.dataset)
            test_generator_loss_per_epoch /= len(test_loader.dataset)
            test_true_loss_per_epoch /= len(test_loader.dataset)
            test_fake_loss_per_epoch /= len(test_loader.dataset)
            test_discriminator_loss_per_epoch /= len(test_loader.dataset)


            train_reconstruction_loss.append(train_reconstruction_loss_per_epoch)
            train_validity_loss.append(train_validity_loss_per_epoch)
            train_generator_loss.append(train_generator_loss_per_epoch)
            train_true_loss.append(train_true_loss_per_epoch)
            train_fake_loss.append(train_fake_loss_per_epoch)
            train_discriminator_loss.append(train_discriminator_loss_per_epoch)
        
            test_reconstruction_loss.append(test_reconstruction_loss_per_epoch)
            test_validity_loss.append(test_validity_loss_per_epoch)
            test_generator_loss.append(test_generator_loss_per_epoch)
            test_true_loss.append(test_true_loss_per_epoch)
            test_fake_loss.append(test_fake_loss_per_epoch)
            test_discriminator_loss.append(test_discriminator_loss_per_epoch)



                    
            print("Test Reconstruction Loss: ", test_reconstruction_loss_per_epoch)
            print("Test Discriminator Loss: ", test_discriminator_loss_per_epoch)
            print("Test Generator Loss: ", test_generator_loss_per_epoch)
            print("Test Validity Loss: ", test_validity_loss_per_epoch)
            print("Test True Loss: ", test_true_loss_per_epoch)
            print("Test Fake Loss: ", test_fake_loss_per_epoch)

            print("Train Reconstruction Loss: ", train_reconstruction_loss_per_epoch)
            print("Train Discriminator Loss: ", train_discriminator_loss_per_epoch)
            print("Train Generator Loss: ", train_generator_loss_per_epoch)
            print("Train Validity Loss: ", train_validity_loss_per_epoch)
            print("Train True Loss: ", train_true_loss_per_epoch)
            print("Train Fake Loss: ", train_fake_loss_per_epoch)

            if tb is not None:
                tb.add_scalar('Test Reconstruction Loss', test_reconstruction_loss_per_epoch, epoch)
                tb.add_scalar('Test Discriminator Loss', test_discriminator_loss_per_epoch, epoch)
                tb.add_scalar('Test Generator Loss', test_generator_loss_per_epoch, epoch)
                tb.add_scalar('Test Validity Loss', test_validity_loss_per_epoch, epoch)
                tb.add_scalar('Test True Loss', test_true_loss_per_epoch, epoch)
                tb.add_scalar('Test Fake Loss', test_fake_loss_per_epoch, epoch)

                tb.add_scalar('Train Reconstruction Loss', train_reconstruction_loss_per_epoch, epoch)
                tb.add_scalar('Train Discriminator Loss', train_discriminator_loss_per_epoch, epoch)
                tb.add_scalar('Train Generator Loss', train_generator_loss_per_epoch, epoch)
                tb.add_scalar('Train Validity Loss', train_validity_loss_per_epoch, epoch)
                tb.add_scalar('Train True Loss', train_true_loss_per_epoch, epoch)
                tb.add_scalar('Train Fake Loss', train_fake_loss_per_epoch, epoch)

                tb.add_scalar('Learning Rate', Generator_optimizer.param_groups[0]['lr'], epoch)



    autoencoder.save('betaAAE_denoise_var={noise_var}.beta={Beta}.pth')

    for test_x, test_y in test_loader:
        test_x = test_x.to(device)
        test_y = test_y.to(device)

    img_gen, z = autoencoder(Images = test_x)

    for i in range(5):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(test_x[i].cpu().detach().numpy().transpose(1, 2, 0))
        ax[0].set_title("Original Image")
        ax[1].imshow(img_gen[i].cpu().detach().numpy().transpose(1, 2, 0))
        ax[1].set_title("Generated Image")
        
        plt.show()

                    


