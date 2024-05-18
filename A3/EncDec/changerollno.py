import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import pickle
from sklearn.mixture import GaussianMixture
from EncDec import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# EPOCH = 50
# BATCH_SIZE = 64
# LEARNING_RATE = 0.001

def load_images(path):
    img_label_mapping = {i: [] for i in range(0,10)}
    count = 0
    for img in os.listdir(path):
        key = int(img.split("_")[-1].split(".")[0])
        image = torchvision.io.read_image(os.path.join(path, img))
        # convert image to greyscale
        grey_image = TF.rgb_to_grayscale(image)
        img_label_mapping[key].append(grey_image)
        count += 1
    return img_label_mapping, count

class AlteredMNIST:
    """
    dataset description:
    
    X_I_L.png
    X: {aug=[augmented], clean=[clean]}
    I: {Index range(0,60000)}
    L: {Labels range(10)}
    
    Write code to load Dataset
    """
    # loading clean images
    def __init__(self):
        # self.clean_img_path = "Data\clean" # path to directory containing clean images
        self.clean_img_path= '/kaggle/input/dldatafora3/DLA3/Data/clean'
        # self.aug_img_path = "Data\\aug"  # path to directory containing augmented images
        self.aug_img_path = '/kaggle/input/dldatafora3/DLA3/Data/aug'
        self.clean_images, self.number_of_clean_images = load_images(self.clean_img_path)
        # print('Clean images loaded successfully')
        # make a GMM model for each digit
        self.gmm_models = {i: GaussianMixture(n_components=40, covariance_type='full') for i in range(0,10)}
        # print(len(self.clean_images[0]))

        # training the gmm models
        for i in range(10):
            # Flatten the list of images before fitting GMM
            images = torch.stack(self.clean_images[i], dim=0)
            images = images.view(images.size(0), -1)  # Flatten images
            self.gmm_models[i].fit(images)
            
            # Save the models as pkl files in the directory DlA3/gmm_models
            pickle.dump(self.gmm_models[i], open(f"gmm_model_{i}.pkl", "wb"))
        # print('GMM models saved successfully')
        self.aug_images, self.number_of_augmented_images = load_images(self.aug_img_path)
        # print('Augmented images loaded successfully')
        self.predicted_clean_labels = [[[] for i in range(40)] for j in range(10)]
        self.predicted_aug_labels = [[[] for i in range(40)] for j in range(10)]
        # load the GMM models from the directory DlA3/gmm_models
        for i in range(10):
            self.gmm_models[i] = pickle.load(open(f"gmm_model_{i}.pkl", "rb"))

        # predict labels of clean images
        for i in range(10):
            images = torch.stack(self.clean_images[i], dim=0)
            flattened_images = images.view(images.size(0), -1)
            # print(images.shape)
            predicted_labels = self.gmm_models[i].predict(flattened_images)
            for j in range(len(predicted_labels)):
                self.predicted_clean_labels[i][predicted_labels[j]].append(images[j])
        # print('Predicted labels of clean images successfully done')
        # predict labels of augmented images
        for i in range(10):
            images = torch.stack(self.aug_images[i], dim=0)
            images = images.view(images.size(0), -1)
            predicted_labels = self.gmm_models[i].predict(images)
            for j in range(len(predicted_labels)):
                self.predicted_aug_labels[i][predicted_labels[j]].append(images[j])
        # print('Predicted labels of augmented images successfully done')

        self.clean_images_mapping = []
        # map augmented images to clean images
        for i in range(10):
            for j in range(40):
                for k in range(len(self.predicted_aug_labels[i][j])):
                    self.clean_images_mapping.append((self.predicted_aug_labels[i][j][k], self.predicted_clean_labels[i][j][0]))
        # print('Clean images mapping successfully done')
#         print(len(self.clean_images_mapping))
        # make images into float tensors
        for i in range(len(self.clean_images_mapping)):
            self.clean_images_mapping[i] = (self.clean_images_mapping[i][0].float(), self.clean_images_mapping[i][1].float())
        # print('Images converted to float tensors successfully')
        
    def __len__(self):
        return self.number_of_augmented_images
    
    def __getitem__(self, idx):
        """
        return clean image and augmented image
        """
        return self.clean_images_mapping[idx][1].reshape(1, 28, 28), self.clean_images_mapping[idx][0].reshape(1, 28, 28)
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels))
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    """
    Write code for Encoder ( Logits/embeddings shape must be [batch_size,channel,height,width] )
    """
    # print('encoder called')
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.residual_block1 = ResidualBlock(64, 128, stride=2)
        self.residual_block2 = ResidualBlock(128, 256, stride=2)
        
        self.vae_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.vae_bn1 = nn.BatchNorm2d(64)
        self.vae_relu = nn.ReLU()
        self.vae_residual_block1 = ResidualBlock(64, 128, stride=2)
        self.vae_residual_block2 = ResidualBlock(128, 256, stride=2)
        self.vae_mu = nn.Linear(256*7*7, 256)
        self.vae_log_var = nn.Linear(256*7*7, 256)

    def forward(self, x, vae=False):
        if vae:
            x = self.vae_conv1(x)
            x = self.vae_bn1(x)
            x = self.vae_relu(x)
            x = self.vae_residual_block1(x)
            x = self.vae_residual_block2(x)
            x = x.view(x.size(0), -1)
            mu = self.vae_mu(x)
            log_var = self.vae_log_var(x)
            x = self.reparameterize(mu, log_var)
            return x, mu, log_var
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.residual_block1(x)
            x = self.residual_block2(x)
        return x
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std


class Decoder(nn.Module):
    """
    Write code for decoder here ( Output image shape must be same as Input image shape i.e. [batch_size,1,28,28] )
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.residual_block1 = ResidualBlock(256, 128)
        self.residual_block2 = ResidualBlock(128, 64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.vae_fc1 = nn.Linear(256, 256*7*7)
        # reshape to 256 channels
        self.unflatten = nn.Unflatten(1, (256, 7, 7))
        self.vae_residual_block1 = ResidualBlock(256, 128)
        self.vae_residual_block2 = ResidualBlock(128, 64)
        self.vae_deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.vae_bn1 = nn.BatchNorm2d(32)
        self.vae_relu = nn.ReLU()
        self.vae_deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x, vae=False):
        if vae:
            x = self.vae_fc1(x)
            x = self.unflatten(x)
            x = self.vae_residual_block1(x)
            x = self.vae_residual_block2(x)
            x = self.vae_relu(x)
            x = self.vae_deconv1(x)
            x = self.vae_bn1(x)
            x = self.vae_relu(x)
            x = self.vae_deconv2(x)
            x = torch.sigmoid(x)
        else:
            x = self.residual_block1(x)
            x = self.residual_block2(x)
            x = self.relu(x)
            x = self.deconv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.deconv2(x)
            x = torch.sigmoid(x)  
        return x

class AELossFn:
    """
    Loss function for AutoEncoder Training Paradigm
    """
    # print('ae loss function called')
    def __init__(self):
        pass
    def forward(self, original, reconstructed):
        return F.mse_loss(original, reconstructed)

class VAELossFn:
    """
    Loss function for Variational AutoEncoder Training Paradigm
    """
    # loss fn using mse and kdl loss
    def __init__(self):
        pass
    def forward(self, original, reconstructed, mu, log_var):
        mse_loss = F.mse_loss(original, reconstructed)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return mse_loss + kld_loss
    
def ParameterSelector(E, D):
    """
    Write code for selecting parameters to train
    """
    # print('parameter selector called')
    # print(E)
    # print(D)
    model = torch.nn.Sequential(E, D)
    return model.parameters()

class AETrainer:
    """
    Write code for training AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as AE_epoch_{}.png
    """
    def __init__(self, data, E, D, L, O, gpu):
        if gpu == 'T':
            gpu = 'cuda'
        else:
            gpu = 'cpu'
        E = E.to(gpu)
        D = D.to(gpu)
        self.original_data = data
        # print("AETrainer called")
        
        for epoch in range(EPOCH):
            epoch_loss = 0
            epoch_similarity = 0
            epoch_logits = []
            for minibatch, (data, _) in enumerate(self.original_data):
                # Train the model
                O.zero_grad()
                data = data.to(gpu)
                output = E(data).to(gpu)
                epoch_logits.append(output)
                reconstructed = D(output).to(gpu)
                loss = L.forward(data, reconstructed)
                loss.backward()
                O.step()
                # Calculate similarity
                total_similarity = 0
                for i in range(len(data)):
                    ssim = structure_similarity_index(data[i].cpu(), reconstructed[i].cpu())
                    # if ssim is not nan, add it to total similarity
                    if not torch.isnan(torch.tensor(ssim)):
                        total_similarity += ssim
                
                # similarity = structure_similarity_index(data, reconstructed)
                if minibatch % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch, loss, total_similarity/BATCH_SIZE))
                epoch_loss += loss
                epoch_similarity += total_similarity/BATCH_SIZE
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, epoch_loss/len(self.original_data), epoch_similarity/len(self.original_data)))
            if epoch % 5 == 0:
                # tsne plot of logits of whole data
                logits = torch.cat(epoch_logits, dim=0)
                tsne = TSNE(n_components=3)
                logits = logits.view(logits.size(0), -1)
                logits = logits.cpu()
                logits = logits.detach().numpy()
                # apply pca to logits before tsne
                pca = PCA(n_components=8)
                logits = pca.fit_transform(logits)
                logits = tsne.fit_transform(logits)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(logits[:, 0], logits[:, 1], logits[:, 2])
                plt.savefig(f"AE_epoch_{epoch}.png")
                plt.close()

        # print("AE Training Complete")
        # save the model
        torch.save(E.state_dict(), 'encoder.pth')
        torch.save(D.state_dict(), 'decoder.pth')
        # print("Model saved successfully")
            

class VAETrainer:
    """
    Write code for training Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as VAE_epoch_{}.png
    """
    def __init__(self, data, E, D, L, O, gpu):
        if gpu == 'T':
            gpu = 'cuda'
        else:
            gpu = 'cpu'
        E = E.to(gpu)
        D = D.to(gpu)
        self.original_data = data
        for epoch in range(EPOCH):
            epoch_loss = 0
            epoch_similarity = 0
            epoch_logits = []
            for minibatch, (clean, aug) in enumerate(data):
                clean = clean.to(gpu)
                aug = aug.to(gpu)
                O.zero_grad()
                logits, mu, log_var = E(aug, vae=True)
                epoch_logits.append(logits)
                reconstructed = D(logits, vae=True)
                loss = L.forward(aug, reconstructed, mu, log_var)
                loss.backward()
                O.step()
                total_similarity = 0
                for i in range(len(clean)):
                    ssim = structure_similarity_index(clean[i].cpu(), reconstructed[i].cpu())
                    # if ssim is not nan, add it to total similarity
                    if not torch.isnan(torch.tensor(ssim)):
                        total_similarity += ssim

                if minibatch % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,total_similarity/BATCH_SIZE))
                epoch_loss += loss
                epoch_similarity += total_similarity/BATCH_SIZE
            similarity = epoch_similarity/len(data)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, epoch_loss, similarity))
            if epoch % 5 == 0:
                logits = torch.cat(epoch_logits, dim=0)
                # make 3D TSNE plot of logits of whole data
                tsne = TSNE(n_components=3)
                logits = logits.view(logits.size(0), -1)
                logits = logits.cpu()
                logits = logits.detach().numpy()
                # apply pca to logits before tsne
                pca = PCA(n_components=8)
                logits = pca.fit_transform(logits)
                logits = tsne.fit_transform(logits)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(logits[:, 0], logits[:, 1], logits[:, 2])
                plt.savefig(f"VAE_epoch_{epoch}.png")
                plt.close()
                
        # save the model
        torch.save(E.state_dict(), 'vae_encoder.pth')
        torch.save(D.state_dict(), 'vae_decoder.pth')
        # print("Model saved successfully")


class AE_TRAINED(torch.nn.Module):
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self, gpu):
        super(AE_TRAINED, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.load_state_dict(torch.load('encoder.pth'))
        self.decoder.load_state_dict(torch.load('decoder.pth'))
        self.gpu = 'cuda' 
        self.encoder = self.encoder
        self.decoder = self.decoder

    def from_path(self, sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        # sample and original are filepaths of images
        # load the images
        sample = torchvision.io.read_image(sample).float()
        original = torchvision.io.read_image(original).float()
        # make sample a 4d tensor from a 3d tensor
        sample = sample.unsqueeze(0)
        logits = self.encoder(sample)
        reconstructed = self.decoder(logits)
        if type == "SSIM":
            return structure_similarity_index(original.cpu(), reconstructed.cpu())
        elif type == "PSNR":
            return peak_signal_to_noise_ratio(original.cpu(), reconstructed.cpu())


class VAE_TRAINED(torch.nn.Module):
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self, gpu):
        super(VAE_TRAINED, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.load_state_dict(torch.load('vae_encoder.pth'))
        self.decoder.load_state_dict(torch.load('vae_decoder.pth'))
        self.encoder = self.encoder
        self.decoder = self.decoder

    def from_path(self, sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        sample = torchvision.io.read_image(sample).float()
        sample = sample.unsqueeze(0)
        original = torchvision.io.read_image(original).float()
        logits, mu, log_var = self.encoder(sample, vae=True)
        reconstructed = self.decoder(logits, vae=True)
        if type == "SSIM":
            return structure_similarity_index(original.cpu(), reconstructed.cpu())
        elif type == "PSNR":
            return peak_signal_to_noise_ratio(original.cpu(), reconstructed.cpu())
        

class CVAELossFn():
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    def __init__(self):
        pass
    def forward(self, original, reconstructed, mu, log_var):
        mse_loss = F.mse_loss(original, reconstructed)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return mse_loss + kld_loss

class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    def __init__(self, Data, E, D, CL, O):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        E.to(self.device)
        D.to(self.device)
        self.original_data = Data
        for epoch in range(1):
            epoch_loss = 0
            epoch_similarity = 0
            epoch_logits = []

            for minibatch, (original, augmented) in enumerate(self.original_data):
                original = original.to(self.device)
                augmented = augmented.to(self.device)
                logits, mu, log_var = E(augmented, vae=True)
                epoch_logits.append(logits)
                reconstructed = D(logits, vae=True)
                loss = CL.forward(original, reconstructed, mu, log_var)
                O.zero_grad()
                loss.backward()
                O.step()
                total_similarity = 0
                for i in range(len(original)):
                    ssim = structure_similarity_index(original[i].cpu(), reconstructed[i].cpu())
                    # if ssim is not nan, add it to total similarity
                    if not torch.isnan(torch.tensor(ssim)):
                        total_similarity += ssim

                if minibatch % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,total_similarity/BATCH_SIZE))
                epoch_loss += loss
                epoch_similarity += total_similarity/BATCH_SIZE
            similarity = epoch_similarity/len(self.original_data)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, epoch_loss, similarity))
            if epoch % 5 == 0:
                logits = torch.cat(epoch_logits, dim=0)
                # make 3D TSNE plot of logits of whole data
                tsne = TSNE(n_components=3)
                logits = logits.view(logits.size(0), -1)
                logits = logits.cpu()
                logits = logits.detach().numpy()
                # apply pca to logits before tsne
                pca = PCA(n_components=8)
                logits = pca.fit_transform(logits)
                logits = tsne.fit_transform(logits)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(logits[:, 0], logits[:, 1], logits[:, 2])
                plt.savefig(f'CVAE_epoch_{epoch}.png')
                plt.close()
                # pass
        
        # saving model
        torch.save(E.state_dict(), 'cvae_encoder.pth')
        torch.save(D.state_dict(), 'cvae_decoder.pth')

class CVAE_Generator(torch.nn.Module):
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    
    def save_image(digit, save_path):
        pass

def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()