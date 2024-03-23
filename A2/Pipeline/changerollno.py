import torch
import torchaudio
import torchvision
import torch.nn as nn
from Pipeline import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import gc

"""
Write Code for Downloading Image and Audio Dataset Here
"""
# Image Downloader
image_dataset_downloader = torchvision.datasets.CIFAR10(
    root="data",
    # train=True,
    download=True
)

# Audio Downloader
audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(
    root="data",
    url="speech_commands_v0.02",
    download=True,
    subset="training"
)
size_of_data = len(audio_dataset_downloader)
indices = torch.randperm(size_of_data).tolist()

class ImageDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        val_split = 0.2
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        self.datasplit = split
        if split == "train" or split == "val":
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)
                )
            ])
            
            # Load CIFAR-10 dataset
            full_dataset = torchvision.datasets.CIFAR10(
                root="data",
                train=True,
                download=False,
                transform=self.transform
            )
            
            train_indices, val_indices = train_test_split(
                range(len(full_dataset)),
                test_size=val_split,
                random_state=42,
                shuffle=True
            )
            
            if split == "train":
                self.dataset = torch.utils.data.Subset(full_dataset, train_indices)
            else:
                self.dataset = torch.utils.data.Subset(full_dataset, val_indices)
        

        elif split == "test":
            self.dataset = torchvision.datasets.CIFAR10(
                root="data",
                train=False,
                download=False,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5)
                    )
                ])
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class AudioDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        """
        Write your code here
        """
        self.transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
        self.data = []
        self.labels = []
        self.mapping = {
            "backward": 0,
            "bed": 1,
            "bird": 2,
            "cat": 3,
            "dog": 4,
            "down": 5,
            "eight": 6,
            "five": 7,
            "follow": 8,
            "forward": 9,
            "four": 10,
            "go": 11,
            "happy": 12,
            "house": 13,
            "learn": 14,
            "left": 15,
            "marvin": 16,
            "nine": 17,
            "no": 18,
            "off": 19,
            "on": 20,
            "one": 21,
            "right": 22,
            "seven": 23,
            "sheila": 24,
            "six": 25,
            "stop": 26,
            "three": 27,
            "tree": 28,
            "two": 29,
            "up": 30,
            "visual": 31,
            "wow": 32,
            "yes": 33,
            "zero": 34,
        }
        
        if split == 'train':
            for i in range(int(0.7*size_of_data)):
                waveform, sample_rate, label, speaker_id, utterance_number = audio_dataset_downloader[indices[i]]
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
                waveform = self.transform(waveform)
                # if audio duration is less than 1 second, remove it
                if waveform.shape[1] == 8000:
                    self.data.append(waveform)
                    self.labels.append(self.mapping[label])
        elif split == 'val':
            for i in range(int(0.7*size_of_data), int(0.9*size_of_data)):
                waveform, sample_rate, label, speaker_id, utterance_number = audio_dataset_downloader[indices[i]]
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
                waveform = self.transform(waveform)
                # if audio duration is less than 1 second, remove it
                if waveform.shape[1] == 8000:
                    self.data.append(waveform)
                    self.labels.append(self.mapping[label])
        elif split == 'test':
            for i in range(int(0.9*size_of_data), size_of_data):
                waveform, sample_rate, label, speaker_id, utterance_number = audio_dataset_downloader[indices[i]]
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
                waveform = self.transform(waveform)
                # if audio duration is less than 1 second, remove it
                if waveform.shape[1] == 8000:
                    self.data.append(waveform)
                    self.labels.append(self.mapping[label])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample:
            residual = self.downsample(x)
        x += residual
        x = self.relu(x)
        return x
    
    
class Resnet_Q1(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # make a resnet 18 model using ResidualBlock
        self.in_channels = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        # make a layer which has audio as input 
        self.audio_conv1 = nn.Sequential(
                        nn.Conv1d(1, 64, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm1d(64),
                        nn.ReLU())
        
        self.layer1 = self.make_layer(ResidualBlock, 64, 18)
        # self.layer2 = self.make_layer(ResidualBlock, 128, 4, 2)
        # self.layer3 = self.make_layer(ResidualBlock, 256, 6, 2)
        # self.layer4 = self.make_layer(ResidualBlock, 512, 4, 2)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(4096, 10)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """
        cur_channels = 64
        # image
        # set number of channels to 64
        self.conv1 = nn.Conv2d(3, cur_channels, kernel_size=1, padding=3)
        self.conv2 = nn.Conv2d(cur_channels, cur_channels, kernel_size=3, padding=3)
        self.conv3 = nn.Conv2d(cur_channels, cur_channels, kernel_size=3, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        prev_channels = cur_channels
        cur_channels = math.ceil(cur_channels * 0.65)
        kernel_size = 4
        self.conv4 = nn.Conv2d(prev_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv5 = nn.Conv2d(cur_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        prev_channels = cur_channels
        cur_channels = math.ceil(cur_channels * 0.65)
        kernel_size = math.ceil(kernel_size * 1.25)
        self.conv6 = nn.Conv2d(prev_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv7 = nn.Conv2d(cur_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv8 = nn.Conv2d(cur_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        prev_channels = cur_channels
        cur_channels = math.ceil(cur_channels * 0.65)
        kernel_size = math.ceil(kernel_size * 1.25)
        self.conv9 = nn.Conv2d(prev_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv10 = nn.Conv2d(cur_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv11 = nn.Conv2d(cur_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        prev_channels = cur_channels
        cur_channels = math.ceil(cur_channels * 0.65)
        kernel_size = math.ceil(kernel_size * 1.25)
        self.conv12 = nn.Conv2d(prev_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv13 = nn.Conv2d(cur_channels, cur_channels, kernel_size=kernel_size, padding=4)
        self.conv14 = nn.Conv2d(cur_channels, cur_channels, kernel_size=kernel_size, padding=4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(13, 23)
        self.fc2 = nn.Linear(23, 69)
        self.fc3 = nn.Linear(69, 10)

        # audio
        self.conv1_audio = nn.Conv1d(1, cur_channels, kernel_size=1, padding=3)
        self.conv2_audio = nn.Conv1d(cur_channels, cur_channels, kernel_size=3, padding=3)
        self.conv3_audio = nn.Conv1d(cur_channels, cur_channels, kernel_size=3, padding=3)
        self.pool_audio = nn.MaxPool1d(2, 2)
        prev_channels = cur_channels
        cur_channels = math.ceil(cur_channels * 0.65)
        kernel_size = 4
        self.conv4_audio = nn.Conv1d(prev_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv5_audio = nn.Conv1d(cur_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.pool_audio = nn.MaxPool1d(2, 2)
        prev_channels = cur_channels
        cur_channels = math.ceil(cur_channels * 0.65)
        kernel_size = math.ceil(kernel_size * 1.25)
        self.conv6_audio = nn.Conv1d(prev_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv7_audio = nn.Conv1d(cur_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv8_audio = nn.Conv1d(cur_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.pool_audio = nn.MaxPool1d(2, 2)
        prev_channels = cur_channels
        cur_channels = math.ceil(cur_channels * 0.65)
        kernel_size = math.ceil(kernel_size * 1.25)
        self.conv9_audio = nn.Conv1d(prev_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv10_audio = nn.Conv1d(cur_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv11_audio = nn.Conv1d(cur_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.pool_audio = nn.MaxPool1d(2, 2)
        prev_channels = cur_channels
        cur_channels = math.ceil(cur_channels * 0.65)
        kernel_size = math.ceil(kernel_size * 1.25)
        self.conv12_audio = nn.Conv1d(prev_channels, cur_channels, kernel_size=kernel_size, padding=3)
        self.conv13_audio = nn.Conv1d(cur_channels, cur_channels, kernel_size=kernel_size, padding=4)
        self.conv14_audio = nn.Conv1d(cur_channels, cur_channels, kernel_size=kernel_size, padding=4)
        self.pool_audio = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1_audio = nn.Linear(750, 23)
        self.fc2_audio = nn.Linear(23, 69)
        self.fc3_audio = nn.Linear(69, 35)
        self.relu = nn.ReLU()

    def forward(self, x):
        if (len(x.shape) == 4):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.pool(x)

            x = self.conv4(x)
            x = self.conv5(x)
            x = self.pool(x)

            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.pool(x)

            x = self.conv9(x)
            x = self.conv10(x)
            x = self.conv11(x)
            x = self.pool(x)

            x = self.conv12(x)
            x = self.conv13(x)
            x = self.conv14(x)
            x = self.pool(x)

            x = x.view(-1, 13)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        else:
            x = self.conv1_audio(x)
            x = self.conv2_audio(x)
            x = self.conv3_audio(x)
            x = self.pool_audio(x)

            x = self.conv4_audio(x)
            x = self.conv5_audio(x)
            x = self.pool_audio(x)

            x = self.conv6_audio(x)
            x = self.conv7_audio(x)
            x = self.conv8_audio(x)
            x = self.pool_audio(x)

            x = self.conv9_audio(x)
            x = self.conv10_audio(x)
            x = self.conv11_audio(x)
            x = self.pool_audio(x)

            x = self.conv12_audio(x)
            x = self.conv13_audio(x)
            x = self.conv14_audio(x)
            x = self.pool_audio(x)

            x = self.flatten(x)
            x = self.fc1_audio(x)
            x = self.relu(x)
            x = self.fc2_audio(x)
            x = self.relu(x)
            x = self.fc3_audio(x)

        return x

class CNA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(CNA_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.audio_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        if len(x.shape) == 4:
            return self.layer(x)
        else:
            return self.audio_layer(x)
    

class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_b1, out_b2, out_b3
                 , bias=False):

        super(Inception_Block, self).__init__()
        self.branch1 = nn.Sequential(
            CNA_Block(in_channels, out_b1, 1, 1, 0, bias=bias)
        )
        self.branch1_audio = nn.Sequential(
            CNA_Block(in_channels, out_b1, 1, 1, 0, bias=bias)
        )
        self.branch2 = nn.Sequential(
            CNA_Block(in_channels, out_b2[0], 3, 1, 1, bias=bias),
            CNA_Block(out_b2[0], out_b2[1], 5, 1, 2, bias=bias)
        )
        self.branch2_audio = nn.Sequential(
            CNA_Block(in_channels, out_b2[0], 3, 1, 1, bias=bias),
            CNA_Block(out_b2[0], out_b2[1], 5, 1, 2, bias=bias)
        )
        self.branch3 = nn.Sequential(
            CNA_Block(in_channels, out_b3[0], 3, 1, 1, bias=bias),
            CNA_Block(out_b3[0], out_b3[1], 5, 1, 2, bias=bias)
        )
        self.branch3_audio = nn.Sequential(
            CNA_Block(in_channels, out_b3[0], 3, 1, 1, bias=bias),
            CNA_Block(out_b3[0], out_b3[1], 5, 1, 2, bias=bias)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
        )
        self.branch4_audio = nn.Sequential(
            nn.MaxPool1d(3, 1, 1),
        )
        
    
    def forward(self, x):
        if len(x.shape) == 4:
            return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        else:
            return torch.cat([self.branch1_audio(x), self.branch2_audio(x), self.branch3_audio(x), self.branch4_audio(x)], 1)

class Inception_Q3(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

        self.layer1 = Inception_Block(3, 15, [16, 31], [16, 31])
        self.layer2 = Inception_Block(80, 32, [32, 64], [32, 64])
        self.layer3 = Inception_Block(240, 64, [64, 128], [64, 128])
        self.layer4 = Inception_Block(560, 64, [64, 128], [64, 128])
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(880*4*4, 10)
        self.fc2 = nn.Linear(10, 10)


        self.layer1_audio = Inception_Block(1, 6, [16, 15], [16, 20])
        self.layer2_audio = Inception_Block(42, 6, [32, 15], [32, 20])
        self.layer3_audio = Inception_Block(83, 6, [64, 15], [64, 20])
        self.layer4_audio = Inception_Block(124, 6, [64, 15], [64, 20])
        self.pool_audio = nn.MaxPool1d(2, 2)
        self.fc1_audio = nn.Linear(165000, 35)
        self.fc2_audio = nn.Linear(35, 35)

    def forward(self, x):
        """
        Write your code here
        """
        if (len(x.shape) == 4):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.pool(x)
            x = self.layer3(x)
            x = self.pool(x)
            x = self.layer4(x)
            x = self.pool(x)
            x = x.view(-1, 880*4*4)
            x = self.fc1(x)
            x = self.fc2(x)
        else:
            x = self.layer1_audio(x)
            x = self.layer2_audio(x)
            x = self.pool_audio(x)
            x = self.layer3_audio(x)
            x = self.pool_audio(x)
            x = self.layer4_audio(x)
            x = self.pool_audio(x)
            x = x.view(x.size(0), -1)
            x = self.fc1_audio(x)
            x = self.fc2_audio(x)
        return x
    

class CustomNetwork_Q4(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)
    
    # Write your code here
    for epoch in range(EPOCH):
        tot_loss = 0
        accuracy = 0
        for batch in tqdm(dataloader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            # print(outputs)
            outputs = network(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            accuracy += (outputs.argmax(1) == labels).sum().item()
            del images,labels,outputs
            torch.cuda.empty_cache()
            gc.collect()
        accuracy /= len(dataloader.dataset)
        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            tot_loss / len(dataloader),
            accuracy
        ))
        # save trained checkpoint with the name corresponding to model used and the dataset
        torch.save(network.state_dict(), network.__class__.__name__ +".pt")
        if accuracy > 0.75:
            break

    """
    Only use this print statement to print your epoch loss, accuracy
    print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """

def validator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    
    network = network.to(device)
    
    # Write your code here
    for epoch in range(EPOCH):
        tot_loss = 0
        accuracy = 0
        for batch in tqdm(dataloader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            accuracy += (outputs.argmax(1) == labels).sum().item()
            del images,labels,outputs
            torch.cuda.empty_cache()
            gc.collect()
        accuracy /= len(dataloader.dataset)
        print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            tot_loss / len(dataloader),
            accuracy
        ))

        if accuracy > 0.75:
            break
    """
    Only use this print statement to print your epoch loss, accuracy
    print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """


def evaluator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    # load trained checkpoint
    network.load_state_dict(torch.load(network.__class__.__name__ +".pt"))
    
    with torch.no_grad():  
        # for epoch in range(EPOCH):
        tot_loss = 0
        accuracy = 0
        for batch in tqdm(dataloader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            loss = criterion(outputs, labels)
            tot_loss += loss.item()
            accuracy += (outputs.argmax(1) == labels).sum().item()
            del images,labels,outputs
            torch.cuda.empty_cache()
            gc.collect()
        accuracy /= len(dataloader.dataset)
        print("[Loss: {}, Accuracy: {}]".format(
            tot_loss / len(dataloader),
            accuracy
        ))


    """
    Only use this print statement to print your loss, accuracy
    print("[Loss: {}, Accuracy: {}]".format(
        loss,
        accuracy
    ))
    """
    
    