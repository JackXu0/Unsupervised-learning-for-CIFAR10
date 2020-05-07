import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from random import randint

from IPython.display import Image
from IPython.core.display import Image, display
from PIL import Image as I

from imgaug import augmenters as iaa
from torchsummary import summary

bs = 10
na = 3
printed = False

def augmentation(im):
    """
    image: 1, 3, n, n tensor
    
    output: num, 3, n, n tensor
    """
    transformations = [
        iaa.Crop(px=(0, 10), name='Crop'), # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5, name='Fliplr'), # horizontally flip 50% of the images
#         iaa.Flipud(0.5, name-'Flipud'), # vertically flip 50% of the images
#         iaa.GaussianBlur(sigma=(0, 3.0), name='GaussianBlur'), # blur images with a sigma of 0 to 3.0
#         iaa.CropAndPad(pad_mode='constant', name='CropAndPad'),  # crop or pad
#         iaa.Grayscale(alpha=randint(0,255), from_colorspace='RGB', name='GrayScale', deterministic=False),
#         iaa.AverageBlur(k=[1,2,4,7,10], name='AverageBlur', deterministic=False),
#         iaa.Sharpen(alpha=randint(0,255), lightness=1, name='Sharpen', deterministic=False),
#         iaa.Emboss(alpha=randint(0,255), strength=1, name='Emboss', deterministic=False),
#         iaa.Dropout(p=0, per_channel=False, name=None, deterministic=False, random_state=None),
#         iaa.ElasticTransformation(alpha=randint(0,255),sigma=0,name='ElasticTrans', deterministic=False),
        iaa.Noop(name='Noop')
    ]

    im = im.permute(1, 2, 0).numpy()

    return torch.Tensor([t(image=im) for t in transformations]).permute(0, 3, 1, 2)




# def randomPick(trans, n):
#     tran_list = []
#     global printed
#     for i in range (0, n):
#         tran_list.append(trans[randint(0, n-1)]) 
    
#     if not printed:
#         for i in range(0, n):
#             print(tran_list[i].name)
#         printed = True
    
#     return tran_list


# Pretrained dataset
dataset = datasets.CIFAR10('.', transform = transforms.Compose([transforms.ToTensor(), augmentation]), download = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

# Evaluation dataset
trainset = torchvision.datasets.CIFAR10(root='.',transform = transforms.Compose([transforms.ToTensor()]), train=True,download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)


class Net(nn.Module):
    def __init__(self, na=4):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 512)
        
        self.conv4 = nn.Conv2d(32, 16, 3,padding = 1)
        self.conv5 = nn.Conv2d(16, 6, 3,padding = 1)
        self.conv6 = nn.Conv2d(6, 3, 3,padding = 1)
        
        self.selfattn = nn.MultiheadAttention(128, 1)
        
        self.na = na

        
    def encoder(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return x
    
    
    # TODO: Use Fc for pooling    
    def pooling(self, vectors):
        na = self.na
        if not na:
            return vectors
        
        vectors = vectors.view(bs, na, 128)
        vectors = vectors[:, :na-1, :]
        vectors = vectors.permute(1, 0, 2)
        vectors, _ = self.selfattn(vectors, vectors, vectors)
        return vectors.mean(dim=0)
    
    def decoder(self, vector):
        x = self.fc2(vector)
        # 1, 512 --> 1, 32, 4, 4
        x = x.reshape(-1, 32, 4, 4)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv4(x)
        x = F.relu(x)
        
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv5(x)
        x = F.relu(x)

        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv6(x)

        x = torch.sigmoid(x)
        return x
       
        
    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.encoder(x)
        x = self.pooling(x)
        y = self.decoder(x)
        return x, y

class Evaluate(nn.Module):
    def __init__(self):
        super(Evaluate, self).__init__()
        self.fc1 = nn.Linear(128, 10)
    
    def forward(self,vector):
        x = self.fc1(vector)
        return x

def train(train_loader, epochs=2):
    model = Net(na = na)
    model.cuda()
    optimizer = optim.Adam(model.parameters())

    for param in model.parameters():
        param.requires_grad = True


    loss = nn.MSELoss()
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.cuda()
            
            optimizer.zero_grad()
            _, I_prime = model(data)

            I = data[:, -1, :]
            l1 = loss(I, I_prime)
            l1.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), l1.item()))

def evaluation(model, trainloader):

    net = Evaluate()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    epoch = 1
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data) in enumerate(trainloader):
        inputs, labels = data[0].cuda(), data[1].cuda()
        
        optimizer.zero_grad()

        I_pool, _ = model(inputs)
        outputs = net(I_pool)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10000:
            print('Step: [%5d] loss: %.3f' %(batch_idx + 1, loss.item()))
    
    print('Accuracy of the network: %d %%' % (100 * correct / total))


def main():
    train(dataloader)
    evaluation(trainloader)
    print('Finished Training')

if __name__ == '__main__':
    main()