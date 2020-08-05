import argparse

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import time

# import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD,Adam,lr_scheduler
from torch.utils.data import random_split
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='SoLeakyReLU')
parser.add_argument('--name', type=str, default="1", help='Name')
parser.add_argument('--model', type=str, default="densenet121", help='Model')
args = parser.parse_args()

# define transformations for train
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=.40),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# define transformations for test
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# define training dataloader
def get_training_dataloader(train_transform, batch_size=128, num_workers=0, shuffle=True):
    """ return training dataloader
    Args:
        train_transform: transfroms for train dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = train_transform
    cifar10_training = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

# define test dataloader
def get_testing_dataloader(test_transform, batch_size=128, num_workers=0, shuffle=True):
    """ return training dataloader
    Args:
        test_transform: transforms for test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = test_transform
    cifar10_test = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

# implement mish activation function
def f_mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    '''
    return input * torch.tanh(F.softplus(input))

# implement class wrapper for mish activation function
class mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_mish(input)

# implement swish activation function
def f_swish(input):
    '''
    Applies the swish function element-wise:
    swish(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input)

# implement class wrapper for swish activation function
class swish(nn.Module):
    '''
    Applies the swish function element-wise:
    swish(x) = x * sigmoid(x)

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = swish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_swish(input)


def mish_grad(x):
    old_dim = x.to(device).shape
    x = torch.flatten(x).to(device)
    dim = x.to(device).shape
    assert len(dim) == 1
    x_rows = dim[0]
    t_0 = torch.exp(x).to(device)
    t_1 = (torch.ones(x_rows).to(device) + t_0).to(device)
    t_2 = torch.tanh(torch.log(t_1)).to(device)
    mish_val = (x * t_2)
    grad = t_2 + ((x * (torch.ones(x_rows).to(device) - (t_2 ** 2))).to(device) * t_0) / t_1
    return grad.reshape(old_dim).to(device)


class R_Mish_ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        mish_grads = mish_grad(input).to(device)
        grad_input[(input < 0) * (0 > grad_input)] = mish_grads[(input < 0) * (0 > grad_input)]
        grad_input[(input < 0) * (0 <= grad_input)] = 0
        return grad_input


class R_LeakyReLU_ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(input < 0) * (0 <= grad_input)] = 0
        grad_input[input < 0] /= 100
        return grad_input


# custom activation
class UniGrad(nn.Module):
    def __init__(self, autograd_func):
        super(UniGrad, self).__init__()
        self.autograd_func = autograd_func

    def forward(self, x):
        return self.autograd_func.apply(x)

#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, activation = 'relu'):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        if activation == 'relu':
            f_activation = nn.ReLU(inplace=True)

        if activation == 'LeakyReLU':
            f_activation = nn.LeakyReLU(inplace=True)

        if activation == 'swish':
            f_activation = swish()

        if activation == 'mish':
            f_activation = mish()

        if activation == "R_LeakyReLU_ReLU":
            f_activation = UniGrad(autograd_func=R_LeakyReLU_ReLU)

        if activation == "R_Mish_ReLU":
            f_activation = UniGrad(autograd_func=R_Mish_ReLU)

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            f_activation,
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100, activation = 'relu'):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        if activation == 'relu':
            f_activation = nn.ReLU(inplace=True)

        if activation == 'LeakyReLU':
            f_activation = nn.LeakyReLU(inplace=True)

        if activation == 'swish':
            f_activation = swish()

        if activation == 'mish':
            f_activation = mish()

        if activation == "R_LeakyReLU_ReLU":
            f_activation = UniGrad(autograd_func=R_LeakyReLU_ReLU)

        if activation == "R_Mish_ReLU":
            f_activation = UniGrad(autograd_func=R_Mish_ReLU)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('activation', f_activation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121(activation = 'relu'):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, activation = activation)

def densenet169(activation = 'relu'):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, activation = activation)

def densenet201(activation = 'relu'):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, activation = activation)

def densenet161(activation = 'relu'):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, activation = activation)

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, activation = 'relu', **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)

        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)

        if activation == 'LeakyReLU':
            self.relu = nn.LeakyReLU(inplace=True)

        if activation == 'swish':
            self.relu = swish()

        if activation == 'mish':
            self.relu = mish()

        if activation == "R_LeakyReLU_ReLU":
            self.relu = UniGrad(autograd_func=R_LeakyReLU_ReLU)

        if activation == "R_Mish_ReLU":
            self.relu = UniGrad(autograd_func=R_Mish_ReLU)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

#same naive inception module
class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features, activation = 'relu'):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 64, kernel_size=1, activation = activation)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_channels, 48, kernel_size=1, activation = activation),
            BasicConv2d(48, 64, kernel_size=5, padding=2, activation = activation)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1, activation = activation),
            BasicConv2d(64, 96, kernel_size=3, padding=1, activation = activation),
            BasicConv2d(96, 96, kernel_size=3, padding=1, activation = activation)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, pool_features, kernel_size=3, padding=1, activation = activation)
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(x)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(x)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)

#downsample
#Factorization into smaller convolutions
class InceptionB(nn.Module):

    def __init__(self, input_channels, activation = 'relu'):
        super().__init__()

        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=3, stride=2, activation = activation)

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1, activation = activation),
            BasicConv2d(64, 96, kernel_size=3, padding=1, activation = activation),
            BasicConv2d(96, 96, kernel_size=3, stride=2, activation = activation)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x - > 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 3x3 -> 3x3(downsample)
        branch3x3stack = self.branch3x3stack(x)

        #x -> avgpool(downsample)
        branchpool = self.branchpool(x)

        #"""We can use two parallel stride 2 blocks: P and C. P is a pooling
        #layer (either average or maximum pooling) the activation, both of
        #them are stride 2 the filter banks of which are concatenated as in
        #figure 10."""
        outputs = [branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

#Factorizing Convolutions with Large Filter Size
class InceptionC(nn.Module):
    def __init__(self, input_channels, channels_7x7, activation = 'relu'):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1, activation = activation)

        c7 = channels_7x7

        #In theory, we could go even further and argue that one can replace any n × n
        #convolution by a 1 × n convolution followed by a n × 1 convolution and the
        #computational cost saving increases dramatically as n grows (see figure 6).
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1, activation = activation),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0), activation = activation),
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3), activation = activation)
        )

        self.branch7x7stack = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1, activation = activation),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0), activation = activation),
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3), activation = activation),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0), activation = activation),
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3), activation = activation)
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 192, kernel_size=1, activation = activation),
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1layer 1*7 and 7*1 (same)
        branch7x7 = self.branch7x7(x)

        #x-> 2layer 1*7 and 7*1(same)
        branch7x7stack = self.branch7x7stack(x)

        #x-> avgpool (same)
        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(self, input_channels, activation = 'relu'):
        super().__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1, activation = activation),
            BasicConv2d(192, 320, kernel_size=3, stride=2, activation = activation)
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1, activation = activation),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3), activation = activation),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0), activation = activation),
            BasicConv2d(192, 192, kernel_size=3, stride=2, activation = activation)
        )

        self.branchpool = nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x -> 1x1 -> 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 1x1 -> 1x7 -> 7x1 -> 3x3 (downsample)
        branch7x7 = self.branch7x7(x)

        #x -> avgpool (downsample)
        branchpool = self.branchpool(x)

        outputs = [branch3x3, branch7x7, branchpool]

        return torch.cat(outputs, 1)


#same
class InceptionE(nn.Module):
    def __init__(self, input_channels, activation = 'relu'):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 320, kernel_size=1, activation = activation)

        self.branch3x3_1 = BasicConv2d(input_channels, 384, kernel_size=1, activation = activation)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1), activation = activation)
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0), activation = activation)

        self.branch3x3stack_1 = BasicConv2d(input_channels, 448, kernel_size=1, activation = activation)
        self.branch3x3stack_2 = BasicConv2d(448, 384, kernel_size=3, padding=1, activation = activation)
        self.branch3x3stack_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1), activation = activation)
        self.branch3x3stack_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0), activation = activation)

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 192, kernel_size=1, activation = activation)
        )

    def forward(self, x):

        #x -> 1x1 (same)
        branch1x1 = self.branch1x1(x)

        # x -> 1x1 -> 3x1
        # x -> 1x1 -> 1x3
        # concatenate(3x1, 1x3)
        #"""7. Inception modules with expanded the filter bank outputs.
        #This architecture is used on the coarsest (8 × 8) grids to promote
        #high dimensional representations, as suggested by principle
        #2 of Section 2."""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        # x -> 1x1 -> 3x3 -> 1x3
        # x -> 1x1 -> 3x3 -> 3x1
        #concatenate(1x3, 3x1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)

        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionV3(nn.Module):

    def __init__(self, num_classes=10, activation = 'relu'):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, padding=1, activation = activation)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1, activation = activation)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1, activation = activation)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1, activation = activation)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, activation = activation)

        #naive inception module
        self.Mixed_5b = InceptionA(192, pool_features=32, activation = activation)
        self.Mixed_5c = InceptionA(256, pool_features=64, activation = activation)
        self.Mixed_5d = InceptionA(288, pool_features=64, activation = activation)

        #downsample
        self.Mixed_6a = InceptionB(288, activation = activation)

        self.Mixed_6b = InceptionC(768, channels_7x7=128, activation = activation)
        self.Mixed_6c = InceptionC(768, channels_7x7=160, activation = activation)
        self.Mixed_6d = InceptionC(768, channels_7x7=160, activation = activation)
        self.Mixed_6e = InceptionC(768, channels_7x7=192, activation = activation)

        #downsample
        self.Mixed_7a = InceptionD(768, activation = activation)

        self.Mixed_7b = InceptionE(1280, activation = activation)
        self.Mixed_7c = InceptionE(2048, activation = activation)

        #6*6 feature size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):

        #32 -> 30
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)

        #30 -> 30
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        #30 -> 14
        #Efficient Grid Size Reduction to avoid representation
        #bottleneck
        x = self.Mixed_6a(x)

        #14 -> 14
        #"""In practice, we have found that employing this factorization does not
        #work well on early layers, but it gives very good results on medium
        #grid-sizes (On m × m feature maps, where m ranges between 12 and 20).
        #On that level, very good results can be achieved by using 1 × 7 convolutions
        #followed by 7 × 1 convolutions."""
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        #14 -> 6
        #Efficient Grid Size Reduction
        x = self.Mixed_7a(x)

        #6 -> 6
        #We are using this solution only on the coarsest grid,
        #since that is the place where producing high dimensional
        #sparse representation is the most critical as the ratio of
        #local processing (by 1 × 1 convolutions) is increased compared
        #to the spatial aggregation."""
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        #6 -> 1
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def inceptionv3(activation = 'relu'):
    return InceptionV3(activation = activation)

trainloader = get_training_dataloader(train_transform)
testloader = get_testing_dataloader(test_transform)

epochs = 100
batch_size = 128
learning_rate = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

for name in ["1", "2", "3"]:
    args.name = name
    for model in ["inceptionv3", "densenet201", "densenet169", "densenet161"]:
        args.model = model
        for activation_choice in ["R_LeakyReLU_ReLU", "R_Mish_ReLU", "LeakyReLU", "mish", "swish", "relu"]:
        # for activation_choice in ["R_Mish_ReLU", "LeakyReLU", "mish", "swish", "relu"]:
            if args.model == "densenet121":
                model = densenet121(activation = activation_choice).to(device)
            elif args.model == "densenet169":
                model = densenet169(activation = activation_choice).to(device)
            elif args.model == "densenet201":
                model = densenet201(activation = activation_choice).to(device)
            elif args.model == "densenet161":
                model = densenet161(activation = activation_choice).to(device)
            elif args.model == "inceptionv3":
                model = inceptionv3(activation = activation_choice)

            # set loss function
            criterion = nn.CrossEntropyLoss()

            # set optimizer, only train the classifier parameters, feature parameters are frozen
            optimizer = Adam(model.parameters(), lr=learning_rate)

            train_stats = pd.DataFrame(columns = ['Epoch', 'Time per epoch', 'Avg time per step', 'Train loss', 'Train accuracy', 'Train top-3 accuracy','Test loss', 'Test accuracy', 'Test top-3 accuracy'])

            #train the model
            model.to(device)

            steps = 0
            running_loss = 0
            for epoch in range(epochs):

                since = time.time()

                train_accuracy = 0
                top3_train_accuracy = 0
                for inputs, labels in trainloader:
                    steps += 1
                    # Move input and label tensors to the default device
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    logps = model.forward(inputs).to(device)
                    loss = criterion(logps, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    # calculate train top-1 accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    # Calculate train top-3 accuracy
                    np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
                    target_numpy = labels.cpu().numpy()
                    top3_train_accuracy += np.mean([1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

                time_elapsed = time.time() - since

                test_loss = 0
                test_accuracy = 0
                top3_test_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate test top-1 accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        # Calculate test top-3 accuracy
                        np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
                        target_numpy = labels.cpu().numpy()
                        top3_test_accuracy += np.mean([1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Time per epoch: {time_elapsed:.4f}.. "
                      f"Average time per step: {time_elapsed/len(trainloader):.4f}.. "
                      f"Train loss: {running_loss/len(trainloader):.4f}.. "
                      f"Train accuracy: {train_accuracy/len(trainloader):.4f}.. "
                      f"Top-3 train accuracy: {top3_train_accuracy/len(trainloader):.4f}.. "
                      f"Test loss: {test_loss/len(testloader):.4f}.. "
                      f"Test accuracy: {test_accuracy/len(testloader):.4f}.. "
                      f"Top-3 test accuracy: {top3_test_accuracy/len(testloader):.4f}")

                train_stats = train_stats.append({'Epoch': epoch, 'Time per epoch':time_elapsed, 'Avg time per step': time_elapsed/len(trainloader), 'Train loss' : running_loss/len(trainloader), 'Train accuracy': train_accuracy/len(trainloader), 'Train top-3 accuracy':top3_train_accuracy/len(trainloader),'Test loss' : test_loss/len(testloader), 'Test accuracy': test_accuracy/len(testloader), 'Test top-3 accuracy':top3_test_accuracy/len(testloader)}, ignore_index=True)

                running_loss = 0
                model.train()

            train_stats.to_csv('train_log_{}_{}_{}.csv'.format(model, activation_choice, args.name))