from torch import nn
import torch.nn.functional as F
import torch
import math

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class ConvEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            #CBAM(16), 
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True) 
            #CBAM(32)
        )
        self.encoder_cnn2 = nn.Sequential(
            nn.MaxPool2d(2,2), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #CBAM(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
            #CBAM(128)
        )
        self.encoder_cnn3 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #CBAM(128),
            nn.MaxPool2d(2,2)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
    def forward(self, x):
        x1 = self.encoder_cnn1(x)
        x2 = self.encoder_cnn2(x1)
        x3 = self.encoder_cnn3(x2)
        x = self.flatten(x3)
        z = [x1,x2]
        return x, z

class LinearEncoder(nn.Module):

    def __init__(self,features):
        super().__init__()

        ### Linear section
        self.encoder_lin = nn.Sequential(
            #256 image
            nn.Linear(256*16*16, features), 
            nn.BatchNorm1d(num_features=features),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder_lin(x)
        return x


class ConvDecoder(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.unflatten = nn.Unflatten(dim=1, 
                    unflattened_size=(24, 16, 16))   #24 , 16 ,16

        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(24, 64, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

class LinearDecoder(nn.Module):
    
    def __init__(self, features):
        super().__init__()
        self.decoder_lin = nn.Sequential(

            nn.Linear(features, 24 * 16 * 16), 
            nn.ReLU(True),
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        return x


class AutoEncoderDecoder(nn.Module):

    def __init__(self, features = 1024):
        super().__init__()

        self.convenoder = ConvEncoder()
        self.linencoder = LinearEncoder(features)

        self.convdecoder = ConvDecoder()
        self.lindecoder = LinearDecoder(features)

    
    def forward(self,x):
        x,z = self.convenoder(x)

        y = self.linencoder(x)

        x = self.lindecoder(y)

        x = self.convdecoder(x)

        return x , y


'''inp = torch.ones(2,1,512,512)
model = AutoEncoderDecoder(1024)
out,y = model(inp)
print (out.shape)'''
