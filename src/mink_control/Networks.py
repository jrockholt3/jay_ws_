import os 
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
# from Object_v2 import rand_object, Cylinder
import torch
from time import time 
from mink_control.env_config import *

conv_out1 = 32
conv_out2 = 64
conv_out3 = 64
conv_out4 = 128
linear_out = 512
dropout = 0.2
k1 = (3,4,4,4) # kernel shape
s1 = (2,2,2,2) # stride 
k2 = (3,4,4,4)
s2 = (2,2,2,2)
k3 = (2,2,2,2)
s3 = (2,2,2,2)
k4 = (1,17,17,13)
s4 = (1,2,2,2)
kg1= ME.KernelGenerator(
    kernel_size = k1,
    stride = s1,
    region_type=ME.RegionType.HYPER_CUBE,
    dimension=4
)
kg2= ME.KernelGenerator(
    kernel_size = k2,
    stride = s2,
    region_type=ME.RegionType.HYPER_CUBE,
    dimension=4
)
kg3= ME.KernelGenerator(
    kernel_size = k3,
    stride = s3,
    region_type=ME.RegionType.HYPER_CUBE,
    dimension=4
)
kg4= ME.KernelGenerator(
    kernel_size = k4,
    stride = s4,
    region_type=ME.RegionType.HYPER_CUBE,
    dimension=4
)
class Actor(ME.MinkowskiNetwork,nn.Module):

    def __init__(self, in_feat=1, jnt_dim=5, D=4, name='actor', chckpt_dir = 'tmp', device='cuda', top_only=False):
        self.D = D
        super(Actor, self).__init__(self.D)
        self.name = name 
        self.file_path = os.path.join(chckpt_dir,name+'_ddpg')
        self.chckpt_dir = chckpt_dir
        
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_feat,
                out_channels=conv_out1,
                kernel_size=k1,
                stride=s1,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out1).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out1,
                out_channels=conv_out2,
                kernel_size=k2,
                stride=s2,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out2).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out2,
                out_channels=conv_out3,
                kernel_size=k3,
                stride=s3,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out3).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out3,
                out_channels=conv_out4,
                kernel_size=k4,
                stride=s4,
                bias=True,
                dimension=D).double(),
            # ME.MinkowskiBatchNorm(conv_out4),
            # ME.MinkowskiSELU()
            ME.MinkowskiGlobalMaxPooling()
        )
        self.norm = nn.Sequential(
            nn.LayerNorm(conv_out4).double(),
            nn.SELU().double()
        )
        self.dropout1 = nn.Dropout(dropout).double()
        self.linear = nn.Sequential(
            nn.Linear(conv_out4+2*jnt_dim+3,linear_out).double(),
            nn.LayerNorm(linear_out).double(),
            nn.SELU().double()
        )
        self.dropout2 = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(linear_out,jnt_dim,bias=True).double(),
            nn.Tanh().double()
        )

        # self.action_var = torch.full((n_actions, ), action_std*action_std).to(device)
        self.device = device
        self.to(self.device)
        self.top_only = top_only

    def to_dense_tnsr(self, x:ME.SparseTensor):
        y = torch.zeros_like(x.features)
        for c in x.coordinates:
            y[int(c[0])] = x.features[int(c[0])]
        return y


    def forward(self,x:ME.SparseTensor,jnt_pos,jnt_goal,weights):
        x = self.conv1(x)
        x = self.to_dense_tnsr(x)
        x = self.norm(x)
        x = self.dropout1(x)
        x = torch.cat((jnt_pos,jnt_goal,weights,x),dim=1)
        x = self.linear(x)
        x = self.dropout2(x)
        x = self.out(x) * tau_max
        return x

    def save_checkpoint(self, save_as=None):
        if isinstance(save_as,str):
            file_path = os.path.join(self.chckpt_dir,save_as)
        else:
            file_path = self.file_path
        print('...saving ' + self.name + '...')
        torch.save(self.state_dict(), file_path)
        return True 

    def load_checkpoint(self, load_as=None):
        if isinstance(load_as, str):
            file_path = os.path.join(self.chckpt_dir, load_as)
        else:
            file_path = self.file_path
        print('...loading ' + self.name + '...')
        self.load_state_dict(torch.load(file_path))

class SupervisedActor(ME.MinkowskiNetwork,nn.Module):

    def __init__(self, in_feat=1, jnt_dim=5, D=4, name='temp_supervised_actor', chckpt_dir = 'tmp', device='cuda', top_only=False):
        self.D = D
        super(SupervisedActor, self).__init__(self.D)
        self.name = name 
        self.file_path = os.path.join(chckpt_dir,name)
        self.chckpt_dir = chckpt_dir
        
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_feat,
                out_channels=conv_out1,
                kernel_size=k1,
                stride=s1,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out1).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out1,
                out_channels=conv_out2,
                kernel_size=k2,
                stride=s2,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out2).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out2,
                out_channels=conv_out3,
                kernel_size=k3,
                stride=s3,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out3).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out3,
                out_channels=conv_out4,
                kernel_size=k4,
                stride=s4,
                bias=True,
                dimension=D).double(),
            # ME.MinkowskiBatchNorm(conv_out4),
            # ME.MinkowskiSELU()
            ME.MinkowskiGlobalMaxPooling()
        )
        self.norm = nn.Sequential(
            nn.LayerNorm(conv_out4).double(),
            nn.SELU().double()
        )
        self.dropout1 = nn.Dropout(dropout).double()
        self.linear = nn.Sequential(
            nn.Linear(conv_out4+2*jnt_dim,linear_out).double(),
            nn.LayerNorm(linear_out).double(),
            nn.SELU().double()
        )
        self.dropout2 = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(linear_out,jnt_dim,bias=True).double(),
            nn.Tanh().double()
        )

        # self.action_var = torch.full((n_actions, ), action_std*action_std).to(device)
        self.device = device
        self.to(self.device)
        self.top_only = top_only

    def to_dense_tnsr(self, x:ME.SparseTensor):
        y = torch.zeros_like(x.features)
        for c in x.coordinates:
            y[int(c[0])] = x.features[int(c[0])]
        return y


    def forward(self,x:ME.SparseTensor,jnt_pos,jnt_goal,weights):
        x = self.conv1(x)
        x = self.to_dense_tnsr(x)
        x = self.norm(x)
        x = self.dropout1(x)
        x = torch.cat((jnt_pos,jnt_goal,x),dim=1)
        x = self.linear(x)
        x = self.dropout2(x)
        x = self.out(x) * tau_max
        return x

    def save_checkpoint(self, save_as=None):
        if isinstance(save_as,str):
            file_path = os.path.join(self.chckpt_dir,save_as)
        else:
            file_path = self.file_path
        print('...saving ' + self.name + '...')
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, load_as=None):
        if isinstance(load_as, str):
            file_path = os.path.join(self.chckpt_dir, load_as)
        else:
            file_path = self.file_path
        print('...loading ' + self.name + '...')
        if not torch.cuda.is_available():
            map_location='cpu'
            self.load_state_dict(torch.load(file_path,map_location=map_location))
        else:
            self.load_state_dict(torch.load(file_path))


class Critic(ME.MinkowskiNetwork,nn.Module):

    def __init__(self, in_feat, jnt_dim, D, name, chckpt_dir = 'tmp', device='cuda',top_only=False):
        super(Critic, self).__init__(D)
        self.name = name 
        self.file_path = os.path.join(chckpt_dir,name+'_ddpg')
        self.chckpt_dir = chckpt_dir

        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_feat,
                out_channels=conv_out1,
                kernel_size=k1,
                stride=s1,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out1).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out1,
                out_channels=conv_out2,
                kernel_size=k2,
                stride=s2,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out2).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out2,
                out_channels=conv_out3,
                kernel_size=k3,
                stride=s3,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out3).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out3,
                out_channels=conv_out4,
                kernel_size=k4,
                stride=s4,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiGlobalMaxPooling()
        )
        self.norm = nn.Sequential(
            nn.LayerNorm(conv_out4).double(),
            nn.SELU().double()
        )
        self.dropout1 = nn.Dropout(dropout).double()
        self.linear = nn.Sequential(
            nn.Linear(conv_out4+3*jnt_dim+3,linear_out).double(),
            nn.LayerNorm(linear_out).double(),
            nn.SELU().double()
        )
        self.dropout2 = nn.Dropout(dropout).double()
        self.out = nn.Sequential(
            nn.Linear(linear_out,1,bias=True).double()
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.top_only = top_only

    def to_dense_tnsr(self, x:ME.SparseTensor):
        y = torch.zeros_like(x.features)
        for c in x.coordinates:
            y[int(c[0])] = x.features[int(c[0])]
        return y

    def forward(self,x:ME.SparseTensor,jnt_pos,jnt_goal,weights,action):
        x = self.conv1(x)
        x = self.to_dense_tnsr(x)
        x = self.norm(x)
        x = self.dropout1(x)
        x = torch.cat((jnt_pos,jnt_goal,weights,action,x),dim=1)
        x = self.linear(x)
        x = self.dropout2(x)
        x = self.out(x)
        return x 

    def save_checkpoint(self, save_as=None):
        if isinstance(save_as,str):
            file_path = os.path.join(self.chckpt_dir,save_as)
        else:
            file_path = self.file_path
        print('...saving ' + self.name + '...')
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self):
        print('...loading ' + self.name + '...')
        self.load_state_dict(torch.load(self.file_path))


