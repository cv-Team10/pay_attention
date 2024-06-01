# CODE FROM
# https://github.com/SaoYan/LearnToPayAttention 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from blocks import ConvBlock, LinearAttentionBlock, ProjectorBlock

class AttentionModel(nn.Module):
    def __init__(self, im_size, num_keypoints, normalize_attn=True):
        super(AttentionModel, self).__init__()
        # conv blocks
        self.conv_block1 = ConvBlock(in_features=3, out_features=64, num_conv=2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size/32), padding=0, bias=True)

        # Projectors * Compatibility functions
        self.projector = ProjectorBlock(256, 512)
        self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)

        # Final classification layer - in our case regression
        self.regression = nn.Linear(in_features=512*3, out_features=num_keypoints*2, bias=True)

        # Add classification head
        self.classify = nn.Sequential(
            nn.Linear(in_features=512*3, out_features=num_keypoints),
            nn.Sigmoid()
        )

        # Initialize weights - Xavier Uniform
        self.weights_init_xavierUniform()
    
    def weights_init_xavierUniform(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a=0, b=1)
                nn.init.constant_(m.bias, val=0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0.)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        l1 = self.conv_block3(x)
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)
        l2 = self.conv_block4(x)
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)
        l3 = self.conv_block5(x)
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0)
        x = self.conv_block6(x)
        g = self.dense(x) 

        # Pay attention. c is compatibility score corresponding to given layer l
        c1, g1 = self.attn1(self.projector(l1), g)
        c2, g2 = self.attn2(l2, g)
        c3, g3 = self.attn3(l3, g)
        # By getting g, restrict regressor and classifor to only use subset of spatial features
        g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
        # classification layer (regression)
        x1 = self.regression(g) # batch_sizexnum_classes
        # print('-_-_-_-_-_')
        # print(x1)
        # classification
        x2 = self.classify(g)
        # print('-_-_-_-_-_')
        # print(x2)

        print("---LOLOLOL---")
        print(x1.shape)
        print(x2.shape)

        x1 = x1.view(-1, 17, 2)
        x2 = x2.unsqueeze(-1)
        print("---DADADA---")
        print(x1.shape)
        print(x2.shape)

        print("====HIHIHI====")

        output = torch.cat((x1, x2), dim=-1)
        print(output.shape)



        return [output, c1, c2, c3]

    




