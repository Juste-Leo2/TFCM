import torch
import torch.nn as nn
import timm

class ColorizerV14(nn.Module):
    def __init__(self):
        super(ColorizerV14, self).__init__()
        self.brain = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True, out_indices=[2])
        channels = self.brain.feature_info.channels()[0]

        self.w_enc1 = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1), nn.LeakyReLU(0.2)) 
        self.w_enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2)) 
        self.w_enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2)) 
        
        self.brain_conv = nn.Sequential(nn.Conv2d(channels, 128, 1, 1, 0), nn.Sigmoid())
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Conv2d(32, 8, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, feats, e1, e2, e3):
        # Logic split for clearer inference handling in the main class if needed
        # but here we keep the fusion logic
        gate = self.brain_conv(feats)
        fusion = e3 * (1 + gate)
        b = self.bottleneck(fusion)
        d3 = self.dec3(torch.cat([self.up3(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        return self.tanh(self.final(d2))