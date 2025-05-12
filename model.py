import torch
import torch.nn as nn
import torch.nn.utils as utils_nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        proj_q = self.query_conv(x).view(B, -1, W*H).permute(0,2,1)
        proj_k = self.key_conv(x).view(B, -1, W*H)
        energy = torch.bmm(proj_q, proj_k)
        attn  = self.softmax(energy)
        proj_v = self.value_conv(x).view(B, -1, W*H)
        out    = torch.bmm(proj_v, attn.permute(0,2,1))
        out    = out.view(B, C, W, H)
        return self.gamma * out + x

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*16, 4,1,0, bias=False), nn.BatchNorm2d(ngf*16), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*16, ngf*8, 4,2,1, bias=False), nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4,2,1, bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            SelfAttention(ngf*4),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4,2,1, bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            SelfAttention(ngf*2),
            nn.ConvTranspose2d(ngf*2, ngf, 4,2,1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4,2,1, bias=False), nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)
    
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            utils_nn.spectral_norm(nn.Conv2d(nc, ndf, 4,2,1, bias=False)), nn.LeakyReLU(0.2, True),
            utils_nn.spectral_norm(nn.Conv2d(ndf, ndf*2, 4,2,1, bias=False)), nn.LeakyReLU(0.2, True),
            utils_nn.spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4,2,1, bias=False)), nn.LeakyReLU(0.2, True),
            SelfAttention(ndf*4),
            utils_nn.spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4,2,1, bias=False)), nn.LeakyReLU(0.2, True),
            SelfAttention(ndf*8),
            utils_nn.spectral_norm(nn.Conv2d(ndf*8, ndf*16,4,2,1, bias=False)), nn.LeakyReLU(0.2, True),
            utils_nn.spectral_norm(nn.Conv2d(ndf*16,1,4,1,0, bias=False))
        )
    def forward(self, x):
        return self.main(x).view(-1)