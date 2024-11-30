import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Tuple, Union, List
import pywt
import matplotlib.pyplot as plt
from einops import rearrange
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initializing network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class DWT_1lvl(nn.Module):

    def __init__(self, fb_lo = None, flen = 8, nfil = 1,
                 pad_mode="replicate"):
        '''
        DWT_1lvl: 1-level multi-filter DWT using 2D convolution
        Args:
            fb_lo: low-pass filter bank
            flen: filter length
            nfil: number of filters
        '''

        super(DWT_1lvl, self).__init__()

        self.fb_lo = nn.Parameter(self.get_fb_lo(fb_lo, flen, nfil))
        self.nfil = self.fb_lo.shape[0]
        self.flen = self.fb_lo.shape[1]
        self.pad_mode = pad_mode

    def get_fb_lo(self, fb_lo = None, flen = 8, nfil = 1):
        if fb_lo is None:
            fb_lo = torch.rand((nfil, flen)) - 0.5
        else:
            if not isinstance(fb_lo, torch.Tensor):
                if isinstance(fb_lo, np.ndarray):
                    fb_lo = torch.tensor(fb_lo)
                else:
                    raise ValueError('fb_lo should be a tensor or numpy array')
            if fb_lo.dim() > 2:
                raise ValueError('fb_lo should be a 1D or 2D tensor')
            if fb_lo.dim() == 1:
                fb_lo = fb_lo.unsqueeze(0)
        return fb_lo
    
    def get_fb_conv(self, img_channels):
        # Normalize the low-pass filters to have unit norm
        fb_lo = F.normalize(self.fb_lo, p=2, dim=-1)
        # Generate high-pass filters by flipping and changing signs
        fb_hi = fb_lo.flip(-1)
        fb_hi[:, ::2] *= -1
        fb_hi -= fb_hi.mean(dim=-1, keepdim=True)
        # Create 2D filter banks using outer products
        fb_ll = torch.einsum('nf,ng->nfg', fb_lo, fb_lo)
        fb_lh = torch.einsum('nf,ng->nfg', fb_hi, fb_lo)
        fb_hl = torch.einsum('nf,ng->nfg', fb_lo, fb_hi)
        fb_hh = torch.einsum('nf,ng->nfg', fb_hi, fb_hi)
        # Prepare the 2D filter banks for conv2d
        fb_conv = torch.stack([fb_ll, fb_lh, fb_hl, fb_hh], 1)
        fb_conv = fb_conv.view(-1, fb_conv.shape[-2], fb_conv.shape[-1])
        fb_conv = fb_conv.repeat(img_channels, 1, 1)
        fb_conv = fb_conv.unsqueeze(dim=1)
        return fb_conv

    def forward(self, x):

        b, c, h, w = x.shape

        padb = (2 * self.flen - 3) // 2
        padt = (2 * self.flen - 3) // 2
        if h % 2 != 0:
            padb += 1
        padr = (2 * self.flen - 3) // 2
        padl = (2 * self.flen - 3) // 2
        if w % 2 != 0:
            padl += 1

        x_pad = F.pad(x, [padl, padr, padt, padb], mode=self.pad_mode)
        x_dwt = F.conv2d(x_pad, self.get_fb_conv(c).to(x_pad.device), stride=2, groups=c)

        x_dwt = rearrange(x_dwt, 'b (c f) h w -> b c f h w', f=4)
        x_ll, x_lh, x_hl, x_hh = x_dwt.split(1, 2)
        x_dwt = [x_ll.squeeze(2), (x_lh.squeeze(2), x_hl.squeeze(2), x_hh.squeeze(2))]

        return x_dwt

def get_img_dwt(img_dwt, idx=0, nfil=1):
    img_idx_dwt = [None, None]
    img_idx_dwt[0] = img_dwt[0][:,idx::nfil].detach().numpy()
    img_idx_dwt[1] = [img_dwt[1][0][:,idx::nfil].detach().numpy(), img_dwt[1][1][:,idx::nfil].detach().numpy(), img_dwt[1][2][:,idx::nfil].detach().numpy()]
    return img_idx_dwt

class WaveNetX(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, fb_lo=None, flen=8, nfil=1):
        super(WaveNetX, self).__init__()

        # wavelet block
        self.dwt = DWT_1lvl(fb_lo=fb_lo, flen=flen, nfil=nfil)
    
        # main network
        self.M_Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.M_Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.M_Conv2 = conv_block(ch_in=64, ch_out=128)
        self.M_Conv3 = conv_block(ch_in=128, ch_out=256)
        self.M_Conv4 = conv_block(ch_in=256, ch_out=512)
        self.M_Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.M_Up5 = up_conv(ch_in=1024, ch_out=512)
        self.M_Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.M_Up4 = up_conv(ch_in=512, ch_out=256)
        self.M_Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.M_Up3 = up_conv(ch_in=256, ch_out=128)
        self.M_Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.M_Up2 = up_conv(ch_in=128, ch_out=64)
        self.M_Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.M_Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

        # L network
        self.L_Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.L_Conv1 = conv_block(ch_in=nfil*in_channels, ch_out=64)
        self.L_Conv2 = conv_block(ch_in=64, ch_out=128)
        self.L_Conv3 = conv_block(ch_in=128, ch_out=256)
        self.L_Conv4 = conv_block(ch_in=256, ch_out=512)
        self.L_Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.L_Up5 = up_conv(ch_in=1024, ch_out=512)
        self.L_Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.L_Up4 = up_conv(ch_in=512, ch_out=256)
        self.L_Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.L_Up3 = up_conv(ch_in=256, ch_out=128)
        self.L_Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.L_Up2 = up_conv(ch_in=128, ch_out=64)
        self.L_Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.L_Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

        # H network
        self.H_Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.H_Conv1 = conv_block(ch_in=nfil*3*in_channels, ch_out=64)
        self.H_Conv2 = conv_block(ch_in=64, ch_out=128)
        self.H_Conv3 = conv_block(ch_in=128, ch_out=256)
        self.H_Conv4 = conv_block(ch_in=256, ch_out=512)
        self.H_Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.H_Up5 = up_conv(ch_in=1024, ch_out=512)
        self.H_Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.H_Up4 = up_conv(ch_in=512, ch_out=256)
        self.H_Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.H_Up3 = up_conv(ch_in=256, ch_out=128)
        self.H_Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.H_Up2 = up_conv(ch_in=128, ch_out=64)
        self.H_Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.H_Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

        # fusion
        self.M_H_Conv1 = conv_block(ch_in=128, ch_out=64)
        self.M_H_Conv2 = conv_block(ch_in=256, ch_out=128)
        self.M_L_Conv3 = conv_block(ch_in=512, ch_out=256)
        self.M_L_Conv4 = conv_block(ch_in=1024, ch_out=512)

    def forward(self, x_main, *args, **kwargs): #ignore excess
        # main encoder

        x_L, (x_LH, x_HL, x_HH) = self.dwt(x_main)
        # x_H = x_HL + x_LH + x_HH

        x_H = torch.cat((x_LH, x_HL, x_HH), dim=1)

        # Resize x_L and x_H to match the spatial dimensions of x_main
        x_L = F.interpolate(x_L, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)
        x_H = F.interpolate(x_H, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)

        M_x1 = self.M_Conv1(x_main)
        M_x2 = self.M_Maxpool(M_x1)
        M_x2 = self.M_Conv2(M_x2)
        M_x3 = self.M_Maxpool(M_x2)
        M_x3 = self.M_Conv3(M_x3)
        M_x4 = self.M_Maxpool(M_x3)
        M_x4 = self.M_Conv4(M_x4)
        M_x5 = self.M_Maxpool(M_x4)
        M_x5 = self.M_Conv5(M_x5)

        # L encoder
        L_x1 = self.L_Conv1(x_L)
        L_x2 = self.L_Maxpool(L_x1)
        L_x2 = self.L_Conv2(L_x2)
        L_x3 = self.L_Maxpool(L_x2)
        L_x3 = self.L_Conv3(L_x3)
        L_x4 = self.L_Maxpool(L_x3)
        L_x4 = self.L_Conv4(L_x4)
        L_x5 = self.L_Maxpool(L_x4)
        L_x5 = self.L_Conv5(L_x5)

        # H encoder
        H_x1 = self.H_Conv1(x_H)
        H_x2 = self.H_Maxpool(H_x1)
        H_x2 = self.H_Conv2(H_x2)
        H_x3 = self.H_Maxpool(H_x2)
        H_x3 = self.H_Conv3(H_x3)
        H_x4 = self.H_Maxpool(H_x3)
        H_x4 = self.H_Conv4(H_x4)
        H_x5 = self.H_Maxpool(H_x4)
        H_x5 = self.H_Conv5(H_x5)

        # fusion
        M_H_x1 = torch.cat((M_x1, H_x1), dim=1)
        M_H_x1 = self.M_H_Conv1(M_H_x1)
        M_H_x2 = torch.cat((M_x2, H_x2), dim=1)
        M_H_x2 = self.M_H_Conv2(M_H_x2)
        M_L_x3 = torch.cat((M_x3, L_x3), dim=1)
        M_L_x3 = self.M_L_Conv3(M_L_x3)
        M_L_x4 = torch.cat((M_x4, L_x4), dim=1)
        M_L_x4 = self.M_L_Conv4(M_L_x4)

        # main decoder

        M_d5 = self.M_Up5(M_x5)
        M_d5 = torch.cat((M_L_x4, M_d5), dim=1)
        M_d5 = self.M_Up_conv5(M_d5)

        M_d4 = self.M_Up4(M_d5)
        M_d4 = torch.cat((M_L_x3, M_d4), dim=1)
        M_d4 = self.M_Up_conv4(M_d4)

        M_d3 = self.M_Up3(M_d4)
        M_d3 = torch.cat((M_H_x2, M_d3), dim=1)
        M_d3 = self.M_Up_conv3(M_d3)

        M_d2 = self.M_Up2(M_d3)
        M_d2 = torch.cat((M_H_x1, M_d2), dim=1)
        M_d2 = self.M_Up_conv2(M_d2)

        M_d1 = self.M_Conv_1x1(M_d2)

        # L decoder
        L_d5 = self.L_Up5(L_x5)
        L_d5 = torch.cat((M_L_x4, L_d5), dim=1)
        L_d5 = self.L_Up_conv5(L_d5)

        L_d4 = self.L_Up4(L_d5)
        L_d4 = torch.cat((M_L_x3, L_d4), dim=1)
        L_d4 = self.L_Up_conv4(L_d4)

        L_d3 = self.L_Up3(L_d4)
        L_d3 = torch.cat((L_x2, L_d3), dim=1)
        L_d3 = self.L_Up_conv3(L_d3)

        L_d2 = self.L_Up2(L_d3)
        L_d2 = torch.cat((L_x1, L_d2), dim=1)
        L_d2 = self.L_Up_conv2(L_d2)

        L_d1 = self.L_Conv_1x1(L_d2)

        # H decoder
        H_d5 = self.H_Up5(H_x5)
        H_d5 = torch.cat((H_x4, H_d5), dim=1)
        H_d5 = self.H_Up_conv5(H_d5)

        H_d4 = self.H_Up4(H_d5)
        H_d4 = torch.cat((H_x3, H_d4), dim=1)
        H_d4 = self.H_Up_conv4(H_d4)

        H_d3 = self.H_Up3(H_d4)
        H_d3 = torch.cat((M_H_x2, H_d3), dim=1)
        H_d3 = self.H_Up_conv3(H_d3)

        H_d2 = self.H_Up2(H_d3)
        H_d2 = torch.cat((M_H_x1, H_d2), dim=1)
        H_d2 = self.H_Up_conv2(H_d2)

        H_d1 = self.H_Conv_1x1(H_d2)

        return M_d1, L_d1, H_d1
    
    def get_M_net_params(self):
        return list(self.M_Conv1.parameters()) + list(self.M_Conv2.parameters()) + list(self.M_Conv3.parameters()) + list(self.M_Conv4.parameters()) + list(self.M_Conv5.parameters()) + \
               list(self.M_Up_conv5.parameters()) + list(self.M_Up_conv4.parameters()) + list(self.M_Up_conv3.parameters()) + list(self.M_Up_conv2.parameters()) + list(self.M_Conv_1x1.parameters())
    
    def get_L_net_params(self):
        return list(self.L_Conv1.parameters()) + list(self.L_Conv2.parameters()) + list(self.L_Conv3.parameters()) + list(self.L_Conv4.parameters()) + list(self.L_Conv5.parameters()) + \
               list(self.L_Up_conv5.parameters()) + list(self.L_Up_conv4.parameters()) + list(self.L_Up_conv3.parameters()) + list(self.L_Up_conv2.parameters()) + list(self.L_Conv_1x1.parameters())
    
    def get_H_net_params(self):
        return list(self.H_Conv1.parameters()) + list(self.H_Conv2.parameters()) + list(self.H_Conv3.parameters()) + list(self.H_Conv4.parameters()) + list(self.H_Conv5.parameters()) + \
               list(self.H_Up_conv5.parameters()) + list(self.H_Up_conv4.parameters()) + list(self.H_Up_conv3.parameters()) + list(self.H_Up_conv2.parameters()) + list(self.H_Conv_1x1.parameters())
    
    def get_fusion_params(self):
        return list(self.M_H_Conv1.parameters()) + list(self.M_H_Conv2.parameters()) + list(self.M_L_Conv3.parameters()) + list(self.M_L_Conv4.parameters())

def wavenetx(in_channels, num_classes, flen=8, nfil=16, **kwargs):
    print('Building WaveNetX model with %d %d-tap filters' % (nfil, flen))
    model = WaveNetX(in_channels, num_classes, flen=flen, nfil=nfil)
    init_weights(model, 'kaiming')
    return model

def plot_dwt(x_dwt, idx=0):
    if isinstance(x_dwt[0], torch.Tensor):
        _LL = x_dwt[0].detach().cpu()
        _LH = x_dwt[1][0].detach().cpu()
        _HL = x_dwt[1][1].detach().cpu()
        _HH = x_dwt[1][2].detach().cpu()
    elif isinstance(x_dwt[0], np.ndarray):
        _LL = torch.tensor(x_dwt[0])
        _LH = torch.tensor(x_dwt[1][0])
        _HL = torch.tensor(x_dwt[1][1])
        _HH = torch.tensor(x_dwt[1][2])
        if len(_LL.shape) == 3:
            _LL = _LL.unsqueeze(0)
            _LH = _LH.unsqueeze(0)
            _HL = _HL.unsqueeze(0)
            _HH = _HH.unsqueeze(0)
    else:
        raise ValueError('x_dwt should be a list of tensors or numpy arrays')
    # normalize to [0, 1]
    _LL = (_LL - _LL.min()) / (_LL.max() - _LL.min())
    _LH = (_LH - _LH.min()) / (_LH.max() - _LH.min())
    _HL = (_HL - _HL.min()) / (_HL.max() - _HL.min())
    _HH = (_HH - _HH.min()) / (_HH.max() - _HH.min())
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(_LL[idx].permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('LL')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(_LH[idx].permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('LH')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(_HL[idx].permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('HL')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(_HH[idx].permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('HH')
    plt.axis('off')
    return

def plot_idwt(x_idwt, x_orig=None):
    if isinstance(x_idwt, torch.Tensor):
        x_idwt = x_idwt.detach().cpu()
    elif isinstance(x_idwt, np.ndarray):
        x_idwt = torch.tensor(x_idwt)
        if len(x_idwt.shape) == 3:
            x_idwt = x_idwt.unsqueeze(0)
    else:
        raise ValueError('x_idwt should be a tensor or numpy array')
    # normalize to [0, 1]
    x_idwt = (x_idwt - x_idwt.min()) / (x_idwt.max() - x_idwt.min())
    if x_orig is None:
        plt.figure()
        plt.imshow(x_idwt[0].permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
    else:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(x_idwt[0].permute(1, 2, 0).numpy(), cmap='gray')
        plt.title('IDWT')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(x_orig[0].permute(1, 2, 0).numpy(), cmap='gray')
        plt.title('Original')
        plt.axis('off')
    return

def plot_fil(fil, fil_hi=None):
    if isinstance(fil, torch.Tensor):
        fil = fil.detach().cpu()
    elif isinstance(fil, np.ndarray) or isinstance(fil, list):
        fil = torch.tensor(fil)
    else:
        raise ValueError('fil should be a tensor, numpy array or list')
    fil = fil.squeeze()

    if len(fil.shape) != 1:
        raise ValueError('fil should be a 1D tensor')

    if fil_hi is None:
        plt.figure()
        plt.bar(range(fil.shape[0]), fil.numpy())
        plt.title('Filter Bank')
        plt.xlabel('Index')
    else:
        if isinstance(fil_hi, torch.Tensor):
            fil_hi = fil_hi.detach().cpu()
        elif isinstance(fil_hi, np.ndarray) or isinstance(fil_hi, list):
            fil_hi = torch.tensor(fil_hi)
        else:
            raise ValueError('fil_hi should be a tensor, numpy array or list')
        fil_hi = fil_hi.squeeze()

        if len(fil_hi.shape) != 1:
            raise ValueError('fb_hi should be a 1D tensor')
        plt.figure()
        plt.bar(range(fil.shape[0]), fil.numpy(), label='Low-pass FB')
        plt.bar(range(fil_hi.shape[0]), fil_hi.numpy(), label='High-pass FB', alpha=0.7)
        plt.title('Low-pass and High-pass FB')
        plt.xlabel('Index')
        plt.legend()
    return

def plot_fil_2d(fil_lo = None, fil_hi = None, fil_2d = None, figure_name='Filter Bank 2D'):
    fil_ll, fil_lh, fil_hl, fil_hh = None, None, None, None
    if fil_lo is None and fil_hi is None and fil_2d is None:
        raise ValueError('At least one of fil_lo, fil_hi or fil_conv should be provided')

    if fil_2d is not None:
        if isinstance(fil_2d[0], torch.Tensor):
            fil_ll = fil_2d[0].detach().cpu()
            fil_lh = fil_2d[1].detach().cpu()
            fil_hl = fil_2d[2].detach().cpu()
            fil_hh = fil_2d[3].detach().cpu()
        elif isinstance(fil_2d[0], np.ndarray or isinstance(fil_2d[0], list)):
            fil_ll = torch.tensor(fil_2d[0])
            fil_lh = torch.tensor(fil_2d[1])
            fil_hl = torch.tensor(fil_2d[2])
            fil_hh = torch.tensor(fil_2d[3])
        else:
            raise ValueError('fil_2d should be a list of tensors or numpy arrays')
    elif fil_lo is not None:
        if isinstance(fil_lo, torch.Tensor):
            fil_lo = fil_lo.detach().cpu()
        elif isinstance(fil_lo, np.ndarray) or isinstance(fil_lo, list):
            fil_lo = torch.tensor(fil_lo)
        else:
            raise ValueError('fil_lo should be a tensor, numpy array or list')
        
        fil_lo = fil_lo.squeeze()
        if len(fil_lo.shape) != 1:
            raise ValueError('fil_lo should be a 1D tensor')
        
        fil_lo = F.normalize(fil_lo, p=2, dim=-1)

        if fil_hi is None:
            fil_hi = fil_lo.flip(-1)
            fil_hi[:, ::2] *= -1
            fil_hi -= fil_hi.mean(dim=-1, keepdim=True)
        else:
            if isinstance(fil_hi, torch.Tensor):
                fil_hi = fil_hi.detach().cpu()
            elif isinstance(fil_hi, np.ndarray) or isinstance(fil_hi, list):
                fil_hi = torch.tensor(fil_hi)
            else:
                raise ValueError('fil_hi should be a tensor, numpy array or list')
            fil_hi = fil_hi.squeeze()
            if len(fil_hi.shape) != 1:
                raise ValueError('fil_hi should be a 1D tensor')
            fil_hi = F.normalize(fil_hi, p=2, dim=-1)
        
        fil_ll = torch.einsum('n,m->nm', fil_lo, fil_lo)
        fil_lh = torch.einsum('n,m->nm', fil_hi, fil_lo)
        fil_hl = torch.einsum('n,m->nm', fil_lo, fil_hi)
        fil_hh = torch.einsum('n,m->nm', fil_hi, fil_hi)
    
    plt.figure()
    plt.suptitle(figure_name)
    plt.subplot(2, 2, 1)
    plt.imshow(fil_ll.numpy(), cmap='gray')
    plt.title('LL')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(fil_lh.numpy(), cmap='gray')
    plt.title('LH')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(fil_hl.numpy(), cmap='gray')
    plt.title('HL')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(fil_hh.numpy(), cmap='gray')
    plt.title('HH')
    plt.axis('off')
    return

if __name__ == '__main__':

    from PIL import Image
    import torchvision.transforms as transforms
    import os
    import sys
    
    # if -v is passed as argument, then plot the DWT and IDWT images
    plot_img = False
    if '-v' in sys.argv:
        plot_img = True


    _current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = _current_dir+'/../../dataset/GLAS/Test Folder/img/0001.png'
    img = Image.open(image_path)
    img_torch = transforms.ToTensor()(img).unsqueeze(0)

    print('Image size: ', img_torch.size())
    b, c, h_img, w_img = img_torch.size()
    crop_size = 512
    rnd_h_off = torch.randint(0, h_img - crop_size + 1, (1,)).item()
    rnd_w_off = torch.randint(0, w_img - crop_size + 1, (1,)).item()
    img_crop = img_torch[:, :, rnd_h_off:rnd_h_off + crop_size, rnd_w_off:rnd_w_off + crop_size]

    nfil = 16
    rand_dwt_layer = DWT_1lvl(nfil=nfil)
    img_rand_dwt = rand_dwt_layer(img_crop)

    # if plot_img:
    #     for idx in range(nfil):
    #         img_idx_dwt = get_img_dwt(img_rand_dwt, idx=idx, nfil=nfil)
    #         plot_dwt(img_idx_dwt)

    # test with dwt2 from pywt with db4
    wletstr = 'db4'
    wavelet = pywt.Wavelet(wletstr)
    img_dwt = pywt.dwt2(img_crop.squeeze().numpy(), wavelet)

    print('LL '+wletstr+':', img_dwt[0].shape)
    print('LH '+wletstr+':', img_dwt[1][0].shape)
    print('HL '+wletstr+':', img_dwt[1][1].shape)
    print('HH '+wletstr+':', img_dwt[1][2].shape)

    # plot the DWT filter from wavelet
    dec_lo = wavelet.dec_lo
    dec_hi = wavelet.dec_hi
    print(' DWT LP '+wletstr+': ', dec_lo)
    print(' DWT HP '+wletstr+': ', dec_hi)
    if plot_img:
        plot_fil(dec_lo, dec_hi)
        plot_fil_2d(fil_lo=dec_lo, fil_hi=dec_hi)

    rec_lo = wavelet.rec_lo
    rec_hi = wavelet.rec_hi
    print('IDWT LP '+wletstr+': ', rec_lo)
    print('IDWT HP '+wletstr+': ', rec_hi)
    if plot_img:
        plot_fil(rec_lo, rec_hi)
        plot_fil_2d(fil_lo=rec_lo, fil_hi=rec_hi)
    

    if plot_img:
        plot_dwt(img_dwt)

    # test idwt2 from pywt
    img_idwt = pywt.idwt2(img_dwt, wavelet)

    if plot_img:
        plot_idwt(img_idwt, img_crop)

    if plot_img:
        plt.show()

    exit(-1)