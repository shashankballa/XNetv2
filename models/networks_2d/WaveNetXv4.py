import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import numpy as np
from typing import Sequence, Tuple, Union, List
import pywt
import matplotlib.pyplot as plt
from einops import rearrange
from torch.nn import init

class IDWT_1lvl(nn.Module):
    def __init__(self, out_channels=1):
        super(IDWT_1lvl, self).__init__()
        self.out_channels = out_channels
    
    def get_fb_2d_list(self, dwt_fb_hi=None, dwt_fb_his=None, out_channels=None):
        if dwt_fb_his is None:
            if dwt_fb_hi is None:
                raise ValueError('Both dwt_fb_hi and dwt_fb_his cannot be None')
            dwt_fb_his = [dwt_fb_hi]
        if out_channels is None:
            out_channels = self.out_channels

        max_flen = max([_fb_hi.shape[1] for _fb_hi in dwt_fb_his])

        fb_ll, fb_lh, fb_hl, fb_hh = [], [], [], []
        for _fb_hi in dwt_fb_his:
            _nfil, _flen = _fb_hi.shape

            _fb_hi = _fb_hi - _fb_hi.mean(dim=-1, keepdim=True)
            _fb_hi = F.normalize(_fb_hi, p=2, dim=-1)

            _fb_lo = _fb_hi.flip(-1)
            _fb_lo[:, ::2] *= -1

            fb_hi = _fb_hi.flip(-1) # flip the Synthesis filters
            fb_lo = _fb_lo.flip(-1)

            flip_dims = (-1, -2)

            _fb_ll = torch.einsum('nf,ng->nfg', fb_lo, fb_lo).flip(dims=flip_dims).view(_nfil, 1, _flen, _flen).repeat(out_channels, 1, 1, 1)
            _fb_lh = torch.einsum('nf,ng->nfg', fb_lo, fb_hi).flip(dims=flip_dims).view(_nfil, 1, _flen, _flen).repeat(out_channels, 1, 1, 1)
            _fb_hl = torch.einsum('nf,ng->nfg', fb_hi, fb_lo).flip(dims=flip_dims).view(_nfil, 1, _flen, _flen).repeat(out_channels, 1, 1, 1)
            _fb_hh = torch.einsum('nf,ng->nfg', fb_hi, fb_hi).flip(dims=flip_dims).view(_nfil, 1, _flen, _flen).repeat(out_channels, 1, 1, 1)

            pads = [(max_flen - _fb_hi.shape[1]) // 2] * 4
            fb_ll.append(F.pad(_fb_ll, pads, mode='constant', value=0))
            fb_lh.append(F.pad(_fb_lh, pads, mode='constant', value=0))
            fb_hl.append(F.pad(_fb_hl, pads, mode='constant', value=0))
            fb_hh.append(F.pad(_fb_hh, pads, mode='constant', value=0))

        fb_ll = torch.cat(fb_ll, dim=0)
        fb_lh = torch.cat(fb_lh, dim=0)
        fb_hl = torch.cat(fb_hl, dim=0)
        fb_hh = torch.cat(fb_hh, dim=0)
        
        return [fb_ll, fb_lh, fb_hl, fb_hh]
    
    def get_pads(self, flen):
        padl = (2 * flen - 3) // 2
        padr = (2 * flen - 3) // 2
        padt = (2 * flen - 3) // 2
        padb = (2 * flen - 3) // 2
        return padl, padr, padt, padb
    
    def safe_unpad(self, x, flen):
        padl, padr, padt, padb = self.get_pads(flen)
        if padt > 0:
            x = x[..., padt:, :]
        if padb > 0:
            x = x[..., :-padb, :]
        if padl > 0:
            x = x[..., padl:]
        if padr > 0:
            x = x[..., :-padr]
        return x
    
    def forward(self, x: Tuple, dwt_fb_hi: torch.Tensor = None, dwt_fb_his: torch.Tensor = None, out_channels: int = None) -> torch.Tensor:
        if out_channels is None:
            out_channels = self.out_channels
        fb = self.get_fb_2d_list(dwt_fb_hi=dwt_fb_hi, dwt_fb_his=dwt_fb_his, out_channels=out_channels)
        x_idwt = F.conv_transpose2d(x[0], fb[0].to(x[0].device), stride=2, groups=out_channels) + \
                    F.conv_transpose2d(x[1][0], fb[1].to(x[1][0].device), stride=2, groups=out_channels) + \
                    F.conv_transpose2d(x[1][1], fb[2].to(x[1][1].device), stride=2, groups=out_channels) + \
                    F.conv_transpose2d(x[1][2], fb[3].to(x[1][2].device), stride=2, groups=out_channels)
        flen = fb[0].shape[-1]
        x_idwt = self.safe_unpad(x_idwt, flen)
        return x_idwt

NFLENS = 4
SYMNFIL = False

class DWT_mtap(nn.Module):

    def __init__(self, nflens=NFLENS , flen_start = 4, nfil_start = 4, flen_step = 4, nfil_step = 4, inp_channels = None, 
                 symm_nfils = SYMNFIL, # Make false for backward compatibility
                 fbl1_nrows = 8,
                 pad_mode="replicate"):
        '''
        DWT_mtap: 1-level DWT with multi-tap Quadrature Mirror Filter Banks
        Args:
            nflens: number of filter lengths
            flen_start: starting filter length
            nfil_start: starting number of filters
            flen_step: step size for filter length
            nfil_step: step size for number of filters
            pad_mode: padding mode
        '''

        super(DWT_mtap, self).__init__()

        if (flen_start % 2 ) + (flen_step % 2) > 0:
            raise ValueError('Filter length `flen_start` and its step size `flen_start` should be even numbers')

        self.nflens = nflens
        self.flen_step = flen_step

        self.flens = torch.tensor([flen_start + self.flen_step*i for i in range(self.nflens)])

        self.flen_max = torch.max(self.flens)

        self.nfil_step = nfil_step

        if symm_nfils:
            self.nfils = torch.tensor([nfil_start + self.nfil_step * (self.nflens // 2 - abs(i - self.nflens // 2)) for i in range(self.nflens)])
        else:
            self.nfils = torch.tensor([nfil_start + self.nfil_step*i for i in range(self.nflens)])

        self.nfil = torch.sum(self.nfils)
        self.fb_his0, self.fb_his1, self.fb_his2, self.fb_his3, self.fb_his4, self.fb_his5, self.fb_his6, self.fb_his7 = None, None, None, None, None, None, None, None
        if self.nflens >= 1: self.fb_his0 = nn.Parameter(torch.rand((self.nfils[0], self.flens[0])) - 0.5)
        if self.nflens >= 2: self.fb_his1 = nn.Parameter(torch.rand((self.nfils[1], self.flens[1])) - 0.5)
        if self.nflens >= 3: self.fb_his2 = nn.Parameter(torch.rand((self.nfils[2], self.flens[2])) - 0.5)
        if self.nflens >= 4: self.fb_his3 = nn.Parameter(torch.rand((self.nfils[3], self.flens[3])) - 0.5)
        if self.nflens >= 5: self.fb_his4 = nn.Parameter(torch.rand((self.nfils[4], self.flens[4])) - 0.5)
        if self.nflens >= 6: self.fb_his5 = nn.Parameter(torch.rand((self.nfils[5], self.flens[5])) - 0.5)
        if self.nflens >= 7: self.fb_his6 = nn.Parameter(torch.rand((self.nfils[6], self.flens[6])) - 0.5)
        if self.nflens >= 8: self.fb_his7 = nn.Parameter(torch.rand((self.nfils[7], self.flens[7])) - 0.5)
        self.pad_mode = pad_mode
        self.inp_channels = inp_channels

        # add scales for losses based on filter lengths: larger flen -> higher loss scale
        _loss_offset = self.flen_max // 2
        self.loss_scales = torch.tensor([(self.nfils[i]*self.flens[i]**2 + _loss_offset) for i in range(self.nflens)], dtype=torch.float32) 
        self.loss_scales = self.loss_scales / torch.sum(self.loss_scales)
        self.fbl1_nrows = fbl1_nrows

    
    def get_fb_hi_list(self):
        fb_hi_list = [self.fb_his0, self.fb_his1, self.fb_his2, self.fb_his3, self.fb_his4, self.fb_his5, self.fb_his6, self.fb_his7]
        return fb_hi_list[:self.nflens]
    
    def get_pads(self, x_shape):
        padb = (2 * self.flen_max - 3) // 2
        padt = (2 * self.flen_max - 3) // 2
        if x_shape[2] % 2 != 0:
            padb += 1
        padr = (2 * self.flen_max - 3) // 2
        padl = (2 * self.flen_max - 3) // 2
        if x_shape[3] % 2 != 0:
            padl += 1
        return [padl, padr, padt, padb]
    
    def get_fb_2d_list(self, inp_channels = None, for_vis=False):
        if inp_channels is None:
            inp_channels = self.inp_channels
        fb_ll, fb_lh, fb_hl, fb_hh = [], [], [], []
        for _fb_hi in self.get_fb_hi_list():

            _nfil, _flen = _fb_hi.shape

            _fb_hi = _fb_hi - _fb_hi.mean(dim=-1, keepdim=True)
            _fb_hi = F.normalize(_fb_hi, p=2, dim=-1)

            _fb_lo = _fb_hi.flip(-1)
            _fb_lo[:, ::2] *= -1

            _fb_hh = torch.einsum('nf,ng->nfg', _fb_hi, _fb_hi)
            _fb_hl = torch.einsum('nf,ng->nfg', _fb_hi, _fb_lo)
            _fb_lh = torch.einsum('nf,ng->nfg', _fb_lo, _fb_hi)
            _fb_ll = torch.einsum('nf,ng->nfg', _fb_lo, _fb_lo)

            _padval = 0
            if not for_vis:
                _fb_hh = _fb_hh.view(_nfil, 1, _flen, _flen).repeat(inp_channels, 1, 1, 1)
                _fb_hl = _fb_hl.view(_nfil, 1, _flen, _flen).repeat(inp_channels, 1, 1, 1)
                _fb_lh = _fb_lh.view(_nfil, 1, _flen, _flen).repeat(inp_channels, 1, 1, 1)
                _fb_ll = _fb_ll.view(_nfil, 1, _flen, _flen).repeat(inp_channels, 1, 1, 1)
            else:
                # map tp [0, 1] for visualization
                _max = torch.max(torch.tensor([_fb_hh.max(), _fb_hl.max(), _fb_lh.max(), _fb_ll.max()]))
                _min = torch.min(torch.tensor([_fb_hh.min(), _fb_hl.min(), _fb_lh.min(), _fb_ll.min()]))
                _maxx = torch.max(torch.tensor([_max.abs(), _min.abs()]))
                _fb_hh = (_fb_hh/_maxx + 1) / 2
                _fb_hl = (_fb_hl/_maxx + 1) / 2
                _fb_lh = (_fb_lh/_maxx + 1) / 2
                _fb_ll = (_fb_ll/_maxx + 1) / 2
                _padval = 0.5

            pads = [(self.flen_max - _fb_hi.shape[1]) // 2] * 4
            _fb_hh = F.pad(_fb_hh, pads, mode='constant', value=_padval)
            _fb_hl = F.pad(_fb_hl, pads, mode='constant', value=_padval)
            _fb_lh = F.pad(_fb_lh, pads, mode='constant', value=_padval)
            _fb_ll = F.pad(_fb_ll, pads, mode='constant', value=_padval)

            fb_hh.append(_fb_hh)
            fb_hl.append(_fb_hl)
            fb_lh.append(_fb_lh)
            fb_ll.append(_fb_ll)
        
        fb_ll = torch.cat(fb_ll, dim=0)
        fb_lh = torch.cat(fb_lh, dim=0)
        fb_hl = torch.cat(fb_hl, dim=0)
        fb_hh = torch.cat(fb_hh, dim=0)
        return fb_ll, fb_lh, fb_hl, fb_hh
    
    def get_fb_hi_orthnorm_loss(self):
        """
        Compute orthonormality loss for all fb_hi filter banks in DWT_mtap.
        Ensures that the rows of each filter bank are orthogonal and unit norm.

        Returns:
            torch.Tensor: Orthonormality loss.
        """
        orthonormal_loss = 0.0
        f_idx = 0
        for _fb_hi in self.get_fb_hi_list():
            # Normalize the filter bank rows
            _fb_hi = _fb_hi - _fb_hi.mean(dim=-1, keepdim=True)
            _fb_hi = F.normalize(_fb_hi, p=2, dim=-1)

            # Compute Gram matrix: G = W * W^T
            gram_matrix = _fb_hi @ _fb_hi.T

            # Orthonormality loss: ||G - I||_F^2
            identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
            _loss = torch.linalg.norm(gram_matrix - identity, ord='fro') ** 2
            orthonormal_loss += _loss * self.loss_scales[f_idx]
            f_idx += 1
        return orthonormal_loss

    def get_fb_hi_orthnorm_loss_v2(self):
        """
        Pad all filter banks symmetrically to the maximum filter length and stack them together.
        If the total number of filters `nfil` is less than the half maximum filter length `flen_max`,
        then split filter banks alternately into `nfil/_nrows_dvsr` rows and stack them together.
        """
        orth_loss = 0.0
        fb_his = self.get_fb_hi_list()
        fb_his = [fb_hi - fb_hi.mean(dim=-1, keepdim=True) for fb_hi in fb_his]
        fb_his = [F.normalize(fb_hi, p=2, dim=-1) for fb_hi in fb_his]
        fb_his = [F.pad(fb_hi, (self.flen_max - fb_hi.shape[1] // 2, self.flen_max - fb_hi.shape[1] // 2), mode='constant', value=0) for fb_hi in fb_his]
        fb_his = torch.cat(fb_his, dim=0)
        n_splits = torch.ceil(fb_his.shape[0]/torch.tensor(self.fbl1_nrows)).int()
        for i in range(n_splits):
            _fb_hi_split = fb_his[i::n_splits]
            gram_matrix = _fb_hi_split @ _fb_hi_split.T
            identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
            _loss = torch.linalg.norm(gram_matrix - identity, ord='fro') ** 2
            orth_loss += _loss

        return orth_loss

    def get_fb_hh_orthnorm_loss(self):
        """
        Compute outer product of the high-pass filter banks and pad them symmetrically to the maximum filter length.
        flatten the filter banks and compute the Gram matrix.
        """
        orth_loss = 0.0
        fb_his = self.get_fb_hi_list()
        fb_his = [fb_hi - fb_hi.mean(dim=-1, keepdim=True) for fb_hi in fb_his]
        fb_his = [F.normalize(fb_hi, p=2, dim=-1) for fb_hi in fb_his]
        nfils = [fb_hi.shape[0] for fb_hi in fb_his]
        fb_hh_flats = torch.tensor([]).to(fb_his[0].device)
        for i in range(len(fb_his)):
            fb_hi = fb_his[i]
            fb_lo = fb_hi.flip(-1)
            fb_lo[:, ::2] *= -1
            fb_hh = torch.einsum('nf,ng->nfg', fb_lo, fb_lo)
            fb_hh = F.pad(fb_hh, [(self.flen_max - fb_hi.shape[1]) // 2] * 4, mode='constant', value=0)
            # print("outer product shape: ", fb_hh.shape)
            fb_hh_flat = fb_hh.reshape(nfils[i], -1)
            fb_hh_flats = torch.cat((fb_hh_flats, fb_hh_flat), dim=0)

        n_splits = torch.ceil(len(fb_hh_flats)/torch.tensor(self.fbl1_nrows)).int()
        for i in range(n_splits):
            _fb_hh_split = fb_hh_flats[i::n_splits]
            # print("_fb_hh_split shape: ", _fb_hh_split.shape)
            gram_matrix = _fb_hh_split @ _fb_hh_split.T
            identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
            _loss = torch.linalg.norm(gram_matrix - identity, ord='fro') ** 2
            orth_loss += _loss
        return orth_loss
    
    def forward(self, x):
        x_pad = F.pad(x, self.get_pads(x.shape), mode=self.pad_mode)
        fb_ll, fb_lh, fb_hl, fb_hh = self.get_fb_2d_list(inp_channels=x.shape[1])
        x_ll = F.conv2d(x_pad, fb_ll.to(x_pad.device), stride=2, groups=x.shape[1])
        x_lh = F.conv2d(x_pad, fb_lh.to(x_pad.device), stride=2, groups=x.shape[1])
        x_hl = F.conv2d(x_pad, fb_hl.to(x_pad.device), stride=2, groups=x.shape[1])
        x_hh = F.conv2d(x_pad, fb_hh.to(x_pad.device), stride=2, groups=x.shape[1])
        return x_ll, (x_lh, x_hl, x_hh)

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
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
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

class WaveNetXv4(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, nflens=NFLENS, flen_start = 4, nfil_start = 4, flen_step = 4, nfil_step = 4, 
                 symm_nfils = SYMNFIL, # Make false for backward compatibility
                 fbl1_nrows = 8,
                 pad_mode="replicate"):
        super(WaveNetXv4, self).__init__()

        # wavelet block
        self.dwt = DWT_mtap(flen_start=flen_start, nfil_start=nfil_start, flen_step=flen_step, nfil_step=nfil_step, nflens=nflens, inp_channels=in_channels, symm_nfils=symm_nfils,
                            fbl1_nrows=fbl1_nrows, pad_mode=pad_mode)
        self.idwt = IDWT_1lvl(out_channels=num_classes)
    
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
        self.L_Conv1 = conv_block(ch_in=2*self.dwt.nfil*in_channels, ch_out=64)
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
        self.L_Conv_1x1 = nn.Conv2d(64, 2*self.dwt.nfil*num_classes, kernel_size=1, stride=1, padding=0)

        # H network
        self.H_Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.H_Conv1 = conv_block(ch_in=2*self.dwt.nfil*in_channels, ch_out=64)
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
        self.H_Conv_1x1 = nn.Conv2d(64, 2*self.dwt.nfil*num_classes, kernel_size=1, stride=1, padding=0)

        # fusion
        self.M_H_Conv1 = conv_block(ch_in=128, ch_out=64)
        self.M_H_Conv2 = conv_block(ch_in=256, ch_out=128)
        self.M_L_Conv3 = conv_block(ch_in=512, ch_out=256)
        self.M_L_Conv4 = conv_block(ch_in=1024, ch_out=512)

    def forward(self, x_main, *args, **kwargs): #ignore excess
        # main encoder

        x_LL, (x_LH, x_HL, x_HH) = self.dwt(x_main)
        # x_LHHL = x_HL + x_LH + x_HH

        x_LLHH = torch.cat((x_LL, x_HH), dim=1)
        x_LHHL = torch.cat((x_LH, x_HL), dim=1)

        # Resize x_LLHH and x_LHHL to match the spatial dimensions of x_main
        x_LLHH = F.interpolate(x_LLHH, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)
        x_LHHL = F.interpolate(x_LHHL, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)

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
        L_x1 = self.L_Conv1(x_LLHH)
        L_x2 = self.L_Maxpool(L_x1)
        L_x2 = self.L_Conv2(L_x2)
        L_x3 = self.L_Maxpool(L_x2)
        L_x3 = self.L_Conv3(L_x3)
        L_x4 = self.L_Maxpool(L_x3)
        L_x4 = self.L_Conv4(L_x4)
        L_x5 = self.L_Maxpool(L_x4)
        L_x5 = self.L_Conv5(L_x5)

        # H encoder
        H_x1 = self.H_Conv1(x_LHHL)
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

        L_d1 = self.L_Conv_1x1(L_d2) # ll component

        L_d1 = L_d1.chunk(2, dim=1)

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

        H_d1 = self.H_Conv_1x1(H_d2) # lh, hl, hh components
        
        # split H_d1 into LH, HL, HH components
        H_d1 = H_d1.chunk(2, dim=1)

        # IDWT
        M_d0 = self.idwt([L_d1[0], [H_d1[0], H_d1[1], L_d1[1]]], dwt_fb_his=self.dwt.get_fb_hi_list())

        # shrink M_d0 to the original size
        M_d0 = F.interpolate(M_d0, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)
        
        return M_d1 + M_d0

def wavenetxv4(in_channels, num_classes, nflens=NFLENS, flen_start=4, nfil_start=4, flen_step=4, nfil_step=4, 
             symm_nfils=SYMNFIL, fbl1_nrows=8):
    version=4
    if symm_nfils:
        nfils = [nfil_start + nfil_step * (nflens // 2 - abs(i - nflens // 2)) for i in range(nflens)]
    else:
        nfils = [nfil_start + i*nfil_step for i in range(nflens)]
    flens = [flen_start + i*flen_step for i in range(nflens)]
    lst_str = '[%d' + ', %d'*(nflens-1) + ']'
    print(('Building WaveNetXv'+str(version)+' model with '+lst_str+'-number '+lst_str+'-tap filters') % (*nfils, *flens))
    
    model = WaveNetXv4(in_channels, num_classes, nflens=nflens, flen_start=flen_start, nfil_start=nfil_start, flen_step=flen_step, nfil_step=nfil_step, 
                       symm_nfils=symm_nfils, fbl1_nrows=fbl1_nrows)
    init_weights(model, 'kaiming')
    return model