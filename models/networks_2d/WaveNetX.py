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

class DWT_1lvl(nn.Module):

    def __init__(self, fb_hi = None, flen = 8, nfil = 1, inp_channels = None,
                 pad_mode="replicate"):
        '''
        DWT_1lvl: 1-level multi-filter DWT using 2D convolution
        Args:
            fb_hi: low-pass filter bank
            flen: filter length
            nfil: number of filters
        '''

        super(DWT_1lvl, self).__init__()

        self.fb_hi = nn.Parameter(self.get_fb_hi(fb_hi, flen, nfil))

        # Apply orthogonal parametrization
        # P.register_parametrization(self, 'fb_hi', nn.utils.parametrizations.orthogonal())

        self.nfil = self.fb_hi.shape[0]
        self.flen = self.fb_hi.shape[1]
        self.pad_mode = pad_mode
        self.inp_channels = inp_channels

    def get_fb_hi(self, fb_hi = None, flen = 8, nfil = 1):
        if fb_hi is None:
            fb_hi = torch.rand((nfil, flen)) # - 0.5 # zero-mean not needed
        else:
            if not isinstance(fb_hi, torch.Tensor):
                if isinstance(fb_hi, np.ndarray) or isinstance(fb_hi, list):
                    fb_hi = torch.tensor(fb_hi)
                else:
                    raise ValueError('fb_hi should be a tensor or numpy array')
            if fb_hi.dim() > 2:
                raise ValueError('fb_hi should be a 1D or 2D tensor')
            if fb_hi.dim() == 1:
                fb_hi = fb_hi.unsqueeze(0)
        return fb_hi
    
    def get_pads(self, x_shape):
        padb = (2 * self.flen - 3) // 2
        padt = (2 * self.flen - 3) // 2
        if x_shape[2] % 2 != 0:
            padb += 1
        padr = (2 * self.flen - 3) // 2
        padl = (2 * self.flen - 3) // 2
        if x_shape[3] % 2 != 0:
            padl += 1
        return [padl, padr, padt, padb]
    
    def get_fb_2d_list(self, inp_channels = None, *args, **kwargs): #ignore excess
        if inp_channels is None:
            inp_channels = self.inp_channels
        fb_hi = F.normalize(self.fb_hi, p=2, dim=-1)
        fb_lo = fb_hi.flip(-1)
        fb_lo[:, ::2] *= -1
        # fb_lo -= fb_lo.mean(dim=-1, keepdim=True)
        fb_ll = torch.einsum('nf,ng->nfg', fb_hi, fb_hi)
        fb_lh = torch.einsum('nf,ng->nfg', fb_lo, fb_hi)
        fb_hl = torch.einsum('nf,ng->nfg', fb_hi, fb_lo)
        fb_hh = torch.einsum('nf,ng->nfg', fb_lo, fb_lo)
        # Prepare the 2D filter banks for conv2d
        fb_ll = fb_ll.view(fb_ll.shape[0], 1, fb_ll.shape[1], fb_ll.shape[2]).repeat(inp_channels, 1, 1, 1)
        fb_lh = fb_lh.view(fb_lh.shape[0], 1, fb_lh.shape[1], fb_lh.shape[2]).repeat(inp_channels, 1, 1, 1)
        fb_hl = fb_hl.view(fb_hl.shape[0], 1, fb_hl.shape[1], fb_hl.shape[2]).repeat(inp_channels, 1, 1, 1)
        fb_hh = fb_hh.view(fb_hh.shape[0], 1, fb_hh.shape[1], fb_hh.shape[2]).repeat(inp_channels, 1, 1, 1)
        return fb_ll, fb_lh, fb_hl, fb_hh
    
    def get_fb_hi_0_mean_loss(self):
        # Ensure that fb_lo is zero-mean
        fb_hi = F.normalize(self.fb_hi, p=2, dim=-1)
        fb_lo = fb_hi.flip(-1)
        fb_lo[:, ::2] *= -1
        return fb_lo.sum(dim=-1).abs().sum()

    def forward(self, x, *args, **kwargs): #ignore excess
        x_pad = F.pad(x, self.get_pads(x.shape), mode=self.pad_mode)
        print("Running v1")
        fb_ll, fb_lh, fb_hl, fb_hh = self.get_fb_2d_list(inp_channels=x.shape[1])
        x_ll = F.conv2d(x_pad, fb_ll.to(x_pad.device), stride=2, groups=x.shape[1])
        x_lh = F.conv2d(x_pad, fb_lh.to(x_pad.device), stride=2, groups=x.shape[1])
        x_hl = F.conv2d(x_pad, fb_hl.to(x_pad.device), stride=2, groups=x.shape[1])
        x_hh = F.conv2d(x_pad, fb_hh.to(x_pad.device), stride=2, groups=x.shape[1])

        return x_ll, (x_lh, x_hl, x_hh)

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
            fb_hi = F.normalize(_fb_hi, p=2, dim=-1)
            # Generate high-pass filters by flipping and changing signs
            fb_lo = fb_hi.flip(-1)
            fb_lo[:, ::2] *= -1
            fb_hi = fb_hi.flip(-1) # flip the Synthesis filters
            fb_lo = fb_lo.flip(-1)
            flip_dims = (-1, -2)
            _fb_ll = torch.einsum('nf,ng->nfg', fb_hi, fb_hi).flip(dims=flip_dims).view(_nfil, 1, _flen, _flen).repeat(out_channels, 1, 1, 1)
            _fb_lh = torch.einsum('nf,ng->nfg', fb_lo, fb_hi).flip(dims=flip_dims).view(_nfil, 1, _flen, _flen).repeat(out_channels, 1, 1, 1)
            _fb_hl = torch.einsum('nf,ng->nfg', fb_hi, fb_lo).flip(dims=flip_dims).view(_nfil, 1, _flen, _flen).repeat(out_channels, 1, 1, 1)
            _fb_hh = torch.einsum('nf,ng->nfg', fb_lo, fb_lo).flip(dims=flip_dims).view(_nfil, 1, _flen, _flen).repeat(out_channels, 1, 1, 1)
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

# bash scripts/run_py.sh mac_train_adaVal_WaveNetX.py -b 8 -l 2 -e 1000 -s 60  -w 30 --bs_step 90 --max_bs_steps 2 --fbl0 0.1 --fbl1 0.004 --seed 136 --nfil_step 0 --flen_step 4 --nfil 1 --nfil_step 1 --flen 2 -g 0.8

class DWT_mtap(nn.Module):

    def __init__(self, nflens=NFLENS , flen_start = 4, nfil_start = 4, flen_step = 4, nfil_step = 4, inp_channels = None, symm_nfils = SYMNFIL, # Make false for backward compatibility
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
        _loss_offset = 4
        self.loss_scales = torch.tensor([(flen + _loss_offset) / (self.flen_max + _loss_offset) for flen in self.flens])
        self.loss_scales = self.loss_scales / torch.sum(self.loss_scales)

    
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
            fb_hi = F.normalize(_fb_hi, p=2, dim=-1)
            fb_lo = fb_hi.flip(-1)
            fb_lo[:, ::2] *= -1
            # fb_lo -= fb_lo.mean(dim=-1, keepdim=True)
            _fb_ll = torch.einsum('nf,ng->nfg', fb_hi, fb_hi)
            _fb_lh = torch.einsum('nf,ng->nfg', fb_lo, fb_hi)
            _fb_hl = torch.einsum('nf,ng->nfg', fb_hi, fb_lo)
            _fb_hh = torch.einsum('nf,ng->nfg', fb_lo, fb_lo)

            _padval = 0
            if not for_vis:
                _fb_ll = _fb_ll.view(_nfil, 1, _flen, _flen).repeat(inp_channels, 1, 1, 1)
                _fb_lh = _fb_lh.view(_nfil, 1, _flen, _flen).repeat(inp_channels, 1, 1, 1)
                _fb_hl = _fb_hl.view(_nfil, 1, _flen, _flen).repeat(inp_channels, 1, 1, 1)
                _fb_hh = _fb_hh.view(_nfil, 1, _flen, _flen).repeat(inp_channels, 1, 1, 1)
            else:
                # map tp [0, 1] for visualization
                _max = torch.max(torch.tensor([_fb_ll.max(), _fb_lh.max(), _fb_hl.max(), _fb_hh.max()]))
                _min = torch.min(torch.tensor([_fb_ll.min(), _fb_lh.min(), _fb_hl.min(), _fb_hh.min()]))
                _maxx = torch.max(torch.tensor([_max.abs(), _min.abs()]))
                _fb_ll = (_fb_ll/_maxx + 1) / 2
                _fb_lh = (_fb_lh/_maxx + 1) / 2
                _fb_hl = (_fb_hl/_maxx + 1) / 2
                _fb_hh = (_fb_hh/_maxx + 1) / 2
                _padval = 0.5

            pads = [(self.flen_max - _fb_hi.shape[1]) // 2] * 4
            _fb_ll = F.pad(_fb_ll, pads, mode='constant', value=_padval)
            _fb_lh = F.pad(_fb_lh, pads, mode='constant', value=_padval)
            _fb_hl = F.pad(_fb_hl, pads, mode='constant', value=_padval)
            _fb_hh = F.pad(_fb_hh, pads, mode='constant', value=_padval)

            fb_ll.append(_fb_ll)
            fb_lh.append(_fb_lh)
            fb_hl.append(_fb_hl)
            fb_hh.append(_fb_hh)
        
        fb_ll = torch.cat(fb_ll, dim=0)
        fb_lh = torch.cat(fb_lh, dim=0)
        fb_hl = torch.cat(fb_hl, dim=0)
        fb_hh = torch.cat(fb_hh, dim=0)
        return fb_ll, fb_lh, fb_hl, fb_hh
    
    def get_fb_hi_0_mean_loss(self):
        # Ensure that fb_lo is zero-mean
        fb_hi_loss = 0.0
        f_idx = 0
        for _fb_hi in self.get_fb_hi_list():
            fb_hi = F.normalize(_fb_hi, p=2, dim=-1)
            fb_hi_loss += fb_hi.sum(dim=-1).abs().sum() * self.loss_scales[f_idx]
            f_idx += 1
        return fb_hi_loss
    
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
            fb_hi = F.normalize(_fb_hi, p=2, dim=-1)

            # Compute Gram matrix: G = W * W^T
            gram_matrix = _fb_hi @ _fb_hi.T

            # Orthonormality loss: ||G - I||_F^2
            identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
            _loss = torch.linalg.norm(gram_matrix - identity, ord='fro') ** 2
            orthonormal_loss += _loss * self.loss_scales[f_idx]
            f_idx += 1
        return orthonormal_loss
    
    def get_fb_hi_orthnorm_loss_v2(self, _nrows_fac=4):
        """
        Pad all filter banks symmetrically to the maximum filter length and stack them together.
        If the total number of filters `nfil` is less than the half maximum filter length `flen_max`,
        then split filter banks alternately into `nfil/_nrows_dvsr` rows and stack them together.
        """
        orth_loss = 0.0
        fb_his = self.get_fb_hi_list()
        fb_his = [F.normalize(fb_hi, p=2, dim=-1) for fb_hi in fb_his]
        fb_his = [F.pad(fb_hi, (self.flen_max - fb_hi.shape[1] // 2, self.flen_max - fb_hi.shape[1] // 2), mode='constant', value=0) for fb_hi in fb_his]
        fb_his = torch.cat(fb_his, dim=0)
        n_rows = (self.flen_max * _nrows_fac).int()
        n_splits = torch.ceil(fb_his.shape[0]/n_rows).int()
        for i in range(n_splits):
            _fb_hi_split = fb_his[i::n_splits]
            gram_matrix = _fb_hi_split @ _fb_hi_split.T
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

class WaveNetXv0(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, fb_hi=None, flen=8, nfil=1):
        super(WaveNetXv0, self).__init__()

        # wavelet block
        self.dwt = DWT_1lvl(fb_hi=fb_hi, flen=flen, nfil=nfil)
    
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

class WaveNetXv1(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, fb_hi=None, flen=8, nfil=1):
        super(WaveNetXv1, self).__init__()

        # wavelet block
        self.dwt = DWT_1lvl(fb_hi=fb_hi, flen=flen, nfil=nfil, inp_channels=in_channels)
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
        self.L_Conv_1x1 = nn.Conv2d(64, nfil*num_classes, kernel_size=1, stride=1, padding=0)

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
        self.H_Conv_1x1 = nn.Conv2d(64, 3*nfil*num_classes, kernel_size=1, stride=1, padding=0)

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

        L_d1 = self.L_Conv_1x1(L_d2) # ll component

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
        H_d1 = H_d1.chunk(3, dim=1)

        # IDWT
        M_d0 = self.idwt([L_d1, H_d1], self.dwt.fb_hi) 

        # shrink M_d0 to the original size
        M_d0 = F.interpolate(M_d0, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)
        
        return M_d1 + M_d0

class WaveNetXv2(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, flen_start=4, nfil_start=4, flen_step=4, nfil_step=4, nflens=NFLENS):
        super(WaveNetXv2, self).__init__()

        # wavelet block
        self.dwt = DWT_mtap(flen_start=flen_start, nfil_start=nfil_start, flen_step=flen_step, nfil_step=nfil_step, nflens=NFLENS, inp_channels=in_channels)
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
        self.L_Conv1 = conv_block(ch_in=self.dwt.nfil*in_channels, ch_out=64)
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
        self.L_Conv_1x1 = nn.Conv2d(64, self.dwt.nfil*num_classes, kernel_size=1, stride=1, padding=0)

        # H network
        self.H_Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.H_Conv1 = conv_block(ch_in=self.dwt.nfil*3*in_channels, ch_out=64)
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
        self.H_Conv_1x1 = nn.Conv2d(64, 3*self.dwt.nfil*num_classes, kernel_size=1, stride=1, padding=0)

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

        L_d1 = self.L_Conv_1x1(L_d2) # ll component

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
        H_d1 = H_d1.chunk(3, dim=1)

        # IDWT
        M_d0 = self.idwt([L_d1, H_d1], dwt_fb_his=self.dwt.get_fb_hi_list())

        # shrink M_d0 to the original size
        M_d0 = F.interpolate(M_d0, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)
        
        return M_d1 + M_d0

class WaveNetXv3(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, flen_start=4, nfil_start=4, flen_step=4, nfil_step=4, nflens=NFLENS):
        super(WaveNetXv3, self).__init__()

        # wavelet block
        self.dwt = DWT_mtap(flen_start=flen_start, nfil_start=nfil_start, flen_step=flen_step, nfil_step=nfil_step, nflens=NFLENS, inp_channels=in_channels)
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
        self.L_dwt_att_conv = nn.Conv2d(in_channels*2, 32, kernel_size=1, stride=1, padding=0)
        self.L_dwt_att_actv = nn.Sigmoid()
        self.L_Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.L_Conv1 = conv_block(ch_in=32, ch_out=64)
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
        self.L_Up3 = up_conv(ch_in=64, ch_out=32)
        self.L_Up_conv3 = conv_block(ch_in=64, ch_out=32)
        self.L_idwt_att_conv = nn.Conv2d(32, 2*self.dwt.nfil*num_classes, kernel_size=1, stride=1, padding=0)
        self.L_idwt_att_act = nn.Sigmoid()

        # H network
        self.H_dwt_att_conv = nn.Conv2d(in_channels*2, 32, kernel_size=1, stride=1, padding=0)
        self.H_dwt_att_actv = nn.Sigmoid()
        self.H_Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.H_Conv1 = conv_block(ch_in=self.dwt.nfil*2*in_channels, ch_out=64)
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
        self.H_Up3 = up_conv(ch_in=64, ch_out=32)
        self.H_Up_conv3 = conv_block(ch_in=64, ch_out=32)
        self.H_idwt_att_conv = nn.Conv2d(32, 2*self.dwt.nfil*num_classes, kernel_size=1, stride=1, padding=0)
        self.H_idwt_att_act = nn.Sigmoid()

        # fusion
        self.M_H_Conv1 = conv_block(ch_in=128, ch_out=64)
        self.M_H_Conv2 = conv_block(ch_in=256, ch_out=128)
        self.M_L_Conv3 = conv_block(ch_in=512, ch_out=256)
        self.M_L_Conv4 = conv_block(ch_in=1024, ch_out=512)

    def forward(self, x_main, *args, **kwargs): #ignore excess

        # DWT
        x_LL, (x_LH, x_HL, x_HH) = self.dwt(x_main)

        # main encoder
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
        x_L = torch.cat((x_LL, x_HH), dim=1)
        x_L = F.interpolate(x_L, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)
        L_x1 = self.L_dwt_att_actv(self.L_dwt_att_conv(torch.cat((x_main, x_LL), dim=1))) * x_LL

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
        x_H = torch.cat((x_LH, x_HL), dim=1)
        x_H = F.interpolate(x_H, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)
        H_x1 = self.H_dwt_att_actv(self.H_dwt_att_conv(torch.cat((x_main, x_LH, x_HL), dim=1))) * x_LH

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

        L_d1 = self.L_Conv_1x1(L_d2) # ll, hh components

        L_d1 = F.interpolate(x_L, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)
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

        H_d1 = self.H_Conv_1x1(H_d2) # lh, hl components
        H_d1 = H_d1.chunk(2, dim=1)

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

        # shrink M_d0 to the original size
        M_d0 = F.interpolate(M_d0, size=(x_main.shape[2], x_main.shape[3]), mode='bilinear', align_corners=False)
        
        return M_d1 + M_d0
    
latest_ver = 2

def wavenetx(in_channels, num_classes, version=1, flen=8, nfil=16, flen_start=4, nfil_start=4, flen_step=4, nfil_step=4, nflens=NFLENS, symm_nfils=SYMNFIL):
    if version > latest_ver:
        raise ValueError(('WaveNetXv%d model is not available yet') % version)
    if version >= 2:
        if symm_nfils:
            nfils = [nfil_start + nfil_step * (nflens // 2 - abs(i - nflens // 2)) for i in range(nflens)]
        else:
            nfils = [nfil_start + i*nfil_step for i in range(nflens)]
        flens = [flen_start + i*flen_step for i in range(nflens)]
        lst_str = '[%d' + ', %d'*(nflens-1) + ']'
        print(('Building WaveNetXv'+str(version)+' model with '+lst_str+'-number '+lst_str+'-tap filters') % (*nfils, *flens))
    else:
        print(('Building WaveNetXv'+str(version)+' model with %d %d-tap filters') % (nfil, flen))
    if version == 0:
        model = WaveNetXv0(in_channels, num_classes, flen=flen, nfil=nfil)
    elif version == 1:
        model = WaveNetXv1(in_channels, num_classes, flen=flen, nfil=nfil)
    elif version == 2:
        model = WaveNetXv2(in_channels, num_classes, flen_start=flen_start, nfil_start=nfil_start, flen_step=flen_step, nfil_step=nfil_step, nflens=nflens)
    init_weights(model, 'kaiming')
    return model

def get_img_dwt(img_dwt2, fil_idx=0, nfil=1):
    img_idx_dwt = [None, None]
    img_idx_dwt[0] = img_dwt2[0][:,fil_idx::nfil].detach().numpy()
    img_idx_dwt[1] = [img_dwt2[1][0][:,fil_idx::nfil].detach().numpy(), img_dwt2[1][1][:,fil_idx::nfil].detach().numpy(), img_dwt2[1][2][:,fil_idx::nfil].detach().numpy()]
    return img_idx_dwt

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
            raise ValueError('fb_lo should be a 1D tensor')
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
    if '--plot_img' in sys.argv:
        plot_img = True

    df = '%+.6f' # decimal format

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = _current_dir+'/../../dataset/GLAS/Test Folder/img/0001.png'
    img = Image.open(image_path)
    img_torch = transforms.ToTensor()(img).unsqueeze(0)

    mask_path = _current_dir+'/../../dataset/GLAS/Test Folder/labelcol/0001.png'
    mask = Image.open(mask_path)
    msk_torch = transforms.ToTensor()(mask).reshape(1, 1, mask.size[1], mask.size[0])
    msk_torch = (msk_torch > 0).float()

    print('Image size: ', img_torch.size())
    b, c, h_img, w_img = img_torch.size()
    crop_size = 512
    _resize = 128
    rnd_h_off = torch.randint(0, h_img - crop_size + 1, (1,)).item()
    rnd_w_off = torch.randint(0, w_img - crop_size + 1, (1,)).item()
    img_crop = img_torch[:, :, rnd_h_off:rnd_h_off + crop_size, rnd_w_off:rnd_w_off + crop_size]
    img_crop = transforms.Resize((_resize, _resize))(img_crop)
    msk_crop = msk_torch[:, :, rnd_h_off:rnd_h_off + crop_size, rnd_w_off:rnd_w_off + crop_size]
    msk_crop = transforms.Resize((_resize, _resize))(msk_crop)

    # test with dwt2 from pywt with db4
    wletstr = 'db4'
    wavelet = pywt.Wavelet(wletstr)

    # plot the DWT filter from wavelet
    dec_lo = wavelet.dec_lo
    dec_hi = wavelet.dec_hi
    print((' DWT LP '+wletstr+': ' + (df+' ') * len(dec_lo)) % tuple(dec_lo))
    print(('  \---------> Norm : ' + df) % np.square(dec_lo).sum())
    print(('   \--------> Sum  : ' + df) % np.sum(dec_lo))
    print((' DWT HP '+wletstr+': ' + (df+' ') * len(dec_hi)) % tuple(dec_hi))
    print(('  \---------> Norm : ' + df) % np.square(dec_hi).sum())
    print(('   \--------> Sum  : ' + df) % np.sum(dec_hi))
    if plot_img:
        plot_fil(dec_lo, dec_hi)
        plot_fil_2d(fil_lo=dec_lo, fil_hi=dec_hi)

    rec_lo = wavelet.rec_lo
    rec_hi = wavelet.rec_hi
    print(('IDWT LP '+wletstr+': ' +(df+' ') * len(rec_lo)) % tuple(rec_lo))
    print(('  \---------> Norm : ' +df) % np.square(rec_lo).sum())
    print(('   \--------> Sum  : ' +df) % np.sum(rec_lo))
    print(('IDWT HP '+wletstr+': ' +(df+' ') * len(rec_hi)) % tuple(rec_hi))
    print(('  \---------> Norm : ' +df) % np.square(rec_hi).sum())
    print(('   \--------> Sum  : ' +df) % np.sum(rec_hi))
    
    if plot_img:
        plot_fil(rec_lo, rec_hi)
        plot_fil_2d(fil_lo=rec_lo, fil_hi=rec_hi)

    img_dwt2 = pywt.dwt2(img_crop.squeeze(0).numpy(), wavelet)
    msk_dwt2 = pywt.dwt2(msk_crop.squeeze(0).numpy(), wavelet)

    if plot_img:
        plot_dwt(img_dwt2)
        plot_dwt(msk_dwt2)

    print('LL '+wletstr+' img:', img_dwt2[0].shape)
    print('LH '+wletstr+' img:', img_dwt2[1][0].shape)
    print('HL '+wletstr+' img:', img_dwt2[1][1].shape)
    print('HH '+wletstr+' img:', img_dwt2[1][2].shape)
    print('LL '+wletstr+' msk:', msk_dwt2[0].shape)
    print('LH '+wletstr+' msk:', msk_dwt2[1][0].shape)
    print('HL '+wletstr+' msk:', msk_dwt2[1][1].shape)
    print('HH '+wletstr+' msk:', msk_dwt2[1][2].shape)

    dwt_layer = DWT_1lvl(fb_hi=wavelet.dec_lo)
    img_dwt = dwt_layer(img_crop)
    msk_dwt = dwt_layer(msk_crop)

    print('LL img:', img_dwt[0].shape)
    print('LH img:', img_dwt[1][0].shape)
    print('HL img:', img_dwt[1][1].shape)
    print('HH img:', img_dwt[1][2].shape)

    if plot_img:
        plot_dwt(img_dwt)

    # Show mse between pywt and random layer
    mse_ll = torch.nn.functional.mse_loss(torch.tensor(img_dwt2[0]).unsqueeze(0), img_dwt[0])
    mse_lh = torch.nn.functional.mse_loss(torch.tensor(img_dwt2[1][0]).unsqueeze(0), img_dwt[1][0])
    mse_hl = torch.nn.functional.mse_loss(torch.tensor(img_dwt2[1][1]).unsqueeze(0), img_dwt[1][1])
    mse_hh = torch.nn.functional.mse_loss(torch.tensor(img_dwt2[1][2]).unsqueeze(0), img_dwt[1][2])

    print(('MSE LL  : '+df+'; MSE LH: '+df+'; MSE HL: '+df+'; MSE HH: '+df) % (mse_ll, mse_lh, mse_hl, mse_hh))

    ## IDWT

    # test idwt2 from pywt
    img_idwt2 = pywt.idwt2(img_dwt2, wavelet)
    msk_idwt2 = pywt.idwt2(msk_dwt2, wavelet)

    if plot_img:
        plot_idwt(img_idwt2, img_crop)
        plot_idwt(msk_idwt2, msk_crop)

    idwt_layer = IDWT_1lvl(out_channels=img_crop.shape[1])
    img_idwt = idwt_layer(img_dwt, dwt_fb_hi=dwt_layer.fb_hi)
        
    # show mse between pywt and manual layer
    mse_idwt = torch.nn.functional.mse_loss(torch.tensor(img_idwt2).unsqueeze(0), img_idwt)
    print(('MSE IDWT: '+df) % mse_idwt)

    if plot_img:
        plot_idwt(img_idwt, img_crop)

    nfil = 16
    flen = 8    
    rand_dwt = DWT_1lvl(nfil=nfil, flen=flen)

    img_rand_dwt = rand_dwt(img_crop)
    msk_rand_dwt = rand_dwt(msk_crop)

    print('LL rand:', img_rand_dwt[0].shape)
    print('LH rand:', img_rand_dwt[1][0].shape)
    print('HL rand:', img_rand_dwt[1][1].shape)
    print('HH rand:', img_rand_dwt[1][2].shape)

    if plot_img:
        for i in range(nfil):
            plot_dwt(get_img_dwt(img_rand_dwt, i, nfil))
            plot_dwt(get_img_dwt(msk_rand_dwt, i, nfil))


    rand_idwt = IDWT_1lvl()
    img_rand_idwt = rand_idwt(img_rand_dwt, dwt_fb_hi=rand_dwt.fb_hi, out_channels=img_crop.shape[1])
    msk_rand_idwt = rand_idwt(msk_rand_dwt, dwt_fb_hi=rand_dwt.fb_hi, out_channels=msk_crop.shape[1])

    print('IDWT rand:', img_rand_idwt.shape)

    if plot_img:
        plot_idwt(img_rand_idwt, img_crop)
        plot_idwt(msk_rand_idwt, msk_crop)
    
    # # test WaveNetX
    # ver = latest_ver
    # if '--ver 0' in sys.argv:
    #     ver = 0
    # elif '-ver 1' in sys.argv:
    #     ver = 1

    # model = wavenetx(in_channels=3, num_classes=2, version=ver, flen_start=2, nfil_start=2, flen_step=2, nfil_step=2, nflens=8)

    # pred = model(img_crop)

    # fb_lo_loss = model.dwt.get_fb_hi_0_mean_loss().detach().cpu().numpy()

    # print('FB_HI Loss:', fb_lo_loss)

    # if plot_img:
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(pred[0,0].detach().cpu().numpy(), cmap='gray')
    #     plt.title('Channel 0')
    #     plt.axis('off')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(pred[0,1].detach().cpu().numpy(), cmap='gray')
    #     plt.title('Channel 1')
    #     plt.axis('off')

    if plot_img:
        plt.show()

    exit(-1)