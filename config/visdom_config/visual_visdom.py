from visdom import Visdom
import torch
import numpy as np
import torch.nn.functional as F
import os

def visdom_initialization_XNetv2(env, port):
    visdom = Visdom(env=env, port=port)
    visdom.line([[0., 0., 0., 0., 0.]], [0.], win='train_loss', opts=dict(title='Train Loss', xlabel='Epoch', ylabel='Train Loss', legend=['Train Loss', 'Train Sup1', 'Train Sup2', 'Train Sup3', 'Train Unsup'], width=550, height=350))
    visdom.line([0.], [0.], win='train_jc', opts=dict(title='Train Jc', xlabel='Epoch', ylabel='Train Jc', legend=['Train Jc1'], width=550, height=350))
    visdom.line([[0., 0., 0.]], [0.], win='val_loss', opts=dict(title='Val Loss', xlabel='Epoch', ylabel='Val Loss', legend=['Val Sup1', 'Val Sup2', 'Val Sup3'], width=550, height=350))
    visdom.line([0.], [0.], win='val_jc', opts=dict(title='Val Jc', xlabel='Epoch', ylabel='Val Jc', legend=['Val Jc1'], width=550, height=350))
    return visdom

def visualization_XNetv2(vis, epoch, train_loss, train_loss_sup1, train_loss_sup2, train_loss_sup3, train_loss_unsup, train_m_jc1, val_loss_sup1, val_loss_sup2, val_loss_sup3, val_m_jc1):
    vis.line([[train_loss, train_loss_sup1, train_loss_sup2, train_loss_sup3, train_loss_unsup]], [epoch], win='train_loss', update='append')
    vis.line([train_m_jc1], [epoch], win='train_jc', update='append')
    vis.line([[val_loss_sup1, val_loss_sup2, val_loss_sup3]], [epoch], win='val_loss', update='append')
    vis.line([val_m_jc1], [epoch], win='val_jc', update='append')

def visual_image_sup(vis, mask_train, pred_train, mask_val, pred_val):

    vis.heatmap(mask_train, win='train_mask', opts=dict(title='Train Mask', colormap='Viridis'))
    vis.heatmap(pred_train, win='train_pred1', opts=dict(title='Train Pred', colormap='Viridis'))
    vis.heatmap(mask_val, win='val_mask', opts=dict(title='Val Mask', colormap='Viridis'))
    vis.heatmap(pred_val, win='val_pred1', opts=dict(title='Val Pred', colormap='Viridis'))

def vis_filter_bank_WaveNetX(vis, fil_lo = None, fil_hi = None, fb_2d_list = None, fil_idx = None, figure_name='Filter Bank 2D'):
    fil_ll, fil_lh, fil_hl, fil_hh = None, None, None, None
    
    if fil_lo is None and fil_hi is None and fb_2d_list is None:
        raise ValueError('At least one of fil_lo, fil_hi or fil_conv should be provided')

    if fb_2d_list is not None:
        _shape = fb_2d_list[0].shape
        for _fil in fb_2d_list:
            if not isinstance(_fil, torch.Tensor):
                raise ValueError('fb_2d_list should be a list of tensors')
            if _shape != _fil.shape:
                raise ValueError('All filters in fb_2d_list should have the same shape')
            
        if isinstance(fb_2d_list[0], torch.Tensor):
            fil_ll = fb_2d_list[0].detach().cpu().squeeze()
            fil_lh = fb_2d_list[1].detach().cpu().squeeze()
            fil_hl = fb_2d_list[2].detach().cpu().squeeze()
            fil_hh = fb_2d_list[3].detach().cpu().squeeze()
        elif isinstance(fb_2d_list[0], np.ndarray or isinstance(fb_2d_list[0], list)):
            fil_ll = torch.tensor(fb_2d_list[0])
            fil_lh = torch.tensor(fb_2d_list[1])
            fil_hl = torch.tensor(fb_2d_list[2])
            fil_hh = torch.tensor(fb_2d_list[3])
        else:
            raise ValueError('fb_2d_list should be a list of tensors or numpy arrays')
        
        if (fil_ll.dim() > 3) or (fil_ll.dim() < 2):
            raise ValueError('fb_2d_list should be a list of 2D or 3D tensors')
        
        if fil_ll.dim() == 3:
            if fil_idx is None:
                raise ValueError('fil_idx should be provided for 3D filters')
            if not isinstance(fil_idx, int):
                raise ValueError('fil_idx should be an integer')
        
            fil_idx = fil_idx % fil_ll.shape[0]
            fil_ll = fil_ll[fil_idx]
            fil_lh = fil_lh[fil_idx]
            fil_hl = fil_hl[fil_idx]
            fil_hh = fil_hh[fil_idx]

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
            fil_hi[::2] *= -1
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

    # the filters are very small, upsample to visualize
    fil_ll = torch.repeat_interleave(torch.repeat_interleave(fil_ll, 16, dim=0), 16, dim=1)
    fil_lh = torch.repeat_interleave(torch.repeat_interleave(fil_lh, 16, dim=0), 16, dim=1)
    fil_hl = torch.repeat_interleave(torch.repeat_interleave(fil_hl, 16, dim=0), 16, dim=1)
    fil_hh = torch.repeat_interleave(torch.repeat_interleave(fil_hh, 16, dim=0), 16, dim=1)

    # visualize in 2 x 2 grid
    vis.images([fil_ll.unsqueeze(0), fil_lh.unsqueeze(0), fil_hl.unsqueeze(0), fil_hh.unsqueeze(0)], nrow=2, win=figure_name, opts=dict(title=figure_name))

    