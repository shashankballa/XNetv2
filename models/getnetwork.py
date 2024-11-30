import sys
from models import *
import torch.nn as nn

def get_network(network, in_channels, num_classes, nfil=16, flen=8, **kwargs):

    # 2d networks
    if network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'XNetv2' or network == 'xnetv2':
        net = xnetv2(in_channels, num_classes)
    elif network == 'WaveNetX' or network == 'wavenetx':
        net = wavenetx(in_channels, num_classes, flen=flen, nfil=nfil)
    elif network == 'WaveNetX2' or network == 'wavenetx2':
        net = wavenetx2(in_channels, num_classes, **kwargs)
    # 3d networks
    elif network == 'XNetv2_3D_min' or network == 'xnetv2_3d_min':
        net = xnetv2_3d_min(in_channels, num_classes)
    elif network == 'MLWNet' or network == 'mlwnet':
        net = mlwnet(in_channels, num_classes)
    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
