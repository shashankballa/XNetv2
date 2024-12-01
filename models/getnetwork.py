import sys
from models import *
import torch.nn as nn

def get_network(network, in_channels, num_classes, version=1, flen=8, nfil=16, flen_start=4, nfil_start=4, flen_step=4, nfil_step=4, *args, **kwargs):

    # 2d networks
    if network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'XNetv2' or network == 'xnetv2':
        net = xnetv2(in_channels, num_classes)
    # 3d networks
    elif network == 'XNetv2_3D_min' or network == 'xnetv2_3d_min':
        net = xnetv2_3d_min(in_channels, num_classes)
    elif network == 'MLWNet' or network == 'mlwnet':
        net = mlwnet(in_channels, num_classes)

    # WaveNetX
    elif network == 'WaveNetX' or network == 'wavenetx':
        net = wavenetx(in_channels, num_classes, flen=flen, nfil=nfil, version=version)
    elif network.lower() == 'wavenetxv0':
        net = wavenetx(in_channels, num_classes, version=0, flen=flen, nfil=nfil)
    elif network.lower() == 'wavenetxv1':
        net = wavenetx(in_channels, num_classes, version=1, flen=flen, nfil=nfil)
    elif network.lower() == 'wavenetxv2':
        net = wavenetx(in_channels, num_classes, version=2, flen_start=flen_start, nfil_start=nfil_start, flen_step=flen_step, nfil_step=nfil_step)

    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
