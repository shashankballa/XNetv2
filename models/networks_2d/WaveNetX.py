import torch
import torch.nn as nn
import torch.nn.functional as F
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

    print('initialize network with %s' % init_type)
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
    def __init__(self, dec_lo = None, fil_len = 8, pad_mode="replicate"):

        super(DWT_1lvl, self).__init__()

        if dec_lo is None:
            dec_lo = torch.rand(fil_len) - 0.5
            dec_lo = dec_lo / dec_lo.norm()

        self.dec_lo = dec_lo
        self.pad_mode = pad_mode

    def forward(self, x):

        b, c, h, w = x.shape

        dec_fil_lo = torch.reshape(self.dec_lo, [-1])
        dec_fil_hi = dec_fil_lo.flip(0)
        dec_fil_hi[::2] *= -1
        dec_fil_hi -= dec_fil_hi.mean()
        dec_fil_ll = torch.unsqueeze(dec_fil_lo, dim=-1) * torch.unsqueeze(dec_fil_lo, dim=0) # outer product: lo * lo
        dec_fil_lh = torch.unsqueeze(dec_fil_hi, dim=-1) * torch.unsqueeze(dec_fil_lo, dim=0) # outer product: hi * lo
        dec_fil_hl = torch.unsqueeze(dec_fil_lo, dim=-1) * torch.unsqueeze(dec_fil_hi, dim=0) # outer product: lo * hi
        dec_fil_hh = torch.unsqueeze(dec_fil_hi, dim=-1) * torch.unsqueeze(dec_fil_hi, dim=0) # outer product: hi * hi

        dec_dwt_kernel = torch.stack([dec_fil_ll, dec_fil_lh, dec_fil_hl, dec_fil_hh], 0)
        dec_dwt_kernel = dec_dwt_kernel.repeat(c, 1, 1)
        dec_dwt_kernel = dec_dwt_kernel.unsqueeze(dim=1)

        _fil_len = len(self.dec_lo)

        padb = (2 * _fil_len - 3) // 2
        padt = (2 * _fil_len - 3) // 2
        if h % 2 != 0:
            padb += 1
        padr = (2 * _fil_len - 3) // 2
        padl = (2 * _fil_len - 3) // 2
        if w % 2 != 0:
            padl += 1

        x_pad = F.pad(x, [padl, padr, padt, padb], mode=self.pad_mode)
        x_dwt = F.conv2d(x_pad, dec_dwt_kernel, stride=2, groups=c)

        x_dwt = rearrange(x_dwt, 'b (c f) h w -> b c f h w', f=4)
        x_ll, x_lh, x_hl, x_hh = x_dwt.split(1, 2)
        x_dwt = [x_ll.squeeze(2), (x_lh.squeeze(2), x_hl.squeeze(2), x_hh.squeeze(2))]

        return x_dwt

class WaveNetX(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(WaveNetX, self).__init__()

        # wavelet block
        self.dwt = DWT_1lvl()

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
        self.L_Conv1 = conv_block(ch_in=in_channels, ch_out=64)
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
        self.H_Conv1 = conv_block(ch_in=in_channels, ch_out=64)
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

    def forward(self, x_main):
        # main encoder

        x_L, (x_LH, x_HL, x_HH) = self.dwt(x_main)
        x_H = x_HL + x_LH + x_HH

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

        return M_d1, L_d1, H_d1, None

def wavenetx(in_channels, num_classes):
    model = WaveNetX(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

def plot_dwt(x_dwt, idx=0):
    _LL = torch.tensor(x_dwt[0])
    _LH = torch.tensor(x_dwt[1][0])
    _HL = torch.tensor(x_dwt[1][1])
    _HH = torch.tensor(x_dwt[1][2])
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

if __name__ == '__main__':

    from PIL import Image
    import torchvision.transforms as transforms
    import os
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
    # img_gray = img_crop.mean(dim=1, keepdim=True)
    img_gray = img_crop

    # fil_len = 8
    # rand_lo = torch.rand(fil_len) - 0.5
    # unit_lo = rand_lo / rand_lo.norm()
    # print('unit_lo: ', unit_lo)
    # print('L2 norm: ', unit_lo.norm().item())  # Should be 1
    # print('Sum: ', unit_lo.sum().item())  # Should be 1

    unit_lo = torch.tensor(pywt.Wavelet('haar').dec_lo)
    rand_dwt_layer = DWT_1lvl(unit_lo)
    img_rand_dwt = rand_dwt_layer(img_gray)

    print('LL shape: ', img_rand_dwt[0].size())
    print('LH shape: ', img_rand_dwt[1][0].size())
    print('HL shape: ', img_rand_dwt[1][1].size())
    print('HH shape: ', img_rand_dwt[1][2].size())

    plot_dwt(img_rand_dwt)

    img_dwt2 = pywt.dwt2(img_gray.numpy(), 'haar', axes=(-2, -1))

    print('LL shape: ', img_dwt2[0].shape)
    print('LH shape: ', img_dwt2[1][0].shape)
    print('HL shape: ', img_dwt2[1][1].shape)
    print('HH shape: ', img_dwt2[1][2].shape)

    plot_dwt(img_dwt2)

    plt.show()

    exit(-1)