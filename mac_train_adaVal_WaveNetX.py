from torchvision import transforms, datasets
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.getnetwork import get_network
from models.networks_2d.WaveNetX import latest_ver
import argparse
import time
import os
import numpy as np
import random
from PIL import Image
from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from loss.loss_function import segmentation_loss
from dataload.dataset_2d import imagefolder_WaveNetX
from config.visdom_config.visual_visdom import visdom_initialization_WaveNetX, visualization_WaveNetX, visual_image_sup, vis_filter_bank_WaveNetX
from config.warmup_config.warmup import GradualWarmupScheduler
from config.train_test_config.train_test_config import print_train_loss_WaveNetX, print_val_loss_WaveNetX, print_train_eval_sup, print_val_eval_sup, save_val_best_sup_2d, draw_pred_sup, print_best_sup
from warnings import simplefilter
from torchsummary import summary

simplefilter(action='ignore', category=FutureWarning)

def init_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='GLAS', help='CREMI, GlaS, ISIC-2017')
    parser.add_argument('-b', '--batch_size', default=4, type=int) #default 16 but my ram can't take it
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-w', '--warm_up_duration', default=20, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--wavelet_type', default='haar', help='haar, db2, bior1.5, coif1, dmey')
    parser.add_argument('--alpha', default=[0, 0.4])
    parser.add_argument('--beta', default=[0.5, 0.8])
    parser.add_argument('-i', '--display_iter', default=1, type=int)
    parser.add_argument('-n', '--network', default='WaveNetX', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    parser.add_argument('-v', '--vis', default=True, help='need visualization or not')
    parser.add_argument('--visdom_port', default=16672)
    parser.add_argument('--show_args', default=True, help='show the arguments or not')
    parser.add_argument('--print_net', action='store_true', default=False, help='print the network or not')
    parser.add_argument('--bs_step_size', default=100, type=int, help='batch size step')
    parser.add_argument('--max_bs_steps', default=3, type=int, help='maximum number of batch size steps')
    parser.add_argument('--nfil', default=4, type=int, help='number of filters for smallest filter length in the DWT layer')
    parser.add_argument('--nfil_step', default=4, type=int, help='number of filters step size in the DWT layer')
    parser.add_argument('--flen', default=4, type=int, help='filter length in the DWT layer')
    parser.add_argument('--flen_step', default=4, type=int, help='filter length step size in the DWT layer')
    parser.add_argument('-l1', '--lambda1', default=0, type=float)
    parser.add_argument('-l2', '--lambda2', default=0, type=float)
    parser.add_argument('--fbl0', default=1, type=float, help='fb hi zero mean loss weight')
    parser.add_argument('--fbl1', default=1, type=float, help='fb lo orthnorm loss weight')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ver', default=latest_ver, type=int, help='version of WaveNetX')
    parser.add_argument('--big2small', action='store_true', default=False, help='batch size big to small')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu')

    args = parser.parse_args()

    args.ver = min(args.ver, latest_ver)
    args.network = args.network +'v' + str(args.ver)

    if args.show_args:
        print(args)

    # Set device to MPS or CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if args.use_cpu:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    init_seeds(args.seed)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    # Trained model save
    path_trained_models = cfg['PATH_TRAINED_MODEL'] + '/' + str(dataset_name)
    os.makedirs(path_trained_models, exist_ok=True)
    path_trained_models = path_trained_models + '/' + args.network + '-adaVal' +  '-l=' + str(args.lr) + \
            '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + \
                '-b=' + str(args.batch_size) + '-w=' + str(args.warm_up_duration) + \
                    '-nf=' + str(args.nfil) + '-fl=' + str(args.flen) + '-nfs=' + str(args.nfil_step) + '-fls=' + str(args.flen_step) + \
                        '-bs=' + str(args.bs_step_size) + '-mbs=' + str(args.max_bs_steps) + '-sd=' + str(args.seed) + \
                            '-fbl0=' + str(args.fbl0) + '-fbl1=' + str(args.fbl1) + args.big2small * '-b2s'
    os.makedirs(path_trained_models, exist_ok=True)

    # Segmentation results save
    path_seg_results = cfg['PATH_SEG_RESULT'] + '/' + str(dataset_name)
    os.makedirs(path_seg_results, exist_ok=True)
    path_seg_results = path_seg_results + '/' + args.network + '-adaVal' +  '-l=' + str(args.lr) + \
            '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + \
                '-b=' + str(args.batch_size) + '-w=' + str(args.warm_up_duration) + \
                    '-nf=' + str(args.nfil) + '-fl=' + str(args.flen) + '-nfs=' + str(args.nfil_step) + '-fls=' + str(args.flen_step) + \
                        '-bs=' + str(args.bs_step_size) + '-mbs=' + str(args.max_bs_steps) + '-sd=' + str(args.seed) + \
                            '-fbl0=' + str(args.fbl0) + '-fbl1=' + str(args.fbl1) + args.big2small * '-b2s'
    os.makedirs(path_seg_results, exist_ok=True)

    # Visualization initialization
    if args.vis:
        visdom_env = args.network + '-adaVal' + '-l=' + str(args.lr) + \
            '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + \
                '-b=' + str(args.batch_size) + '-nf=' + str(args.nfil) + '-fl=' + str(args.flen) + \
                    '-bs=' + str(args.bs_step_size) + '-mbs=' + str(args.max_bs_steps) + '-sd=' + str(args.seed) + \
                        '-fbl0=' + str(args.fbl0) + '-fbl1=' + str(args.fbl1) + args.big2small * '-b2s'
        visdom = visdom_initialization_WaveNetX(env=visdom_env, port=args.visdom_port)

    data_transforms = data_transform_2d(cfg['INPUT_SIZE'])
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    dataset_train_sup = imagefolder_WaveNetX(
        data_dir=cfg['PATH_DATASET'] + '/train_sup_100',
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=args.alpha,
        beta=args.beta,
        sup=True,
        rand_crop=False,
    )

    dataset_val = imagefolder_WaveNetX(
        data_dir=cfg['PATH_DATASET'] + '/val',
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=[0.2, 0.2],
        beta=[0.65, 0.65],
        sup=True,
        num_images=None,
        rand_crop=True,
    )

    # Print total number of images in the dataset
    print('=' * print_num)
    print('| Number of images in the training set: {:d}'.format(len(dataset_train_sup)).ljust(print_num_minus, ' '), '|')
    print('| Number of images in the validation set: {:d}'.format(len(dataset_val)).ljust(print_num_minus, ' '), '|')
    print('=' * print_num)

    dataloaders = dict()
    dataloaders['train_sup_0'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=True)
    dataloaders['train_sup_1'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*2, shuffle=True)
    dataloaders['train_sup_2'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*4, shuffle=True)
    dataloaders['train_sup_3'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*8, shuffle=True)

    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)#, num_workers=8)

    num_batches = {'train_sup_0': len(dataloaders['train_sup_0']), 'train_sup_1': len(dataloaders['train_sup_1']),
                    'train_sup_2': len(dataloaders['train_sup_2']), 'train_sup_3': len(dataloaders['train_sup_3']),
                    'val': len(dataloaders['val'])}

    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], 
                            nfil=args.nfil, flen=args.flen, # WaveNetXv0 and WaveNetXv1
                            nfil_start=args.nfil, flen_start=args.flen, flen_step=args.flen_step, nfil_step=args.nfil_step # WaveNetXv2
                            ).to(device)

    if args.print_net:
        summary(model1.cpu(), input_size=(cfg['IN_CHANNELS'], cfg['INPUT_SIZE'][0], cfg['INPUT_SIZE'][1]))

    criterion = segmentation_loss(args.loss, False).to(device)

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, 
                                               after_scheduler=exp_lr_scheduler1)

    since = time.time()
    count_iter = 0

    best_model = model1
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        model1.train()

        train_loss_sup_1 = 0.0
        train_loss_sup_2 = 0.0
        train_loss_sup_3 = 0.0
        train_loss = 0.0

        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0
        val_loss_sup_3 = 0.0

        fb_l0 = args.fbl0 * 1e-1
        fb_l1 = args.fbl1 * 1e-1

        fb_l0 *= (0.7+args.gamma) ** (epoch // args.step_size)
        fb_l1 *= (1.3+args.gamma) ** (epoch // args.step_size)

        bs_idx = min(epoch // args.bs_step_size, args.max_bs_steps)

        if args.big2small:
            bs_idx = args.max_bs_steps - bs_idx

        bs_str = 'train_sup_' + str(bs_idx)

        dataset_train_sup = iter(dataloaders[bs_str])

        for i in range(num_batches[bs_str]):

            loss_train = 0.0
            optimizer1.zero_grad()

            sup_index = next(dataset_train_sup)
            img_train_sup = Variable(sup_index['image'].to(device))
            mask_train_sup = Variable(sup_index['bin_mask'].to(device).long())
            loss_train_sup1, loss_train_sup2, loss_train_sup3 = 0, 0, 0

            if args.ver == 0:
                pred_train_sup1, pred_train_sup2, pred_train_sup3 = model1(img_train_sup)
                loss_train_sup1 = criterion(pred_train_sup1, mask_train_sup) + criterion(pred_train_sup2, mask_train_sup) + criterion(pred_train_sup3, mask_train_sup)
                train_loss_sup_1 += loss_train_sup1.item()
                loss_train_sup = loss_train_sup1
            else:
                pred_train_sup1 = model1(img_train_sup)
                loss_train_sup1 = criterion(pred_train_sup1, mask_train_sup)
                train_loss_sup_1 += loss_train_sup1.item()
                loss_train_sup = loss_train_sup1

            loss_train_sup2 = model1.dwt.get_fb_hi_0_mean_loss() * fb_l0
            train_loss_sup_2 += loss_train_sup2.item()
            loss_train_sup += loss_train_sup2

            if args.ver >= 2:
                loss_train_sup3 = model1.dwt.get_fb_lo_orthnorm_loss_v2() * fb_l1
                train_loss_sup_3 += loss_train_sup3.item()
                loss_train_sup += loss_train_sup3


            loss_train_sup.backward()

            optimizer1.step()

            loss_train += loss_train_sup
            train_loss += loss_train.item()

            if (count_iter % args.display_iter == 0) or args.vis:
                if i == 0:
                    score_list_train1 = pred_train_sup1
                    mask_list_train = mask_train_sup
                elif 0 < i <= num_batches[bs_str] / 64:
                    score_list_train1 = torch.cat((score_list_train1, pred_train_sup1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

        scheduler_warmup1.step()

        # Visualization and printing training statistics
        if (count_iter % args.display_iter == 0) or args.vis:
            print('=' * print_num)
            print(f'| Epoch {epoch + 1}/{args.num_epochs}'.ljust(print_num_minus, ' '), '|')
            train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss = print_train_loss_WaveNetX(
                train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss, num_batches, print_num, print_num_minus, num_batches[bs_str])
            train_eval_list1, train_m_jc1, train_m_dc1 = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus)
        
        # Validation loop
        with torch.no_grad():
            model1.eval()
            for i, data in enumerate(dataloaders['val']):

                inputs_val1 = Variable(data['image'].to(device))
                mask_val = Variable(data['bin_mask'].to(device).long())#Variable(data['mask'].to(device))
                name_val = data['ID']

                optimizer1.zero_grad()

                if args.ver == 0:
                    outputs_val1, outputs_val2, outputs_val3 = model1(inputs_val1)
                    val_loss_sup_1 += criterion(outputs_val1, mask_val).item()
                    val_loss_sup_2 += criterion(outputs_val2, mask_val).item()
                    val_loss_sup_3 += criterion(outputs_val3, mask_val).item()
                else:
                    outputs_val1 = model1(inputs_val1)
                    val_loss_sup_1 += criterion(outputs_val1, mask_val).item()

                if i == 0:
                    score_list_val1 = outputs_val1
                    mask_list_val = mask_val
                    name_list_val = name_val
                else:
                    score_list_val1 = torch.cat((score_list_val1, outputs_val1), dim=0)
                    mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                    name_list_val = np.append(name_list_val, name_val, axis=0)

            val_epoch_loss_sup1 = print_val_loss_WaveNetX(val_loss_sup_1, num_batches, print_num, print_num_minus)
            val_eval_list1, val_m_jc1, val_m_dc1 = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val1, mask_list_val, print_num_minus)
            best_val_eval_list = save_val_best_sup_2d(cfg['NUM_CLASSES'], best_val_eval_list, model1, score_list_val1, name_list_val, val_eval_list1, 
                                                      path_trained_models, path_seg_results, cfg['PALETTE'], args.network)
            if args.vis:                
                draw_img = draw_pred_sup(cfg['NUM_CLASSES'], mask_train_sup, mask_val, pred_train_sup1, outputs_val1, train_eval_list1, val_eval_list1)
                visualization_WaveNetX(visdom, epoch + 1, train_epoch_loss, train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, 
                                       train_m_jc1, train_m_dc1, val_epoch_loss_sup1, val_m_jc1, val_m_dc1)
                visual_image_sup(visdom, draw_img[0], draw_img[1], draw_img[2], draw_img[3])
                for f_idx in range(model1.dwt.nfil):
                    vis_filter_bank_WaveNetX(visdom, fb_2d_list=model1.dwt.get_fb_2d_list(for_vis=True), fil_idx=f_idx, figure_name='2D Filter #{}'.format(f_idx))
            print('-' * print_num)
            print('| Epoch {:d}/{:d} took {:.4f}s with batch size {:d}'.format(epoch + 1, args.num_epochs, (time.time() - begin_time), dataloaders[bs_str].batch_size).ljust(print_num_minus, ' '), '|')
    
    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)

    print('=' * print_num)
    print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    print_best_sup(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
    print('=' * print_num)