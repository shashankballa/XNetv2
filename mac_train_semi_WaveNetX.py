from torchvision import transforms, datasets
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.getnetwork import get_network
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
from config.visdom_config.visual_visdom import visdom_initialization_XNetv2, visualization_XNetv2, visual_image_sup, vis_filter_bank_WaveNetX
from config.warmup_config.warmup import GradualWarmupScheduler
from config.train_test_config.train_test_config import print_train_loss_XNetv2, print_val_loss_XNetv2, print_train_eval_sup, print_val_eval_sup, save_val_best_sup_2d, draw_pred_sup, print_best_sup
from warnings import simplefilter
from torchsummary import summary

simplefilter(action='ignore', category=FutureWarning)

def init_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)

def compute_p_norm_conv_kernel(params, p=2):
    """
    Compute the L1 or L2 norm of a 4D convolutional kernel across the last two dimensions
    and sum the results for all filters.

    Args:
        params (torch.Tensor): 4D tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
        p (int): Norm type (1 for L1 norm, 2 for L2 norm)

    Returns:
        torch.Tensor: Scalar p-norm value summed across all filters.
    """
    if params.dim() != 4:
        raise ValueError("Expected a 4D tensor, got tensor with shape {}".format(params.shape))

    if p == 1:
        # Compute the absolute sum over the last two dimensions (kernel_height, kernel_width)
        abs_sum = torch.sum(torch.abs(params), dim=(2, 3))
        total_p_norm = torch.max(abs_sum)
    elif p == 2:
        # Compute the squared sum over the last two dimensions (kernel_height, kernel_width)
        squared_sum = torch.sum(params ** 2, dim=(2, 3))
        # Compute the square root to get the L2 norm for each kernel
        l2_norms = torch.sqrt(squared_sum)
        # Sum the L2 norms across all input and output channels
        total_p_norm = torch.sum(l2_norms)
    else:
        raise ValueError("Only p=1 (L1 norm) and p=2 (L2 norm) are supported.")

    return total_p_norm

def get_parms_lp_norm(params, p):
    '''
    GIVES NAN LOSS! DO NOT USE
    '''
    lp_loss = torch.tensor(0.0, device=params[0].device)
    for param in params:
        if param.dim() == 4:  # Conv layer weights: (out_channels, in_channels, kernel_height, kernel_width)
            # Compute norm across filter dimensions (last two dims)
            lp_loss += compute_p_norm_conv_kernel(param, p)
        elif param.dim() == 2:  # Fully connected layer weights
            lp_loss += compute_p_norm_conv_kernel(param.unsqueeze(0).unsqueeze(0), p)
        elif param.dim() == 1:  # Bias terms or other parameters
            lp_loss += compute_p_norm_conv_kernel(param.unsqueeze(0).unsqueeze(0).unsqueeze(0), p)
    return lp_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='GLAS', help='CREMI, GlaS, ISIC-2017')
    parser.add_argument('--sup_mark', default='20')
    parser.add_argument('--unsup_mark', default='80')
    parser.add_argument('-b', '--batch_size', default=4, type=int) #default 16 but my ram can't take it
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=0.5, type=float)
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
    parser.add_argument('--print_net', action='store_true', default=False,
                        help='print the network or not')
    parser.add_argument('--bs_step', default=100, type=int, help='batch size step')
    parser.add_argument('--nfil', default=16, type=int, help='number of filters in the DWT layer')
    parser.add_argument('--flen', default=8, type=int, help='filter length in the DWT layer')
    parser.add_argument('-l1', '--lambda1', default=0, type=float)
    parser.add_argument('-l2', '--lambda2', default=0, type=float)
    args = parser.parse_args()

    if args.show_args:
        print(args)

    skip_unsup = False
    if args.sup_mark == '100':
        skip_unsup = True
        args.unsup_weight = 0
        args.unsup_mark = '0'
        print('Skipping unsupervised training')

    l1_lambda = args.lambda1 * 1e-10
    l2_lambda = args.lambda2 * 1e-10

    # Set device to MPS or CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device}")

    init_seeds(42)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    # Trained model save
    path_trained_models = cfg['PATH_TRAINED_MODEL'] + '/' + str(dataset_name)
    os.makedirs(path_trained_models, exist_ok=True)
    path_trained_models = path_trained_models + '/' + args.network + '-l=' + str(args.lr) + \
        '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + \
            '-b=' + str(args.batch_size) + '-uw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + \
                '-sup=' + str(args.sup_mark) + '-usup=' + str(args.unsup_mark) + '-nf=' + str(args.nfil) + '-fl=' + str(args.flen) + \
                    '-l1=' + str(args.lambda1) + '-l2=' + str(args.lambda2) + '-bs=' + str(args.bs_step)
    os.makedirs(path_trained_models, exist_ok=True)

    # Segmentation results save
    path_seg_results = cfg['PATH_SEG_RESULT'] + '/' + str(dataset_name)
    os.makedirs(path_seg_results, exist_ok=True)
    path_seg_results = path_seg_results + '/' + args.network + '-l=' + str(args.lr) + \
        '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + \
            '-b=' + str(args.batch_size) + '-uw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + \
                '-sup=' + str(args.sup_mark) + '-usup=' + str(args.unsup_mark) + '-nf=' + str(args.nfil) + '-fl=' + str(args.flen) + \
                    '-l1=' + str(args.lambda1) + '-l2=' + str(args.lambda2) + '-bs=' + str(args.bs_step)
    os.makedirs(path_seg_results, exist_ok=True)

    # Visualization initialization
    if args.vis:
        visdom_env = str('semisup-' + str(dataset_name) + '-' + args.network + '-l=' + str(args.lr) + \
            '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + \
                '-b=' + str(args.batch_size) + '-uw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + \
                    '-sup=' + str(args.sup_mark) + '-usup=' + str(args.unsup_mark)) + '-nf=' + str(args.nfil) + '-fl=' + str(args.flen) + \
                        '-l1=' + str(args.lambda1) + '-l2=' + str(args.lambda2) + '-bs=' + str(args.bs_step)
        visdom = visdom_initialization_XNetv2(env=visdom_env, port=args.visdom_port)

    data_transforms = data_transform_2d(cfg['INPUT_SIZE'])
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    dataset_train_unsup = None
    num_images_unsup = None
    if not skip_unsup:
        dataset_train_unsup = imagefolder_WaveNetX(
            data_dir=cfg['PATH_DATASET'] + '/train_unsup_' + args.unsup_mark,
            data_transform_1=data_transforms['train'],
            data_normalize_1=data_normalize,
            wavelet_type=args.wavelet_type,
            alpha=args.alpha,
            beta=args.beta,
            sup=False,
            num_images=None,
        )
        num_images_unsup = len(dataset_train_unsup)

    dataset_train_sup = imagefolder_WaveNetX(
        data_dir=cfg['PATH_DATASET'] + '/train_sup_' + args.sup_mark,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=args.alpha,
        beta=args.beta,
        sup=True,
        num_images=num_images_unsup,
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

    dataloaders = dict()
    dataloaders['train_sup_0'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=True)
    dataloaders['train_sup_1'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*2, shuffle=True)
    dataloaders['train_sup_2'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*4, shuffle=True)
    dataloaders['train_sup_3'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*8, shuffle=True)

    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)#, num_workers=8)
    if not skip_unsup:
        dataloaders['train_unsup'] = DataLoader(dataset_train_unsup, batch_size=args.batch_size, shuffle=True)#, num_workers=8)

    num_batches = {'train_sup_0': len(dataloaders['train_sup_0']), 'train_sup_1': len(dataloaders['train_sup_1']),
                    'train_sup_2': len(dataloaders['train_sup_2']), 'train_sup_3': len(dataloaders['train_sup_3']),
                    'val': len(dataloaders['val'])}
    if not skip_unsup:
        num_batches['train_unsup'] = len(dataloaders['train_unsup'])

    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], nfil=args.nfil, flen=args.flen).to(device)

    if args.print_net:
        summary(model1.cpu(), input_size=(cfg['IN_CHANNELS'], cfg['INPUT_SIZE'][0], cfg['INPUT_SIZE'][1]))

    criterion = segmentation_loss(args.loss, False).to(device)

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler1)

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
        train_loss_unsup = 0.0
        train_loss = 0.0

        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0
        val_loss_sup_3 = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs if not skip_unsup else 0

        bs_step = 'train_sup_' + str(min(epoch // args.bs_step, 3))

        dataset_train_sup = iter(dataloaders[bs_step])

        if not skip_unsup:
            dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches[bs_step]):

            loss_train = 0.0
            optimizer1.zero_grad()

            if not skip_unsup:
                unsup_index = next(dataset_train_unsup)
                img_train_unsup1 = Variable(unsup_index['image'].to(device))

                pred_train_unsup1, pred_train_unsup2, pred_train_unsup3 = model1(img_train_unsup1)

                max_train1 = torch.max(pred_train_unsup1, dim=1)[1].long()
                max_train2 = torch.max(pred_train_unsup2, dim=1)[1].long()
                max_train3 = torch.max(pred_train_unsup3, dim=1)[1].long()
                loss_train_unsup = criterion(pred_train_unsup1, max_train2) + criterion(pred_train_unsup2, max_train1) + \
                                criterion(pred_train_unsup1, max_train3) + criterion(pred_train_unsup3, max_train1)

                loss_train_unsup = loss_train_unsup * unsup_weight

                # L2 regularization for M, L, and Fusion network parameters
                if l2_lambda > 0:
                    M_params = model1.get_M_net_params()
                    L_params = model1.get_L_net_params()
                    Fusion_params = model1.get_fusion_params()
                    l2_loss = get_parms_lp_norm(M_params, 2) + get_parms_lp_norm(L_params, 2) + get_parms_lp_norm(Fusion_params, 2)
                    loss_train_unsup += l2_lambda * l2_loss

                # L1 regularization for H network parameters
                if l1_lambda > 0:
                    H_params = model1.get_H_net_params()
                    l1_loss = get_parms_lp_norm(H_params, 1)
                    loss_train_unsup += l1_lambda * l1_loss

                loss_train_unsup.backward(retain_graph=True)
                loss_train += loss_train_unsup
                train_loss_unsup += loss_train_unsup.item()

            sup_index = next(dataset_train_sup)
            img_train_sup1 = Variable(sup_index['image'].to(device))
            mask_train_sup = Variable(sup_index['mask'].to(device))
            bin_mask_train_sup = Variable(sup_index['bin_mask'].to(device).long())

            pred_train_sup1, pred_train_sup2, pred_train_sup3 = model1(img_train_sup1)

            if (count_iter % args.display_iter == 0) or args.vis:
                if i == 0:
                    score_list_train1 = pred_train_sup1
                    mask_list_train = mask_train_sup
                elif 0 < i <= num_batches[bs_step] / 64:
                    score_list_train1 = torch.cat((score_list_train1, pred_train_sup1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            loss_train_sup1 = criterion(pred_train_sup1, bin_mask_train_sup) + criterion(pred_train_sup2, bin_mask_train_sup) + criterion(pred_train_sup3, bin_mask_train_sup)
            loss_train_sup = loss_train_sup1

            loss_train_sup2 = 0
            if l2_lambda > 0:
                M_params = model1.get_M_net_params()
                L_params = model1.get_L_net_params()
                Fusion_params = model1.get_fusion_params()
                l2_loss = get_parms_lp_norm(M_params, 2) + get_parms_lp_norm(L_params, 2) + get_parms_lp_norm(Fusion_params, 2)
                loss_train_sup2 = l2_lambda * l2_loss
                loss_train_sup += loss_train_sup2
                train_loss_sup_2 += loss_train_sup2.item()

            loss_train_sup3 = 0
            if l1_lambda > 0:
                H_params = model1.get_H_net_params()
                l1_loss = get_parms_lp_norm(H_params, 1)
                loss_train_sup3 += l1_lambda * l1_loss
                loss_train_sup += loss_train_sup3
                train_loss_sup_3 += loss_train_sup3.item()

            loss_train_sup.backward()

            optimizer1.step()

            loss_train += loss_train_sup
            train_loss += loss_train.item()

        scheduler_warmup1.step()

        # Visualization and printing training statistics
        if (count_iter % args.display_iter == 0) or args.vis:
            print('=' * print_num)
            print(f'| Epoch {epoch + 1}/{args.num_epochs}'.ljust(print_num_minus, ' ') + '|')
            train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss_unsup, train_epoch_loss = print_train_loss_XNetv2(
                train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus, num_batches[bs_step]
            )
            train_eval_list1, train_m_jc1 = print_train_eval_sup(
                cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus
            )
        
        # Validation loop
        with torch.no_grad():
            model1.eval()
            for i, data in enumerate(dataloaders['val']):

                inputs_val1 = Variable(data['image'].to(device))
                mask_val = Variable(data['mask'].to(device))
                bin_mask_val = Variable(data['bin_mask'].to(device).long())
                name_val = data['ID']

                optimizer1.zero_grad()
                outputs_val1, outputs_val2, outputs_val3 = model1(inputs_val1)

                if i == 0:
                    score_list_val1 = outputs_val1
                    mask_list_val = mask_val
                    name_list_val = name_val
                else:
                    score_list_val1 = torch.cat((score_list_val1, outputs_val1), dim=0)
                    mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                    name_list_val = np.append(name_list_val, name_val, axis=0)

                loss_val_sup1 = criterion(outputs_val1, bin_mask_val)
                loss_val_sup2 = criterion(outputs_val2, bin_mask_val)
                loss_val_sup3 = criterion(outputs_val3, bin_mask_val)
                val_loss_sup_1 += loss_val_sup1.item()
                val_loss_sup_2 += loss_val_sup2.item()
                val_loss_sup_3 += loss_val_sup3.item()

            val_epoch_loss_sup1, val_epoch_loss_sup2, val_epoch_loss_sup3 = print_val_loss_XNetv2(val_loss_sup_1, val_loss_sup_2, val_loss_sup_3, num_batches, print_num, print_num_minus)
            val_eval_list1, val_m_jc1 = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val1, mask_list_val, print_num_minus)
            best_val_eval_list = save_val_best_sup_2d(cfg['NUM_CLASSES'], best_val_eval_list, model1, score_list_val1, name_list_val, val_eval_list1, path_trained_models, path_seg_results, cfg['PALETTE'], args.network)
            if args.vis:                
                draw_img = draw_pred_sup(cfg['NUM_CLASSES'], mask_train_sup, mask_val, pred_train_sup1, outputs_val1, train_eval_list1, val_eval_list1)
                visualization_XNetv2(visdom, epoch + 1, train_epoch_loss, train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss_unsup, train_m_jc1, val_epoch_loss_sup1, val_epoch_loss_sup2, val_epoch_loss_sup3, val_m_jc1)
                visual_image_sup(visdom, draw_img[0], draw_img[1], draw_img[2], draw_img[3])
                for f_idx in range(model1.dwt.fb_lo.shape[0]):
                    vis_filter_bank_WaveNetX(visdom, fil_lo=model1.dwt.fb_lo[f_idx], figure_name='Filter #{}'.format(f_idx))
            print('-' * print_num)
            print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(
                    print_num_minus, ' '), '|')
    
    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)

    print('=' * print_num)
    print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    print_best_sup(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
    print('=' * print_num)