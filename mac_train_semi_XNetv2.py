from torchvision import transforms, datasets
import torch
import torch.nn as nn
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
from dataload.dataset_2d import imagefolder_XNetv2
from config.visdom_config.visual_visdom import visdom_initialization_XNetv2, visualization_XNetv2, visual_image_sup
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='GLAS', help='CREMI, GlaS, ISIC-2017')
    parser.add_argument('--sup_mark', default='20')
    parser.add_argument('--unsup_mark', default='80')
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=0.5, type=float)
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--wavelet_type', default='haar', help='haar, db2, bior1.5, coif1, dmey')
    parser.add_argument('--alpha', default=[0, 0.4])
    parser.add_argument('--beta', default=[0.5, 0.8])
    parser.add_argument('-i', '--display_iter', default=5, type=int)
    parser.add_argument('-n', '--network', default='XNetv2', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    parser.add_argument('-v', '--vis', default=True, help='need visualization or not')
    parser.add_argument('--visdom_port', default=16672)
    parser.add_argument('--show_args', default=True, help='show the arguments or not')
    parser.add_argument('--print_net', action='store_true', default=False,
                        help='print the network or not')
    args = parser.parse_args()

    if args.show_args:
        print(args)

    # Set device to MPS or CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
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
    path_trained_models = path_trained_models + '/' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-uw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark) + '-' + str(args.unsup_mark)
    os.makedirs(path_trained_models, exist_ok=True)

    # Segmentation results save
    path_seg_results = cfg['PATH_SEG_RESULT'] + '/' + str(dataset_name)
    os.makedirs(path_seg_results, exist_ok=True)
    path_seg_results = path_seg_results + '/' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-uw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark) + '-' + str(args.unsup_mark)
    os.makedirs(path_seg_results, exist_ok=True)

    # Visualization initialization
    if args.vis:
        visdom_env = str('Semi-XNetv2-' + str(dataset_name) + '-' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-uw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark) + '-' + str(args.unsup_mark))
        visdom = visdom_initialization_XNetv2(env=visdom_env, port=args.visdom_port)

    data_transforms = data_transform_2d(cfg['INPUT_SIZE'])
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    dataset_train_unsup = imagefolder_XNetv2(
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

    dataset_train_sup = imagefolder_XNetv2(
        data_dir=cfg['PATH_DATASET'] + '/train_sup_' + args.sup_mark,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=args.alpha,
        beta=args.beta,
        sup=True,
        num_images=num_images_unsup,
    )
    dataset_val = imagefolder_XNetv2(
        data_dir=cfg['PATH_DATASET'] + '/val',
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=[0.2, 0.2],
        beta=[0.65, 0.65],
        sup=True,
        num_images=None,
    )

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=True)#, num_workers=8)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup, batch_size=args.batch_size, shuffle=True)#, num_workers=8)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)#, num_workers=8)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).to(device)

    # if args.print_net:
    #     summary(model1.cpu(), input_size=(cfg['IN_CHANNELS'], cfg['INPUT_SIZE'][0], cfg['INPUT_SIZE'][1]))

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

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs

        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):

            unsup_index = next(dataset_train_unsup)
            img_train_unsup1 = Variable(unsup_index['image'].to(device))
            img_train_unsup2 = Variable(unsup_index['L'].to(device))
            img_train_unsup3 = Variable(unsup_index['H'].to(device))
            optimizer1.zero_grad()

            if args.network == 'XNetv2':
                pred_train_unsup1, pred_train_unsup2, pred_train_unsup3 = model1(img_train_unsup1, img_train_unsup2, img_train_unsup3)
            elif args.network == 'MLWNet' or args.network == 'mlwnet':
                pred_train_unsup1, pred_train_unsup2, pred_train_unsup3 = model1(img_train_unsup1)

            max_train1 = torch.max(pred_train_unsup1, dim=1)[1].long()
            max_train2 = torch.max(pred_train_unsup2, dim=1)[1].long()
            max_train3 = torch.max(pred_train_unsup3, dim=1)[1].long()
            loss_train_unsup = criterion(pred_train_unsup1, max_train2) + criterion(pred_train_unsup2, max_train1) + \
                               criterion(pred_train_unsup1, max_train3) + criterion(pred_train_unsup3, max_train1)
            
            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train_unsup.backward(retain_graph=True)

            sup_index = next(dataset_train_sup)
            img_train_sup1 = Variable(sup_index['image'].to(device))
            img_train_sup2 = Variable(sup_index['L'].to(device))
            img_train_sup3 = Variable(sup_index['H'].to(device))
            mask_train_sup = Variable(sup_index['mask'].to(device))
            bin_mask_train_sup = Variable(sup_index['bin_mask'].to(device).long())

            pred_train_sup1, pred_train_sup2, pred_train_sup3 = model1(img_train_sup1, img_train_sup2, img_train_sup3)

            if (count_iter % args.display_iter == 0) or args.vis:
                if i == 0:
                    score_list_train1 = pred_train_sup1
                    mask_list_train = mask_train_sup
                elif 0 < i <= num_batches['train_sup'] / 64:
                    score_list_train1 = torch.cat((score_list_train1, pred_train_sup1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            loss_train_sup1 = criterion(pred_train_sup1, bin_mask_train_sup)
            loss_train_sup2 = criterion(pred_train_sup2, bin_mask_train_sup)
            loss_train_sup3 = criterion(pred_train_sup3, bin_mask_train_sup)
            loss_train_sup = loss_train_sup1 + loss_train_sup2 + loss_train_sup3
            loss_train_sup.backward()

            optimizer1.step()

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_sup_3 += loss_train_sup3.item()
            train_loss += loss_train.item()

        scheduler_warmup1.step()

        # Visualization and printing training statistics
        if (count_iter % args.display_iter == 0) or args.vis:
            print('=' * print_num)
            print(f'| Epoch {epoch + 1}/{args.num_epochs}'.ljust(print_num_minus, ' ') + '|')
            train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss_unsup, train_epoch_loss = print_train_loss_XNetv2(
                train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus
            )
            train_eval_list1, train_m_jc1 = print_train_eval_sup(
                cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus
            )
        
        # Validation loop
        with torch.no_grad():
            model1.eval()
            for i, data in enumerate(dataloaders['val']):

                inputs_val1 = Variable(data['image'].to(device))
                inputs_val2 = Variable(data['L'].to(device))
                inputs_val3 = Variable(data['H'].to(device))
                mask_val = Variable(data['mask'].to(device))
                bin_mask_val = Variable(data['bin_mask'].to(device).long())
                name_val = data['ID']

                optimizer1.zero_grad()
                outputs_val1, outputs_val2, outputs_val3 = model1(inputs_val1, inputs_val2, inputs_val3)

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
            best_val_eval_list = save_val_best_sup_2d(cfg['NUM_CLASSES'], best_val_eval_list, model1, score_list_val1, name_list_val, val_eval_list1, path_trained_models, path_seg_results, cfg['PALETTE'], 'XNetv2')
            if args.vis:                
                draw_img = draw_pred_sup(cfg['NUM_CLASSES'], mask_train_sup, mask_val, pred_train_sup1, outputs_val1, train_eval_list1, val_eval_list1)
                visualization_XNetv2(visdom, epoch + 1, train_epoch_loss, train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss_unsup, train_m_jc1, val_epoch_loss_sup1, val_epoch_loss_sup2, val_epoch_loss_sup3, val_m_jc1)
                visual_image_sup(visdom, draw_img[0], draw_img[1], draw_img[2], draw_img[3])
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