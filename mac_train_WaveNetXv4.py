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
from config.visdom_config.visual_visdom import vis_init_WaveNetX4, vis_WaveNetX4, vis_image_WaveNetX4, vis_filter_bank_WaveNetX
from config.warmup_config.warmup import GradualWarmupScheduler
from config.train_test_config.train_test_config import print_train_loss_WaveNetX4, print_val_loss_WaveNetX4, print_train_eval_sup, print_val_scores_WaveNetX4, save_val_sup_2d_best_model, draw_pred_sup_WaveNetX4, print_best_sup, save_test_2d, print_test_eval
from warnings import simplefilter
from torchsummary import summary
import copy

simplefilter(action='ignore', category=FutureWarning)

def init_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)

if __name__ == '__main__':

    # bash scripts/run_py.sh mac_train_WaveNetXv4.py -l 2.0 -e 2000 -s 80 -g 0.7 -b 8 -w 40 --nfil 9 --flen 6 --nfil_step -1 --flen_step 2 --bs_step 100 --max_bs_steps 2 --seed 666666 --fbl0 0.01 --fbl1 0.001 -b2s -ub 0 -nfl 8
    # bash scripts/run_py.sh mac_train_WaveNetXv4.py -l 2.0 -e 2000 -s 80 -g 0.7 -b 4 -w 40 --nfil 7 --flen 6 --nfil_step -1 --flen_step 2 --bs_step 100 --max_bs_steps 3 --seed 666666 --fbl0 0.1 --fbl1 0.01 -b2s -ub 80

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--display_iter', default=1, type=int)
    parser.add_argument('-v', '--vis', default=True, help='need visualization or not')
    parser.add_argument('--visdom_port', default=16662)
    parser.add_argument('--show_args', default=True, help='show the arguments or not')
    parser.add_argument('--print_net', action='store_true', default=False, help='print the network or not')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu')

    parser.add_argument('--dataset_name',       default='GLAS', help='CREMI, GlaS, ISIC-2017')
    parser.add_argument('--loss',               default='dice')
    parser.add_argument('-e', '--num_epochs',   default=200, type=int)
    parser.add_argument('-l', '--lr',           default=0.5, type=float)
    parser.add_argument('-g', '--gamma',        default=0.5, type=float)
    parser.add_argument('-s', '--step_size',    default=50, type=int)
    parser.add_argument('-w', '--warm_up_duration', default=20, type=int)

    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')

    parser.add_argument('-b', '--batch_size',   default=4, type=int)
    parser.add_argument('--bs_step_size', default=100, type=int, help='batch size step')
    parser.add_argument('--max_bs_steps', default=3, type=int, help='maximum number of batch size steps')
    parser.add_argument('-b2s', '--big2small', action='store_true', default=False, help='batch size big to small')

    parser.add_argument('-ub', '--use_best', default=0, type=int, help='number of epochs to bootstrap')
    parser.add_argument('-t4v','--test4val', action='store_true', default=False, help='use test set for validation')
    # parser.add_argument('--threshold', default=0.5, type=float)

    parser.add_argument('-nfl', '--nflens', default=4, type=int, help='number of filter lengths')
    parser.add_argument('--nfil', default=4, type=int, help='number of filters for smallest filter length in the DWT layer')
    parser.add_argument('--nfil_step', default=4, type=int, help='number of filters step size in the DWT layer')
    parser.add_argument('--symnf', action='store_true', default=False, help='use symmetric number of filters in the DWT layer: increase and decrease')
    parser.add_argument('--flen', default=4, type=int, help='filter length in the DWT layer')
    parser.add_argument('--flen_step', default=4, type=int, help='filter length step size in the DWT layer')
    parser.add_argument('--fbl0', default=1, type=float, help='fb hi zero mean loss weight')
    parser.add_argument('--fbl1', default=1, type=float, help='fb lo orthnorm loss weight')
    parser.add_argument('--fbl1v2_nr', default=8, type=float, help='Number of rows for fb lo orthnorm loss weight')

    args = parser.parse_args()

    _ver = 2
    _net = 'WaveNetXv' + str(_ver)

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
                
    model_prm = _net  + '-nfl=' + str(args.nflens) + '-fl=' + str(args.flen) + '-fls=' + str(args.flen_step) + '-nf=' + str(args.nfil) + '-nfs=' + str(args.nfil_step) + \
            '-fbl0=' + str(args.fbl0) + '-fbl1=' + str(args.fbl1) + '-fbl1v2_nr=' + str(args.fbl1v2_nr) + '-symnf' * args.symnf
    
    train_prm_str = '-l=' + str(args.lr) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-w=' + str(args.warm_up_duration) + \
            '-e=' + str(args.num_epochs) + '-b=' + str(args.batch_size) + '-bs=' + str(args.bs_step_size) + '-mbs=' + str(args.max_bs_steps) + \
                '-b2s' * args.big2small + '-t4v' * args.test4val + '-ub=' + str(args.use_best) + '-sd=' + str(args.seed) 
    
    hyper_params_str = model_prm + train_prm_str
                                

    # Trained model save
    path_trained_models = cfg['PATH_TRAINED_MODEL'] + '/' + str(dataset_name)
    os.makedirs(path_trained_models, exist_ok=True)
    path_trained_models = path_trained_models + '/' + hyper_params_str
    os.makedirs(path_trained_models, exist_ok=True)

    # Save the args in a text file in the trained model directory
    with open(path_trained_models + '/args_' + hyper_params_str + '.txt', 'w') as f:
        f.write(str(args))


    # Segmentation results save
    path_seg_results = cfg['PATH_SEG_RESULT'] + '/' + str(dataset_name)
    os.makedirs(path_seg_results, exist_ok=True)
    path_seg_results = path_seg_results + '/' + hyper_params_str
    os.makedirs(path_seg_results, exist_ok=True)

    # Visualization initialization
    if args.vis:
        visdom_env = hyper_params_str
        visdom = vis_init_WaveNetX4(env=visdom_env, port=args.visdom_port)

    data_transforms = data_transform_2d(cfg['INPUT_SIZE'])
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    dataset_train_sup = imagefolder_WaveNetX(
        data_dir=cfg['PATH_DATASET'] + '/train_sup_100',
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=True,
        rand_crop=False,
    )

    dataset_val = imagefolder_WaveNetX(
        data_dir=cfg['PATH_DATASET'] + '/val',
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None,
        rand_crop=True,
    )

    # Dataset
    dataset_test = imagefolder_WaveNetX(
        data_dir=cfg['PATH_DATASET'] + '/test',
        data_transform_1=data_transforms['test'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None
    )

    # Print total number of images in the dataset
    print('=' * print_num)
    print('| Number of images in the training set: {:d}'.format(len(dataset_train_sup)).ljust(print_num_minus, ' '), '|')
    print('| Number of images in the validation set: {:d}'.format(len(dataset_val)).ljust(print_num_minus, ' '), '|')
    print('=' * print_num)

    dataloaders = dict()
    dataloaders['train_sup_0'] = DataLoader(dataset_train_sup, batch_size=args.batch_size   , shuffle=True)
    if args.max_bs_steps > 0:
        dataloaders['train_sup_1'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*2 , shuffle=True)
        if args.max_bs_steps > 1:
            dataloaders['train_sup_2'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*4 , shuffle=True)
            if args.max_bs_steps > 2:
                dataloaders['train_sup_3'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*8 , shuffle=True)
                if args.max_bs_steps > 3:
                    dataloaders['train_sup_4'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*16, shuffle=True)
                    if args.max_bs_steps > 4:
                        dataloaders['train_sup_5'] = DataLoader(dataset_train_sup, batch_size=args.batch_size*32, shuffle=True)
                        if args.max_bs_steps > 5:
                            raise ValueError('Maximum number of batch size steps is 5')

    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    dataloaders['test'] = DataLoader(dataset_test, batch_size=32, shuffle=False)
    

    num_batches = {'train_sup_0': len(dataloaders['train_sup_0']), 'val': len(dataloaders['val']), 'test': len(dataloaders['test'])}
    if args.max_bs_steps > 0:
        num_batches['train_sup_1'] = len(dataloaders['train_sup_1'])
        if args.max_bs_steps > 1:
            num_batches['train_sup_2'] = len(dataloaders['train_sup_2'])
            if args.max_bs_steps > 2:
                num_batches['train_sup_3'] = len(dataloaders['train_sup_3'])
                if args.max_bs_steps > 3:
                    num_batches['train_sup_4'] = len(dataloaders['train_sup_4'])
                    if args.max_bs_steps > 4:
                        num_batches['train_sup_5'] = len(dataloaders['train_sup_5'])

    model1 = get_network(_net, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], 
                            nfil=args.nfil, flen=args.flen, # WaveNetXv0 and WaveNetXv1
                            nfil_start=args.nfil, flen_start=args.flen, flen_step=args.flen_step, nfil_step=args.nfil_step, # WaveNetXv2
                            nflens=args.nflens, ver=_ver, fbl1_nrows=args.fbl1v2_nr, symm_nfils=args.symnf
                            ).to(device)

    if args.print_net:
        summary(model1.cpu(), input_size=(cfg['IN_CHANNELS'], cfg['INPUT_SIZE'][0], cfg['INPUT_SIZE'][1]))

    best_model_weights = copy.deepcopy(model1.state_dict())
    best_model = get_network(_net, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], 
                            nfil=args.nfil, flen=args.flen, # WaveNetXv0 and WaveNetXv1
                            nfil_start=args.nfil, flen_start=args.flen, flen_step=args.flen_step, nfil_step=args.nfil_step, # WaveNetXv2
                            nflens=args.nflens, ver=_ver, fbl1_nrows=args.fbl1v2_nr, symm_nfils=args.symnf
                            ).to(device)
    best_model.load_state_dict(best_model_weights)

    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(4)]

    criterion = segmentation_loss(args.loss, False).to(device)

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, 
                                               after_scheduler=exp_lr_scheduler1)

    since = time.time()
    count_iter = 0

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
        val_loss_sup_best = 0.0
        val_loss_sup_3 = 0.0

        fb_l0 = args.fbl0 * 1e-1
        fb_l1 = args.fbl1 * 1e-1

        # fb_l0 *= (0.6+args.gamma) ** (epoch // args.step_size)
        # fb_l1 *= (1.3+args.gamma) ** (epoch // args.step_size)

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


            pred_train_sup1 = model1(img_train_sup)
            loss_train_sup1 = criterion(pred_train_sup1, mask_train_sup)
            train_loss_sup_1 += loss_train_sup1.item()
            loss_train_sup = loss_train_sup1

            loss_train_sup2 = model1.dwt.get_fb_hi_0_mean_loss() * fb_l0
            train_loss_sup_2 += loss_train_sup2.item()
            loss_train_sup += loss_train_sup2

            loss_train_sup3 = model1.dwt.get_fb_lo_orthnorm_loss() * fb_l1
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
            train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss = print_train_loss_WaveNetX4(
                train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss, num_batches, print_num, print_num_minus, num_batches[bs_str])
            train_eval_list1, train_m_jc1, train_m_dc1 = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus)
        
        # Validation loop
        with torch.no_grad():
            model1.eval()
            best_model.eval()
            val_str = 'val'
            if args.test4val:
                val_str = 'test'
            for i, data in enumerate(dataloaders[val_str]):

                inputs_val1 = Variable(data['image'].to(device))
                mask_val = Variable(data['bin_mask'].to(device).long())#Variable(data['mask'].to(device))
                name_val = data['ID']

                optimizer1.zero_grad()

                pred_val = model1(inputs_val1)
                val_loss_sup_1 += criterion(pred_val, mask_val).item()

                pred_val_best = best_model(inputs_val1)
                val_loss_sup_best += criterion(pred_val_best, mask_val).item()

                if i == 0:
                    score_list_val1 = pred_val
                    score_list_val_best = pred_val_best
                    mask_list_val = mask_val
                    name_list_val = name_val
                else:
                    score_list_val1 = torch.cat((score_list_val1, pred_val), dim=0)
                    score_list_val_best = torch.cat((score_list_val_best, pred_val_best), dim=0)
                    mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                    name_list_val = np.append(name_list_val, name_val, axis=0)

            val_epoch_loss_sup1, val_epoch_loss_sup_best = print_val_loss_WaveNetX4(val_loss_sup_1, val_loss_sup_best, num_batches, print_num, print_num_minus)
            val_eval_list1, val_eval_list_best, val_m_jc1, val_m_dc1, best_jc, best_dc = print_val_scores_WaveNetX4(cfg['NUM_CLASSES'], score_list_val1, score_list_val_best,
                                                                                                mask_list_val, print_num_minus)

            best_val_eval_list, best_model = save_val_sup_2d_best_model(cfg['NUM_CLASSES'], best_val_eval_list, model1, best_model, score_list_val1, name_list_val, val_eval_list1, 
                                                    path_trained_models, path_seg_results, cfg['PALETTE'], hyper_params_str)
            
            if (args.use_best > 0) and (epoch % args.use_best == 0) and (epoch >= args.max_bs_steps * args.bs_step_size):# and (epoch != 0) and ((bs_idx == args.max_bs_steps) or (bs_idx == args.max_bs_steps - 1)):
                print('-' * print_num)
                print('| Using best model...'.ljust(print_num_minus, ' '), '|')
                model1.load_state_dict(copy.deepcopy(best_model.state_dict()))
            
            if args.vis:

                vis_WaveNetX4(visdom, epoch + 1, train_epoch_loss, train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_m_jc1, train_m_dc1, 
                              val_epoch_loss_sup1, val_m_jc1, val_m_dc1, val_epoch_loss_sup_best, best_jc, best_dc)
                
                draw_img = draw_pred_sup_WaveNetX4(cfg['NUM_CLASSES'], mask_train_sup, mask_val, pred_train_sup1, pred_val, pred_val_best, train_eval_list1, val_eval_list1, val_eval_list_best)
                vis_image_WaveNetX4(visdom, draw_img[0], draw_img[1], draw_img[2], draw_img[3], draw_img[4], stich_img=False)
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


    # Dataset
    dataset_test = imagefolder_WaveNetX(
        data_dir=cfg['PATH_DATASET'] + '/test',
        data_transform_1=data_transforms['test'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None
    )

    dataloaders = {'test': DataLoader(dataset_test, batch_size=32, shuffle=False)}

    # load best model
    # Load checkpoint
    # checkpoint = torch.load(args.path_model, map_location=device)
    # model.load_state_dict(checkpoint, strict=False)

    model = best_model

    # Loss function
    criterion = segmentation_loss(args.loss, False).to(device)

    # # Visualization
    # if args.vis:
    #     visdom = visdom_initialization_WaveNetX(env=f'{args.dataset_name}-{network_name}', port=args.visdom_port)

    # Testing
    model.eval()
    # results_path = os.path.join(cfg['PATH_SEG_RESULT'], args.dataset_name, os.path.splitext(os.path.basename(args.path_model))[0])
    # os.makedirs(results_path, exist_ok=True)

    test_loss_sup_1, test_loss_sup_2, test_loss_sup_3 = 0.0, 0.0, 0.0
    score_list_test1, mask_list_test, name_list_test = [], [], []

    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            print(f"Processing {i + 1} out of {len(dataloaders['test'])}...", end="\r")
            inputs_test = Variable(data['image'].to(device))
            mask_test = Variable(data['bin_mask'].to(device).long())
            name_test = data['ID']

            # Model prediction
            if _net.endswith('v0'):
                outputs_test1, outputs_test2, outputs_test3 = model(inputs_test)
                test_loss_sup_1 += criterion(outputs_test1, mask_test).item()
                test_loss_sup_2 += criterion(outputs_test2, mask_test).item()
                test_loss_sup_3 += criterion(outputs_test3, mask_test).item()
            else:
                outputs_test1 = model(inputs_test)
                test_loss_sup_1 += criterion(outputs_test1, mask_test).item()

            # Gather results
            if i == 0:
                score_list_test1 = outputs_test1
                mask_list_test = mask_test
                name_list_test = name_test
            else:
                score_list_test1 = torch.cat((score_list_test1, outputs_test1), dim=0)
                mask_list_test = torch.cat((mask_list_test, mask_test), dim=0)
                name_list_test = np.append(name_list_test, name_test, axis=0)

            # save_test_2d(cfg['NUM_CLASSES'], outputs_test1, name_test, args.threshold, results_path, cfg['PALETTE'])
            # # Visualization of filter banks
            # if args.vis:
            #     for f_idx in range(model.dwt.nfil):
            #         vis_filter_bank_WaveNetX(visdom, fb_2d_list=model.dwt.get_fb_2d_list(for_vis=True), fil_idx=f_idx, figure_name='2D Filter #{}'.format(f_idx))

        # Calculate testing losses and metrics
        print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
        print_num_minus = print_num - 2
        print_num_half = int(print_num / 2 - 1)

        num_batches = {'val': len(dataloaders['test'])}
        
        print(test_loss_sup_1,test_loss_sup_2,test_loss_sup_3)
        test_epoch_loss_sup1 = print_val_loss_WaveNetX4(
            test_loss_sup_1, test_loss_sup_1, num_batches, print_num, print_num_minus
        )
        test_eval_list, test_m_jc, test_m_dice = print_test_eval(cfg['NUM_CLASSES'], score_list_test1, mask_list_test, print_num)

    print("Testing complete.")