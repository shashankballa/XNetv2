import os
import argparse
import random
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from warnings import simplefilter
from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from models.getnetwork import get_network
from dataload.dataset_2d import imagefolder_XNetv2
from loss.loss_function import segmentation_loss
from config.train_test_config.train_test_config import print_test_eval, save_val_best_sup_2d, print_val_loss_XNetv2, save_test_2d
import time

simplefilter(action='ignore', category=FutureWarning)

def init_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)

def safe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_model', default='.../XNetv2/checkpoints/CREMI/.../best_XNetv2_Jc_0.7798.pth')
    parser.add_argument('--dataset_name', default='GLAS', help='GlaS, CREMI')
    parser.add_argument('--if_mask', default=True, type=bool)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('--wavelet_type', default='haar', help='haar, db2, bior1.5, coif1, dmey')
    parser.add_argument('--alpha', default=[0.2, 0.2], type=list)
    parser.add_argument('--beta', default=[0.65, 0.65], type=list)
    parser.add_argument('-n', '--network', default='XNetv2')
    args = parser.parse_args()

    # GPU or CPU selection
    device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device}")

    # Initialize random seeds
    init_seeds(42)

    # Dataset configuration
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    # Create result directories
    path_seg_results = os.path.join(cfg['PATH_SEG_RESULT'], dataset_name)
    path_seg_results = os.path.join(path_seg_results, os.path.splitext(os.path.basename(args.path_model))[0])
    safe_create_dir(path_seg_results)

    # Data transformations
    data_transforms = data_transform_2d(cfg['INPUT_SIZE'])
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    # Loss function
    criterion = segmentation_loss(args.loss, False).to(device)

    # Dataset and DataLoader
    dataset_val = imagefolder_XNetv2(
        data_dir=os.path.join(cfg['PATH_DATASET'], 'test'),
        data_transform_1=data_transforms['test'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=args.alpha,
        beta=args.beta,
        sup=True
    )
    dataloaders = {'test': DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)}

    # Model initialization
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).to(device)
    state_dict = torch.load(args.path_model, map_location=device)
    model.load_state_dict(state_dict)
       
    test_loss_sup_1, test_loss_sup_2, test_loss_sup_3 = 0.0, 0.0, 0.0
    score_list_test1, mask_list_test, name_list_test = [], [], []

    # Testing loop
    since = time.time()
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(dataloaders['test']):
            print(f"Processing {i + 1} out of {len(dataloaders['test'])}...", end="\r")
            inputs_test = Variable(data['image'].to(device))
            inputs_L = Variable(data['L'].to(device))
            inputs_H = Variable(data['H'].to(device))
            mask_test = Variable(data['bin_mask'].to(device).long())
            name_test = data['ID']

            outputs_test1, outputs_test2, outputs_test3, _ = model(inputs_test, inputs_L, inputs_H)
            test_loss_sup_1 += criterion(outputs_test1, mask_test).item()
            test_loss_sup_2 += criterion(outputs_test2, mask_test).item()
            test_loss_sup_3 += criterion(outputs_test3, mask_test).item()

            # Gather results
            if i == 0:
                score_list_test1 = outputs_test1
                mask_list_test = mask_test
                name_list_test = name_test
            else:
                score_list_test1 = torch.cat((score_list_test1, outputs_test1), dim=0)
                mask_list_test = torch.cat((mask_list_test, mask_test), dim=0)
                name_list_test = np.append(name_list_test, name_test, axis=0)
            save_test_2d(cfg['NUM_CLASSES'], outputs_test1, name_test, args.threshold, path_seg_results, cfg['PALETTE'])

        print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
        print_num_minus = print_num - 2
        print_num_half = int(print_num / 2 - 1)

        num_batches = {'val': len(dataloaders['test'])}

        test_epoch_loss_sup1, test_epoch_loss_sup2, test_epoch_loss_sup3 = print_val_loss_XNetv2(
            test_loss_sup_1, test_loss_sup_2, test_loss_sup_3, num_batches, print_num, print_num_minus
        )
        test_eval_list, test_m_jc, test_m_dice = print_test_eval(cfg['NUM_CLASSES'], score_list_test1, mask_list_test, print_num)

    # Print time elapsed
    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)
    print(f"Testing Completed in {h:.0f}h {m:.0f}m {s:.0f}s")
