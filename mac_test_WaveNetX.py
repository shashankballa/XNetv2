import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import os
import re
import numpy as np
import random
from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from dataload.dataset_2d import imagefolder_WaveNetX
from models.getnetwork import get_network
from config.visdom_config.visual_visdom import visdom_initialization_XNetv2, vis_filter_bank_WaveNetX
from config.train_test_config.train_test_config import print_val_eval_sup, save_val_best_sup_2d, print_val_loss_XNetv2
from loss.loss_function import segmentation_loss
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def init_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)


def parse_model_name(filename):
    """
    Parse the model name to extract parameters like nfil, flen, lr, etc.

    Args:
        filename (str): Model filename (e.g., best_WaveNetXv2_Jc_0.8860-l=1.0-e=800-s=100-g=0.5-b=4-w=100-nf=4-fl=4-bs=100-mbs=3-sd=69-fbl0=0.2-fbl1=0.005-b2s.pth)

    Returns:
        dict: Parsed parameters as a dictionary
    """
    # Pre-process filename to remove invalid trailing characters
    filename = re.sub(r'-([^\w=]|$)', '', filename)

    # Regex to match key-value pairs
    param_pattern = re.compile(r'-?([a-zA-Z0-9]+)=([-+]?\d*\.?\d+)')
    params = dict(param_pattern.findall(filename))

    # Convert numerical values to appropriate types
    for k, v in params.items():
        if v.isdigit():
            params[k] = int(v)
        else:
            try:
                params[k] = float(v)
            except ValueError:
                pass  # Leave as a string if it can't be converted

    return params



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_model', default='checkpoints/best_WaveNetXv2_Jc_0.8860-l=1.0-e=800-s=100-g=0.5-b=4-w=100-nf=4-fl=4-bs=100-mbs=3-sd=69-fbl0=0.2-fbl1=0.005-b2s.pth')
    parser.add_argument('--dataset_name', default='GLAS', help='CREMI, GlaS, ISIC-2017')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--visdom_port', default=16672)
    parser.add_argument('--vis', default=True, help='Enable visualization or not')
    parser.add_argument('--loss', default='dice')
    parser.add_argument('--wavelet_type', default='haar', help='haar, db2, bior1.5, coif1, dmey')
    parser.add_argument('--alpha', default=[0, 0.4])
    parser.add_argument('--beta', default=[0.5, 0.8])
    args = parser.parse_args()

    # Parse model parameters from the filename
    model_name = os.path.basename(args.path_model)
    model_params = parse_model_name(model_name)

    # Print parsed parameters
    print(f"Parsed parameters from model name: {model_params}")

    # Set default parameters or use parsed ones
    batch_size = model_params.get('b', 16)
    lr = model_params.get('l', 0.001)
    step_size = model_params.get('s', 50)
    gamma = model_params.get('g', 0.5)
    nfil = model_params.get('nf', 4)
    flen = model_params.get('fl', 4)
    fbl0 = model_params.get('fbl0', 0.2)
    fbl1 = model_params.get('fbl1', 0.005)

    # Initialize device
    device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device}")

    init_seeds(args.seed)

    # Config
    cfg = dataset_cfg(args.dataset_name)
    data_transforms = data_transform_2d(cfg['INPUT_SIZE'])
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    # Dataset
    dataset_test = imagefolder_WaveNetX(
        data_dir=cfg['PATH_DATASET'] + '/test',
        data_transform_1=data_transforms['test'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=args.alpha,
        beta=args.beta,
        sup=True,
        num_images=None
    )
    dataloaders = {'test': DataLoader(dataset_test, batch_size=batch_size, shuffle=False)}

    # Model
    network_name = model_params.get('network', 'WaveNetXv2')
    model = get_network(
        network_name,
        cfg['IN_CHANNELS'],
        cfg['NUM_CLASSES'],
        nfil=nfil,
        flen=flen
    ).to(device)
    model1 = get_network(network_name, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], 
        nfil=nfil, flen=flen, # WaveNetXv0 and WaveNetXv1
        flen_start=flen, nfil_start=nfil# WaveNetXv2
        ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.path_model, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    # Loss function
    criterion = segmentation_loss(args.loss, False).to(device)

    # Visualization
    if args.vis:
        visdom = visdom_initialization_XNetv2(env=f'{args.dataset_name}-{network_name}', port=args.visdom_port)

    # Testing
    model.eval()
    results_path = os.path.join(cfg['PATH_SEG_RESULT'], args.dataset_name, os.path.splitext(os.path.basename(args.path_model))[0])
    os.makedirs(results_path, exist_ok=True)

    test_loss_sup_1, test_loss_sup_2, test_loss_sup_3 = 0.0, 0.0, 0.0
    score_list_test1, mask_list_test, name_list_test = [], [], []

    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            print(f"Processing {i + 1} out of {len(dataloaders['test'])}...", end="\r")
            inputs_test = Variable(data['image'].to(device))
            mask_test = Variable(data['bin_mask'].to(device).long())
            name_test = data['ID']

            # Model prediction
            if network_name.endswith('v0'):
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

            # Visualization of filter banks
            if args.vis:
                for f_idx in range(model1.dwt.nfil):
                    vis_filter_bank_WaveNetX(visdom, fb_2d_list=model1.dwt.get_fb_2d_list(for_vis=True), fil_idx=f_idx, figure_name='2D Filter #{}'.format(f_idx))

        # Calculate testing losses and metrics
        print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
        print_num_minus = print_num - 2
        print_num_half = int(print_num / 2 - 1)

        num_batches = {'val': len(dataloaders['test'])}
        
        print(test_loss_sup_1,test_loss_sup_2,test_loss_sup_3)
        test_epoch_loss_sup1, test_epoch_loss_sup2, test_epoch_loss_sup3 = print_val_loss_XNetv2(
            test_loss_sup_1, test_loss_sup_2, test_loss_sup_3, num_batches, print_num, print_num_minus
        )
        test_eval_list, test_m_jc, test_m_dice = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_test1, mask_list_test, print_num_minus)

    print("Testing complete.")
