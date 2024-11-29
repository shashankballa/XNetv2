from torchvision import transforms
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import time
from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from models.getnetwork import get_network
from dataload.dataset_2d import imagefolder_XNetv2
from config.train_test_config.train_test_config import print_test_eval, save_test_2d
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def init_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)

# "dataset_tiff/GLAS/GLAS/XNetv2-l=0.5-e=200-s=50-g=0.5-b=16-uw=0.5-w=20-20-80/best_XNetv2_Jc_0.7043.pth" 
# "dataset_tiff/GLAS/GLAS/WaveNetX-l=0.5-e=200-s=50-g=0.5-b=16-uw=0.5-w=20-20-80/best_WaveNetX_Jc_0.6898.pth" 
# "dataset_tiff/GLAS/GLAS/WaveNetX2-l=0.5-e=200-s=50-g=0.5-b=16-uw=0.5-w=20-20-80/best_wavenetx2_Jc_0.6215.pth" 

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_model', required=True, help='Path to the trained model file (.pth)')
    parser.add_argument('--dataset_name', default='GLAS', help='Dataset name: GLAS, CREMI, etc.')
    parser.add_argument('--if_mask', default=True, type=bool, help='Whether to evaluate with masks')
    parser.add_argument('--threshold', default=0.5, type=float, help='Threshold for binary predictions')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size for testing')
    parser.add_argument('--wavelet_type', default='haar', help='Wavelet type: haar, db2, etc.')
    parser.add_argument('--alpha', default=[0.2, 0.2], nargs=2, type=float, help='Alpha coefficients for augmentation')
    parser.add_argument('--beta', default=[0.65, 0.65], nargs=2, type=float, help='Beta coefficients for augmentation')
    parser.add_argument('-n', '--network', default='XNetv2', help='Network architecture: XNetv2, WaveNetX, etc.')
    parser.add_argument('--device', default='cuda', help='Device to run the test on: cuda, cpu')
    args = parser.parse_args()

    # Initialize seeds
    init_seeds(42)

    # Configurations
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)
    print(f"Dataset: {dataset_name}")

    # Paths for saving results
    path_seg_results = os.path.join(cfg['PATH_SEG_RESULT'], dataset_name)
    os.makedirs(path_seg_results, exist_ok=True)
    path_seg_results = os.path.join(path_seg_results, os.path.splitext(os.path.basename(args.path_model))[0])
    os.makedirs(path_seg_results, exist_ok=True)
    print(f"Results will be saved to: {path_seg_results}")

    # Data transformations
    data_transforms = data_transform_2d(cfg['INPUT_SIZE'])
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    # Test dataset
    dataset_test = imagefolder_XNetv2(
        data_dir=os.path.join(cfg['PATH_DATASET'], 'test'),
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=args.alpha,
        beta=args.beta,
        sup=True,
        num_images=None
    )

    dataloaders = {'test': DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)}
    print(f"Loaded {len(dataloaders['test'].dataset)} test samples.")

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).to(device)
    model.load_state_dict(torch.load(args.path_model, map_location=device))
    model.eval()
    print(f"Loaded model from: {args.path_model}")

    # Testing
    print("Starting testing...")
    since = time.time()
    test_loss = 0.0
    score_list_test, mask_list_test, name_list_test = None, None, None

    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            inputs_test = Variable(data['image'].to(device))
            inputs_L = Variable(data['L'].to(device))
            inputs_H = Variable(data['H'].to(device))
            name_test = data['ID']

            if args.if_mask:
                mask_test = Variable(data['mask'].to(device))

            # Forward pass
            outputs_test1, outputs_test2, outputs_test3, _ = model(inputs_test, inputs_L, inputs_H)

            # Save predictions or compute metrics
            if args.if_mask:
                if i == 0:
                    score_list_test = outputs_test1
                    name_list_test = name_test
                    mask_list_test = mask_test
                else:
                    score_list_test = torch.cat((score_list_test, outputs_test1), dim=0)
                    name_list_test = np.append(name_list_test, name_test, axis=0)
                    mask_list_test = torch.cat((mask_list_test, mask_test), dim=0)
            else:
                save_test_2d(cfg['NUM_CLASSES'], outputs_test1, name_test, args.threshold, path_seg_results, cfg['PALETTE'])

    # Evaluate and save metrics
    if args.if_mask:
        print('=' * 50)
        test_eval_list = print_test_eval(cfg['NUM_CLASSES'], score_list_test, mask_list_test, 48)
        save_test_2d(cfg['NUM_CLASSES'], score_list_test, name_list_test, args.threshold, path_seg_results, cfg['PALETTE'])

    # Timing
    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)
    print(f"Testing completed in {h:.0f}h {m:.0f}m {s:.0f}s.")
    print('=' * 50)
