
# WaveNetX

This is the official code of WaveNetX, incorporating trainable Wavelets into wavelet-based image segmentation networks. It is inspired by XNet v2 (BIBM 2024).

## Requirements
This code has been designed and optimized to be run on both Metal and Cuda-enabled device. It is also possible to run on CPU but is not recommended.

## Usage
**Data preparation**
Build your own dataset and its directory tree should be look like this:
```
dataset_tiff
├── train_sup_100
    ├── image
        ├── 1.tif
        ├── 2.tif
        └── ...
    └── mask
        ├── 1.tif
        ├── 2.tif
        └── ...
├── train_sup_20
    ├── image
    └── mask
├── train_unsup_80
    ├── image
└── val
    ├── image
    └── mask
```

If you have data stored in png form, you can place it in the same format into a folder named dataset instead of dataset_tiff.

Then, run:
```python
python prep_dataset.py #to convert 
```

**Configure dataset parameters**
>Add configuration in [/config/dataset_config/dataset_config.py](https://github.com/Yanfeng-Zhou/XNetv2/tree/main/config/dataset_config/dataset_config.py)
>The configuration should be as follows：
>
```
# 2D Dataset
'CREMI':
	{
		'PATH_DATASET': '.../XNetv2/dataset/CREMI',
		'PATH_TRAINED_MODEL': '.../XNetv2/checkpoints',
		'PATH_SEG_RESULT': '.../XNetv2/seg_pred',
		'IN_CHANNELS': 1,
		'NUM_CLASSES': 2,
		'MEAN': [0.503902],
		'STD': [0.110739],
		'INPUT_SIZE': (128, 128),
		'PALETTE': list(np.array([
			[255, 255, 255],
			[0, 0, 0],
		]).flatten())
	},
```

**Training**
```python
python mac_train_WaveNetX.py -l 2.0 -e 2000 -s 80 -g 0.7 -b 4 -w 40 --nfil 7 --flen 6 --nfil_step -1 --flen_step 2 --bs_step 100 --max_bs_steps 3 --seed 666666 --fbl0 0.1 --fbl1 0.01 -b2s -ub 80
```
or
```bash
bash scripts/run_py.sh mac_train_WaveNetX.py -l 2.0 -e 2000 -s 80 -g 0.7 -b 4 -w 40 --nfil 7 --flen 6 --nfil_step -1 --flen_step 2 --bs_step 100 --max_bs_steps 3 --seed 666666 --fbl0 0.1 --fbl1 0.01 -b2s -ub 80
```

**Testing**
```python
python mac_test_WaveNetX.py -p "checkpoints/desired_path.pth" #for the model you would like to test
```
or
```bash
bash scripts/run_py.sh mac_test_WaveNetX.py -p "checkpoints/desired_path.pth" 
```

## Citation
If our work is useful for your research, please cite our model.
```
```

