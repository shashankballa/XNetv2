# Test Script Usage

The mac_test.py script is used to evaluate a trained model on a specified test dataset. Below are examples for running the test script with different network architectures.

## Example Commands
Testing XNetv2 Model
```
python mac_test.py -p "dataset_tiff/GLAS/GLAS/XNetv2-l=0.5-e=200-s=50-g=0.5-b=16-uw=0.5-w=20-20-80/best_XNetv2_Jc_0.7043.pth" -n XNetv2
```
Testing WaveNetX Model* (To be Updated from Shashank)
```
python mac_test.py -p "dataset_tiff/GLAS/GLAS/WaveNetX-l=0.5-e=200-s=50-g=0.5-b=16-uw=0.5-w=20-20-80/best_WaveNetX_Jc_0.6898.pth" -n 
```
WaveNetX
Testing WaveNetX2 Model
```
python mac_test.py -p "dataset_tiff/GLAS/GLAS/WaveNetX2-l=0.5-e=200-s=50-g=0.5-b=16-uw=0.5-w=20-20-80/best_wavenetx2_Jc_0.6215.pth" -n 
```
WaveNetX2
Required Arguments
```
-p: Path to the trained model .pth file.
-n: Name of the network architecture used during training (e.g., XNetv2, WaveNetX, WaveNetX2).
Optional Arguments
-b: Batch size for testing (default: 16).
--device: Device to run the test on (cuda or cpu; default: cuda if available).
--dataset_name: Name of the dataset used for testing (default: GLAS).
```