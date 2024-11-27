import os
import argparse
from pathlib import Path
from PIL import Image  # To convert .png to .tif format
import shutil

def create_dirs(base_dir, sub_dirs):
    """Helper function to create directories."""
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

def convert_and_rename(src_dir, dest_dir, start_index=1):
    """Convert .png files to .tif and rename sequentially."""
    files = sorted(Path(src_dir).glob("*.png"))  # Only process .png files
    for idx, file_path in enumerate(files, start=start_index):
        new_name = f"{idx}.tif"
        dest_path = os.path.join(dest_dir, new_name)
        # Convert .png to .tif
        img = Image.open(file_path)
        img.save(dest_path, format="TIFF")
    return len(files)

def organize_dataset(dataset_name, dataset_base_path, processed_base_path):
    # Paths to original dataset
    dataset_path = os.path.join(dataset_base_path, dataset_name)
    train_folder = os.path.join(dataset_path, "Train Folder")
    test_folder = os.path.join(dataset_path, "Test Folder")
    val_folder = os.path.join(dataset_path, "Validation Folder")

    # Paths for the new dataset structure
    output_path = os.path.join(processed_base_path, dataset_name)
    train_sup_100_path = os.path.join(output_path, "train_sup_100")
    train_sup_20_path = os.path.join(output_path, "train_sup_20")
    train_unsup_80_path = os.path.join(output_path, "train_unsup_80")
    val_path = os.path.join(output_path, "val")

    # Create the new directory structure
    create_dirs(output_path, [
        "train_sup_100/image", "train_sup_100/mask",
        "train_sup_20/image", "train_sup_20/mask",
        "train_unsup_80/image",
        "val/image", "val/mask"
    ])

    # Step 1: Process Train Folder
    # Copy all supervised training data to train_sup_100
    num_train_sup_100 = convert_and_rename(
        os.path.join(train_folder, "img"),
        os.path.join(train_sup_100_path, "image"),
    )
    convert_and_rename(
        os.path.join(train_folder, "labelcol"),
        os.path.join(train_sup_100_path, "mask"),
    )

    # Copy 20% of supervised training data to train_sup_20
    num_train_sup_20 = int(0.2 * num_train_sup_100)
    for idx in range(1, num_train_sup_20 + 1):
        img_src = os.path.join(train_sup_100_path, "image", f"{idx}.tif")
        mask_src = os.path.join(train_sup_100_path, "mask", f"{idx}.tif")
        shutil.copy(img_src, os.path.join(train_sup_20_path, "image", f"{idx}.tif"))
        shutil.copy(mask_src, os.path.join(train_sup_20_path, "mask", f"{idx}.tif"))

    # Copy 80% of unsupervised training data (images only) to train_unsup_80
    for idx in range(num_train_sup_20 + 1, num_train_sup_100 + 1):
        img_src = os.path.join(train_sup_100_path, "image", f"{idx}.tif")
        shutil.copy(img_src, os.path.join(train_unsup_80_path, "image", f"{idx}.tif"))

    # Step 2: Process Validation Folder
    convert_and_rename(
        os.path.join(val_folder, "img"),
        os.path.join(val_path, "image"),
    )
    convert_and_rename(
        os.path.join(val_folder, "labelcol"),
        os.path.join(val_path, "mask"),
    )

    # Step 3: Process Test Folder
    # Only rename and convert to .tif within the same directory
    convert_and_rename(
        os.path.join(test_folder, "img"),
        os.path.join(test_folder, "img"),
    )
    convert_and_rename(
        os.path.join(test_folder, "labelcol"),
        os.path.join(test_folder, "labelcol"),
    )

    print(f"Dataset {dataset_name} has been successfully organized in {output_path}!")

    # print the number of images in each folder, and shape of the first image
    print("Number of images in each folder:")
    print(f"train_sup_100/image: {num_train_sup_100}")
    print(f"train_sup_20/image: {num_train_sup_20}")
    print(f"train_unsup_80/image: {num_train_sup_100 - num_train_sup_20}")
    print(f"Training image shape: {Image.open(os.path.join(train_sup_100_path, 'image', '1.tif')).size}")
    print(f"Validation image shape: {Image.open(os.path.join(val_path, 'image', '1.tif')).size}")
    print(f"Test image shape: {Image.open(os.path.join(test_folder, 'img', '1.tif')).size}")
    print(f"Mask image shape: {Image.open(os.path.join(val_path, 'mask', '1.tif')).size}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize dataset into desired structure.")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset folder to process (e.g., GLas)."
    )
    args = parser.parse_args()

    # Common directories for all datasets
    dataset_base_path = "dataset/"
    processed_base_path = "dataset_tiff/"
    
    # Call the organization function
    organize_dataset(args.dataset_name, dataset_base_path, processed_base_path)