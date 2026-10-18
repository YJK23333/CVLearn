# U-Net + Oxford-IIIT Pet

This project follows the CVLearn layout:

```text
CVLearn/
├── data/
│   └── oxford-iiit-pet/       # downloaded dataset
├── Unet/
│   ├── src/
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── test.py
│   │   └── utils.py
│   ├── checkpoints/
│   └── outputs/
└── ...
```

## Dependencies

Install PyTorch and torchvision compatible with your environment. The following packages are also needed:

```powershell
pip install matplotlib pillow numpy
```

## Run

Open PowerShell in `CVLearn\Unet\src`:

```powershell
python model.py
python train.py
python test.py
```

The first dataset initialization downloads Oxford-IIIT Pet if it is not already present.

## Task and mask convention

This starter project performs binary segmentation:

- foreground (pet) = 1
- background = 0
- official trimap values 1 and 3 are mapped to foreground
- official trimap value 2 is mapped to background

Images are resized using bilinear interpolation. Masks are resized using nearest-neighbor interpolation.

## Configuration

Adjust architecture, input size, training, augmentation and paths in `config.py`.
For this implementation, image height and width should be divisible by the total downsampling factor implied by `CHANNEL_MULTIPLIERS`.

The default architecture is intentionally set to `BASE_CHANNELS = 32` to be lighter than the classic 64-channel U-Net.
