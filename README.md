# Voxel Diffusion

## Setup
```
pip install -r requirements.txt
cd smalldiffusion
pip install -e .
```
Put [voxel data](https://www.kaggle.com/datasets/xiuyuanqiu/sub-30-voxel-houses-in-numpy) in `data/`.
 
## Run
Run fashion mnist diffusion with `accelerate launch smol_test.py`.
Try out voxel diffusion with `accelerate launch vox_3d.py`

## Develop
Edit and add models to `smalldiffusion/src/smalldiffusion/`

## Notes
- vocks.py is the testbed for 3d unet stuff
- old_cube_code/ contains a lot of the old cube and square test code
- gif_script.py generates gifs from a directory of images
