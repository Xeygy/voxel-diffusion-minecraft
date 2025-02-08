# Voxel Diffusion

## Setup
```
pip install -r requirements.txt
cd smalldiffusion
pip install -e .
```
Put voxel data in `data/`.
 
## Run
Run fashion mnist diffusion with `accelerate launch smol_test.py`.
Try out voxel diffusion with `accelerate launch vox_3d.py`

## Develop
Edit and add models to `smalldiffusion/src/smalldiffusion/`

## Notes
- vox_3d is the testbed for broken 3d unet stuff
- square/squar_dit is the testbed for generating squares via diffusion
- cube/cube_eval is the testbed for generating cubes via diffusion
- gif_script generates gifs from a directory of images