# Voxel Diffusion

## Setup
```
pip install -r requirements.txt
cd smalldiffusion
pip install -e .
```

## Run
Run fashion mnist diffusion with `accelerate launch smol_test.py`.
Try out voxel diffusion with `accelerate launch vox_3d.py`

## Develop
Edit and add models to `smalldiffusion/src/smalldiffusion/`