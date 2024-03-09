# ReShader

> ReShader: View-Dependent Highlights for Single Image View-Synthesis  
> [Avinash Paliwal](http://avinashpaliwal.com/),
> [Brandon Nguyen](https://brandon.nguyen.vc/about/), 
> [Andrii Tsarov](https://www.linkedin.com/in/andrii-tsarov-b8a9bb13), 
> [Nima Khademi Kalantari](http://nkhademi.com/)   
> SIGGRAPH Asia 2023 (TOG)

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2309.10689)
[![Project Page](https://img.shields.io/badge/ReShader-Website-blue?logo=googlechrome&logoColor=blue)](https://people.engr.tamu.edu/nimak/Papers/SIGAsia2023_Reshader)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://youtu.be/XW-tl48D3Ok)

---------------------------------------------------
<p align="center" >
  <a href="">
    <img src="assets/teaser.gif?raw=true" alt="demo" width="85%">
  </a>
</p>

## Prerequisites
You can setup the anaconda environment using:
```
conda env create -f environment.yml
conda activate reshader
```

Download pretrained models. 
The following script from [3D Moments](https://github.com/google-research/3d-moments) will download their pretrained models and [RGBD-inpainting networks](https://github.com/vt-vl-lab/3d-photo-inpainting).
```
./download.sh
```


## Demos
We provided some examples in the `examples/` folder. You can render novel views with view-dependent highlights using:

```
python renderer.py --input_dir examples/camera/ --config configs/render.txt
```

## Training
Training code and dataset to be added.

## Citation
```
@article{paliwal2023reshader,
author = {Paliwal, Avinash and Nguyen, Brandon G. and Tsarov, Andrii and Kalantari, Nima Khademi},
title = {ReShader: View-Dependent Highlights for Single Image View-Synthesis},
year = {2023},
issue_date = {December 2023},
volume = {42},
number = {6},
journal = {ACM Trans. Graph.},
month = {dec},
articleno = {216},
numpages = {9},
}
```


## Acknowledgement
The novel view synthesis part of the code is borrowed from [3D Moments](https://github.com/google-research/3d-moments).