# SimplerEnv-SOFAR

## Installation

Create an anaconda environment:

```
conda create -n simpler_env python=3.10 (any version above 3.10 should be fine)
conda activate simpler_env
```

Clone this repo:

```
git clone https://github.com/Zhangwenyao1/SimplerEnv-SOFAR
```

This repository's code is based in the [SimplerEnv](https://github.com/simpler-env/SimplerEnv) and the ManiSkill2 based [ManiSkill2](https://github.com/Jiayuan-Gu/ManiSkill2_real2sim) from [JiayuanGU](https://github.com/Jiayuan-Gu) for the Open-Loop control.

Install SimplerEnv:

see [SimplerEnv](https://github.com/simpler-env/SimplerEnv) for installation instructions.

Install GSNET:

see [`GSNET/READNE.md`]

The motion planning module code is based in the [graspnet-baseline](https://github.com/graspnet/graspnet-baseline).

Install Motion Planning Moduel:

see [`./plan/README.md`]

(Acknowledge [haoranliu](https://github.com/lhrrhl0419) for the code) !!!
The motion planning module code is based in the [ompl](https://github.com/lyfkyle/pybullet_ompl).

Install SpatialAgent:

see [SoFar](https://github.com/qizekun/SoFar) for installation instructions.


Notion:

You need modify the relative/absolute path in some files.

## Citation

If you find our ideas / environments helpful, please cite our work at

```

@article{qi2025sofar,
  author = {Qi, Zekun and Zhang, Wenyao and Ding, Yufei and Dong, Runpei and Yu, Xinqiang and Li, Jingwen and Xu, Lingyun and Li, Baoyu and He, Xialin and Fan, Guofan and Zhang, Jiazhao and He, Jiawei and Gu, Jiayuan and Jin, Xin and Ma, Kaisheng and Zhang, Zhizheng and Wang, He and Yi, Li},
  title = {SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation},
  journal = {arXiv preprint arXiv:2502.13143},
  year = {2025}
}
```
