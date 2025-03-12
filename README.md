# SimplerEnv-SOFAR

The [SoFar](https://arxiv.org/pdf/2502.13143) manipulation evaluation for Simpler_Env.

## Installation

**Create an anaconda environment:**

```
conda create -n simpler_env python=3.10 (any version above 3.10 should be fine)
conda activate simpler_env
```

**Clone this repo:**

```
git clone https://github.com/Zhangwenyao1/SimplerEnv-SOFAR
```

This repository's code is based in the [SimplerEnv](https://github.com/simpler-env/SimplerEnv) and the ManiSkill2 based [ManiSkill2](https://github.com/Jiayuan-Gu/ManiSkill2_real2sim) from [JiayuanGU](https://github.com/Jiayuan-Gu) for the Open-Loop control.

**Install SimplerEnv:**

see [SimplerEnv](https://github.com/simpler-env/SimplerEnv) for installation instructions.

**Install GSNET:**

see [`GSNET/READNE.md`]

This code is based on [graspnet-baseline](https://github.com/graspnet/graspnet-baseline), you can use the code to predict the grasp.

**Install Motion Planning Moduel:**

see [`./plan/README.md`]

You need modify the checkpoint or config  path  in following files in plan:

> plan/src/utils/constants.py

The motion planning module code is based in the [ompl](https://github.com/lyfkyle/pybullet_ompl).

**Install SoFar:**

see [SoFar](https://github.com/qizekun/SoFar) for installation instructions.

**Notion:**

You have to install GroundingDINO for the evaluation.

You need modify the checkpoint or config  path  in following files in SoFar:

> SoFar/depth/metric3dv2.py
>
> SoFar/segmentation/grounding_dino.py
>
> SoFar/segmentation/sam.py
>
> SoFar/serve/pointso.py

## Execution

You can run the evaluation in the script folder for different tasks:

> sh scripts/sofar_bridge.sh

## Acknowledgement

We would like to express our deepest gratitude to [haoran liu](https://github.com/lhrrhl0419) for the planning module and experiments !!!

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
