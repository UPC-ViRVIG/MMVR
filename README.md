# Combining Motion Matching and Orientation Prediction to Animate Avatars for Consumer-Grade VR Devices

---

<p align="center">
  <img 
    width="940"
    height="252"
    src="docs/assets/img/teaser.jpg"
  >
</p>

This repository contains the implementation of the method shown in the paper *Combining Motion Matching and Orientation Prediction to Animate Avatars for Consumer-Grade VR Devices* presented at the **21st annual ACM SIGGRAPH / Eurographics Symposium on Computer Animation (SCA 2022)**.

Get an overview of the paper by visiting the [project website](https://upc-virvig.github.io/MMVR/) or watching the [video](https://www.youtube.com/embed/crU9oLX0GnM)!

Download the paper [here](docs/assets/pdf/motion_matching_vr.pdf)!

![dancing avatar](docs/assets/img/dancing.gif)

![first-person view](docs/assets/img/first_person.gif)

---

## Contents

1. [Structure](#structure)
2. [Quick Start](#quick-start)
3. [Data](#data)
4. [Citation](#citation)
5. [License](#license)

## Structure

The project is divided into two folders, ``MMVR`` and ``python``, containing the Unity project and orientation prediction training scripts. Although the neural network is trained using PyTorch, the animation files are preprocessed using Unity before being used for training. Therefore, most of the preprocessing scripts are located inside``MMVR``.

## Quick Start

1. Clone this repository

2. Install **Unity 2021.2.13f1** (other versions may work but are not tested)

3. Download the processed dataset [here]() and extract it into ``Unity_Project_Path/Assets/MMData``, thus, ``MMData`` folder should contain three folders ``Animations``, ``Data`` and ``Models``.

4. Open the project and navigate to the ``Assets/Scenes`` folder and choose either ``Demo`` (FinalIK required) or ``DemoOnlyHMD``.
   
   > Upper Body inverse kinematics are implemented using [FinalIK](https://assetstore.unity.com/packages/tools/animation/final-ik-14290). Import the assets through the Package Manager window (you will need a valid license from the Unity Asset Store).

5. Use Oculus Link or Air Link to connect your Meta Quest to the PC and press Play in the Unity Editor. 

## Data

The processed data required to execute the project can be downloaded from [here]().

If you wish to download all raw *.bvh* files used for training the orientation prediction network and the motion matching animation database, download [here]() instead.

All motions are recorded by the same actor in different VR systems (Meta Quest and HTC VIVE) while performing different actions (VR applications and video games, and some interaction actions to increase the coverage of the dataset). The file *locomotion_vr.bvh* is used as the animation database for Motion Matching.

## Citation

If you find our research useful, please cite our paper:

```
@article{ponton2022mmvr,
author = {Ponton, Jose Luis and Yun, Haoran and Andujar, Carlos and Pelechano, Nuria},
title = {Combining Motion Matching and Orientation Prediction to Animate Avatars for Consumer-Grade VR Devices},
journal = {Computer Graphics Forum},
year = {2022}
}
```

## License

This work is licensed under CC BY-NC-SA 4.0.
The project and data is freely available for free non-commercial use, and may be redistributed under these conditions. Please, see the [license](LICENSE) for further details.
