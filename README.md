# Cricket Motion Prediction

This project focuses on predicting the locomotion direction and velocity of crickets based on joint angle sequences extracted from walking videos. Deep learning models, including LSTM and Transformer architectures, are employed to capture the temporal dynamics of gait patterns.

Parts of this work have been presented at **MHS  2021 (32nd 2021 International Symposium on Micro-NanoMechatronics and Human Science)** and **IROS 2022 (2022 IEEE/RSJ International Conference on Intelligent Robots and Systems)**.  
Related journal publications are available [Prediction of Whole-Body Velocity and Direction From Local Leg Joint Movements in Insect Walking via LSTM Neural Networks](https://ieeexplore.ieee.org/document/9832735).


## Dataset

- The dataset consists of joint angle sequences extracted frame-by-frame from cricket walking videos.
- Joint angles were obtained through pose estimation using **DeepLabCut**, a deep learning-based markerless tracking tool.
- Each sample contains:
  - Input: A time series of joint angles.
  - Output: Corresponding walking direction and velocity.


## Methods

Two main types of deep learning models are used:
- LSTM (Long Short-Term Memory Networks): Capture temporal correlations in gait patterns over time.
- Transformer Models: Leverage self-attention mechanisms to model complex temporal dependencies and predict motion features.

The models are trained to minimize the error between the predicted and true directions and velocities.
