# Cricket Motion Prediction

This project focuses on predicting the locomotion direction and velocity of crickets based on joint angle sequences extracted from walking videos. Deep learning models, including LSTM and Transformer architectures, are employed to capture the temporal dynamics of gait patterns.


## Introduction

Understanding and predicting biological locomotion is a fundamental problem in biomechanics and robotics. This project aims to build predictive models that infer future walking directions and velocities of crickets using only their joint angle trajectories. The framework utilizes LSTM and Transformer-based sequence models to learn temporal dependencies in the gait data.


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
