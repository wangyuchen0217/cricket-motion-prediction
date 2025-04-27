# Cricket Motion Prediction

This project focuses on predicting the locomotion direction and velocity of crickets based on joint angle sequences extracted from walking videos. Deep learning models, including LSTM and Transformer architectures, are employed to capture the temporal dynamics of gait patterns.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methods](#methods)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Contact](#contact)

## Introduction

Understanding and predicting biological locomotion is a fundamental problem in biomechanics and robotics. This project aims to build predictive models that infer future walking directions and velocities of crickets using only their joint angle trajectories. The framework utilizes LSTM and Transformer-based sequence models to learn temporal dependencies in the gait data.

## Dataset

- The dataset consists of joint angle sequences extracted frame-by-frame from cricket walking videos.
- Each sample contains:
  - Input: A time series of joint angles.
  - Output: Corresponding walking direction and velocity.

*Note: Data pre-processing scripts for extracting joint angles are not included in this repository.*

## Methods

Two main types of deep learning models are used:
- LSTM (Long Short-Term Memory Networks): Capture temporal correlations in gait patterns over time.
- Transformer Models: Leverage self-attention mechanisms to model complex temporal dependencies and predict motion features.

The models are trained to minimize the error between the predicted and true directions and velocities.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/wangyuchen0217/cricket-motion-prediction.git
   cd cricket-motion-prediction
