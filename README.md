# Cricket Motion Prediction

This project focuses on predicting the locomotion direction and velocity of crickets based on joint angle sequences extracted from walking videos. Deep learning models, including **LSTM**, **Hammerstein-LSTM**, **ARX-LSTM**,  and **Transformer** architectures, are employed to capture the temporal dynamics of gait patterns.

Parts of this work have been presented at: 
- **MHS  2021 (32nd 2021 International Symposium on Micro-NanoMechatronics and Human Science)** ([Best Paper Award](https://www.mech.tohoku.ac.jp/news/prize/6200/))
- **IROS 2022 (2022 IEEE/RSJ International Conference on Intelligent Robots and Systems)**

Related journal publications are available:
- **IEEE Robotics and Automation Letters**: [Prediction of Whole-Body Velocity and Direction From Local Leg Joint Movements in Insect Walking via LSTM Neural Networks](https://ieeexplore.ieee.org/document/9832735).

![motion prediction framework](Evaluation/Figures/cricket-motion-prediction.gif)

## Dataset

- Joint positions were obtained from behavior videos through pose estimation using **[DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)**, a deep learning-based markerless tracking tool.
    - Joint position data was extracted by DeepLabCut (Jupyter Notebook)
    - Joint angle were calculated by running <pre> ```python skeleton_analysis.py``` </pre>
- Data preparation (including re-sample and smooth) <pre> ```python data_preprocess.py``` </pre>
- The dataset consists of joint angle sequences extracted frame-by-frame from cricket walking videos. <pre> ```python dataset_generate.py``` </pre> Each sample contains:
  - Input: A time series of joint angles.
  - Output: Corresponding walking direction and velocity. 

## Experiments

Train the models and evaluation.
```bash
python model_train.py
python model_evaluate.py
```

There are four types of deep learning time-series prediction neural networks available:

- **LSTM (Long Short-Term Memory Networks)**:  
  Capture temporal correlations in joint angle sequences over time. Suitable for modeling sequential dependencies in biological locomotion.

- **Hammerstein LSTM**:  
  A model that combines a nonlinear static transformation (Hammerstein block) followed by an LSTM layer. This structure enhances the ability to model complex nonlinearities in the relationship between joint angles and locomotion features.

- **ARX LSTM (AutoRegressive with eXogenous inputs LSTM)**:  
  Incorporates past outputs and external inputs (e.g., previous direction and velocity) as part of the LSTM inputs, improving temporal prediction by explicitly modeling auto-regressive dependencies.

- **Transformer Models**:  
  Leverage self-attention mechanisms to model complex and long-range temporal dependencies in the gait patterns, allowing better prediction of motion features even over extended sequences.

The models are trained to minimize the combined prediction error in walking direction and velocity using supervised learning.

## Dependencies

This project requires the following core Python packages:

- torch >= 1.11
- tensorflow >= 2.8
- numpy >= 1.22
- pandas >= 1.4
- matplotlib >= 3.5
- scikit-learn >= 1.0
- tqdm >= 4.60