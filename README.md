# Human-Protein-Atlas-Image-Classification

Kaggle project  https://www.kaggle.com/c/human-protein-atlas-image-classification

Predict the class for each image in the given set. Multiple labels can be predicted for each sample.

The results will be evaluated based on https://en.wikipedia.org/wiki/F1_score.

### Overview

#### Training
- Framework: Keras
- Model: Xception
- Data: Kaggle data
- Augmentation: rotation, shear, horizontal flip, vertical flip
- Optimizer: Adam
- Loss: Binary Cross Entropy 
- Learning rate: 0.001
- Image size: 299
- Batch size: 16
- Epochs: 25
- Training time: about 10 hours on Tesla P4

#### Prediction
- Threshold: search the threshold for each class 
- TTA(Test Time Augmentation) number: 16

#### Improvement
- Oversampling for rare class
- K-folds cross validation for optimizing the data splitting
