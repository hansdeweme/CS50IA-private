Traffic Project - Experiments

Use multiple hidden layers and dropout layers for avoiding overfitting.
Use ReLU in convolutional and dense layers, avoid Sigmoid, which performs worse in deep networks (Sigmoid saturates for large values).
Include dropout layers with reasonable values (0.2, > 0.4 risks underfitting).
Use more convolutional layers and hidden layers to better learn complex patterns.
Add Batch Normalization which helps speed up training and stabilizes the learning process.
Use Data Augmentation which helps with generalization by applying transformations (rotation, zoom, etc.).



