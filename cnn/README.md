# CNN Hyperparameter Experimentation Script

This script systematically tests different CNN hyperparameters on the MNIST dataset to find the optimal configuration.

## 🎯 What This Script Does

1. **Automatically tests multiple configurations:**
   - Kernel sizes: [3, 5, 7]
   - Learning rates: [0.0001, 0.001, 0.01]
   - Number of convolutional layers: [2, 3]

2. **Generates comprehensive results:**
   - Training curves for each experiment
   - Comparison visualizations
   - Detailed report with all results
   - Best model saved as `.pth` file

3. **Total experiments:** 18 (3 kernel sizes × 3 learning rates × 2 layer configurations)
   - **Estimated runtime:** ~50-60 minutes on Google Colab with GPU
