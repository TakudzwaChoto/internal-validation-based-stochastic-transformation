# E-PURIFIER: Internal-Validation-Based Stochastic Transformation for Reducing Membership Inference Distinguishability

Privacy Defense Framework against Membership Inference Attacks

## Overview

This repository implements E-PURIFIER, a comprehensive privacy defense framework that protects deployed machine learning models from membership inference attacks through efficient post-processing techniques. E-PURIFIER extends the original PURIFIER framework by addressing four critical limitations: excessive inference overhead (67×), dependency on external reference datasets, absence of quantifiable privacy diagnostics, and vulnerability to adaptive attacks.

## Features

- **LSH+TinyMLP Detector**: Efficient member detection using locality-sensitive hashing
- **Reference-Free CVAE**: Confidence transformation without requiring reference datasets
- **MI Bound Constraints**: Privacy-aware defense tuning using mutual information bounds
- **Federated Learning Support**: Extension for federated learning environments
- **Comprehensive Evaluation**: 9 different membership inference attack evaluations
- **Multi-Dataset Support**: CIFAR10, CIFAR100, Purchase100, Texas100, Location, FaceScrub530, UTKFace

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy 1.19+
- scikit-learn 0.24+
- tqdm 4.60+
- matplotlib 3.3+

### Setup

```bash
git clone https://github.com/TakudzwaChoto/Internal-Validation-Based-Stochastic-Transformation.git
cd Internal-Validation-Based-Stochastic-Transformation
pip install -r requirements.txt
```

## Usage

### Basic Evaluation

Run evaluation on a single dataset:

```powershell
.\run_one_dataset.ps1 cifar10
```

### Comprehensive Evaluation

Run comprehensive evaluation with multiple attacks:

```bash
python run_comprehensive_evaluation.py --dataset cifar10 --comprehensive
```

### Federated Learning Evaluation

Evaluate federated learning performance:

```powershell
.\run_fl_evaluation.ps1
```

### Complete Evaluation

Run both basic and comprehensive evaluation:

```bash
python run_complete_evaluation.py --dataset cifar10
```

## Project Structure

```
e-purifier/
├── purifier.py                    # Main implementation
├── comprehensive_evaluation.py    # Attack evaluation module
├── run_fl_evaluation.ps1          # FL evaluation script
├── run_one_dataset.ps1            # Single dataset evaluation
├── run_comprehensive_evaluation.py # Comprehensive evaluation runner
├── run_complete_evaluation.py     # Complete evaluation runner
├── methodology.md            # Detailed methodology documentation
├── future_work_enhanced.md         # Future work directions
└── README.md                      # This file
```

## Defense Components

### 1. LSH+TinyMLP Detector
- Uses locality-sensitive hashing for efficient member detection
- TinyMLP classifier for binary member/non-member classification
- 93% efficiency improvement over baseline methods

### 2. Reference-Free CVAE
- Conditional variational autoencoder for confidence transformation
- No reference dataset required
- Preserves model utility while reducing privacy leakage

### 3. MI Bound Constraints
- Mutual information estimation for privacy diagnostics
- Adaptive defense tuning based on MI bounds
- Quantifiable privacy guarantees

## Results

### Centralized Learning

| Dataset      | Test Acc | Attack Success | MI Bounds (nats) |
|--------------|----------|----------------|------------------|
| CIFAR10      | 91.54%   | 47.25%         | [8.33×10^-4, 1.02×10^-2] |
| CIFAR100     | 71.24%   | 50.00%         | [1.25×10^-4, 1.53×10^-3] |
| Purchase100  | 90.10%   | 48.99%         | [7.47×10^-6, 9.12×10^-5] |
| FaceScrub530 | 95.05%   | 47.00%         | [3.46×10^-5, 4.22×10^-4] |
| Texas100     | 87.08%   | 48.00%         | [1.19×10^-5, 1.45×10^-4] |
| Location     | 98.88%   | 47.75%         | [2.14×10^-4, 2.61×10^-3] |
| UTKFace      | 92.57%   | 47.24%         | [1.69×10^-3, 2.06×10^-2] |

### Federated Learning

| Dataset  | Mode    | Test Acc | Attack Success | Comm. Overhead |
|----------|---------|----------|----------------|----------------|
| CIFAR10  | IID     | 91.2%    | 47.8%          | 1.2×            |
| CIFAR10  | Non-IID | 90.8%    | 49.1%          | 1.3×            |


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the PURIFIER framework by Yang et al.
- Extended with federated learning support and enhanced evaluation

## Troubleshooting Guide for Reviewers

### Expected Result Variations

**Numerical Precision and Reproducibility**
- Results may vary slightly due to random seed initialization, even with fixed seeds
- Expected variance: ±0.5% for attack success rates, ±0.3% for test accuracy
- This is normal and within statistical significance thresholds
- All results in the paper are averaged over 5 runs with different seeds (42, 123, 456, 789, 101)

**Hardware Differences**
- GPU vs CPU training may cause minor numerical differences
- Different GPU architectures (RTX 4090 vs others) may produce slight variations
- Results should remain within confidence intervals reported in the paper

**Software Version Compatibility**
- PyTorch versions may cause slight numerical differences
- NumPy random number generator implementations vary across versions
- Use the exact versions specified in requirements.txt for best reproducibility

### Common Issues and Solutions

**Issue: Dataset Download Fails**
- Solution: Manually download datasets from official sources and place in `data/` directory
- CIFAR10/100: https://www.cs.toronto.edu/~kriz/cifar.html
- Purchase100: Kaggle Acquire Valued Shoppers Challenge
- FaceScrub530: https://vintage.winklerbros.net/facescrub.html

**Issue: Out of Memory Errors**
- Solution: Reduce batch size in Config class (line 70-89 in purifier.py)
- Default batch size: 128, try 64 or 32 for smaller GPUs
- Ensure at least 8GB GPU memory for image datasets

**Issue: Training Takes Too Long**
- Solution: Reduce number of epochs in Config class
- Target model epochs: 40 (try 20 for faster testing)
- CVAE epochs: 35 (try 20 for faster testing)
- Note: Results may vary with fewer epochs

**Issue: Attack Success Rate Differs Significantly**
- Check that the random seed is set correctly (default: 42)
- Verify dataset split matches paper (80/20 train/test)
- Ensure defense hyperparameters match DEFENSE_PROFILES in Config
- Try running multiple times and averaging results

**Issue: Federated Learning Errors**
- Ensure all clients have sufficient data (minimum 100 samples per client)
- For Non-IID partitioning, alpha=0.5 is recommended
- If batch norm errors occur, increase local batch size to 64

**Issue: Import Errors**
- Solution: Install all dependencies: `pip install -r requirements.txt`
- Ensure Python version is 3.8 or higher
- Check that PyTorch is installed with CUDA support if using a GPU

### Reproducibility Checklist

To reproduce results exactly as in the paper:

1. **Environment Setup**
   - Python 3.8+
   - PyTorch 1.9+
   - CUDA-compatible GPU (recommended)
   - 8GB+ RAM

2. **Configuration**
   - Use default hyperparameters in Config.DEFENSE_PROFILES
   - Set random seed to 42 (default in set_seed function)
   - Use default batch sizes and epochs

3. **Data Preparation**
   - Use standard dataset splits (80% train, 20% test)
   - For defense training, use 10% of training data as validation
   - Ensure data preprocessing matches paper (normalization, augmentation)

4. **Running Experiments**
   - Run each experiment 5 times with different seeds
   - Report mean ± standard deviation
   - Compare against confidence intervals in paper

### Contact for Reproducibility Issues

If you encounter issues not covered in this guide, please contact:
- Email: ctakudzwa95@gmail.com
- GitHub Issues: https://github.com/TakudzwaChoto/Internal-Validation-Based-Stochastic-Transformation/issues

We are committed to ensuring full reproducibility of our results and will provide additional assistance as needed.
