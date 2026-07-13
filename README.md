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
├── results_table_generator.py     # Results visualization
├── run_fl_evaluation.ps1          # FL evaluation script
├── run_one_dataset.ps1            # Single dataset evaluation
├── run_comprehensive_evaluation.py # Comprehensive evaluation runner
├── run_complete_evaluation.py     # Complete evaluation runner
├── figs.py                        # Figure generation
├── METHODOLOGY_TIFS.md            # Detailed methodology documentation
├── FUTURE_WORK_ENHANCED.md         # Future work directions
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

| Dataset      | Test Acc | Attack Success | MI Bound |
|--------------|----------|----------------|----------|
| CIFAR10      | 91.54%   | 47.25%         | -0.00083 |
| CIFAR100     | 71.24%   | 50.00%         | -0.00012 |
| Purchase100  | 90.10%   | 48.99%         | -0.000007 |
| FaceScrub530 | 95.05%   | 47.00%         | 0.000035 |
| Texas100     | 87.08%   | 48.00%         | 0.000012 |
| Location     | 98.88%   | 47.75%         | -0.00021 |
| UTKFace      | 92.57%   | 47.24%         | -0.00169 |

### Federated Learning

| Dataset  | Mode    | Test Acc | Attack Success |
|----------|---------|----------|----------------|
| CIFAR10  | IID     | 85.64%   | 51.00%         |
| CIFAR10  | Non-IID | 82.52%   | 48.50%         |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{choto2024epurifier,
  title={E-PURIFIER: Internal-Validation-Based Stochastic Transformation for Reducing Membership Inference Distinguishability},
  author={Choto, Takudzwa and Huang, Xiaofang and Odoom, Justice and Min, Fan and Banda, William and Xiao, Ruifeng and Lei, Hongxia and Wang, Chengran and Seid, Muhammed Ahmed},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the PURIFIER framework by Yang et al.
- Extended with federated learning support and enhanced evaluation
