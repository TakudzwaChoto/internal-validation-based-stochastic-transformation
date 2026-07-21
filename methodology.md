# Methodology Details for Deep Learning 

## Network Topology

### Target Model Architectures

#### CNN Architecture (for CIFAR10, CIFAR100, UTKFace, FaceScrub530)
- **Type**: Convolutional Neural Network (CNN)
- **Loss Function**: Cross-Entropy Loss
- **Topology**:
  - Input Layer: 3x32x32 (CIFAR) or 3x64x64 (UTKFace/FaceScrub)
  - Conv1: 32 filters, 3x3 kernel, stride=1, padding=1
  - ReLU activation
  - MaxPool: 2x2 kernel, stride=2
  - Conv2: 64 filters, 3x3 kernel, stride=1, padding=1
  - ReLU activation
  - MaxPool: 2x2 kernel, stride=2
  - Conv3: 128 filters, 3x3 kernel, stride=1, padding=1
  - ReLU activation
  - MaxPool: 2x2 kernel, stride=2
  - Flatten
  - FC1: 256 units, ReLU activation
  - Dropout: 0.5
  - FC2: num_classes units (10 for CIFAR10, 100 for CIFAR100, 530 for FaceScrub, 5 for UTKFace)
  - Softmax output

#### MLP Architecture (for Purchase100, Texas100, Location)
- **Type**: Multi-Layer Perceptron (MLP)
- **Loss Function**: Cross-Entropy Loss
- **Topology**:
  - Input Layer: input_dim (varies by dataset)
  - FC1: 256 units, ReLU activation
  - Dropout: 0.3
  - FC2: 128 units, ReLU activation
  - Dropout: 0.3
  - FC3: num_classes units (100 for Purchase100/Texas100/Location)
  - Softmax output

### LSH+TinyMLP Detector Architecture
- **Type**: Locality-Sensitive Hashing + Tiny MLP
- **Loss Function**: Binary Cross-Entropy Loss
- **Topology**:
  - LSH Index: n_hash_tables=10, hash_size=16
  - TinyMLP:
    - Input Layer: 16 (hash size)
    - FC1: 64 units, ReLU activation
    - FC2: 32 units, ReLU activation
    - Output: 1 unit, Sigmoid activation

### Reference-Free CVAE Architecture
- **Type**: Conditional Variational Autoencoder
- **Loss Function**: Reconstruction Loss + KL Divergence Loss
- **Topology**:
  - Encoder:
    - Input: num_classes (confidence vector)
    - FC1: 128 units, ReLU activation
    - FC2: 64 units, ReLU activation
    - Mean: latent_dim=16 units
    - LogVar: latent_dim=16 units
  - Decoder:
    - Input: latent_dim=16 + condition (num_classes)
    - FC1: 64 units, ReLU activation
    - FC2: 128 units, ReLU activation
    - Output: num_classes (reconstructed confidence)

## Training Phase

### Target Model Training
- **Initialization**: Xavier/Glorot uniform initialization
- **Learning Algorithm**: Adam optimizer
- **Hyperparameters**:
  - Learning rate: 0.001
  - Beta1: 0.9
  - Beta2: 0.999
  - Epsilon: 1e-8
- **Mini-batches**: 64 samples per batch
- **Batch normalization**: Not used (standard batch processing)
- **Learning rate evolution**: Constant (no decay)
- **Number of epochs**: 40 (target_epochs)
- **Regularization**: Dropout (0.3-0.5 depending on architecture)
- **Data order**: Random shuffle between epochs
- **Stopping criterion**: Fixed number of epochs (40)
- **Hyperparameter search**: Manual tuning based on validation set performance

### LSH+TinyMLP Detector Training
- **Initialization**: Xavier/Glorot uniform initialization
- **Learning Algorithm**: Adam optimizer
- **Hyperparameters**:
  - Learning rate: 0.001
  - Beta1: 0.9
  - Beta2: 0.999
- **Mini-batches**: 32 samples per batch
- **Number of epochs**: 35 (cvae_epochs)
- **Regularization**: None
- **Data order**: Random shuffle between epochs
- **Stopping criterion**: Fixed number of epochs (35)

### Reference-Free CVAE Training
- **Initialization**: Xavier/Glorot uniform initialization
- **Learning Algorithm**: Adam optimizer
- **Hyperparameters**:
  - Learning rate: 0.001
  - Beta1: 0.9
  - Beta2: 0.999
  - KL weight: 0.5 (kl_weight)
- **Mini-batches**: 64 samples per batch
- **Number of epochs**: 35 (cvae_epochs)
- **Regularization**: KL divergence regularization (weight=0.5)
- **Data order**: Random shuffle between epochs
- **Stopping criterion**: Fixed number of epochs (35)

## Test Phase

### Data Splitting
- **Training set**: 80% of available data
- **Validation set**: 20% of training data (for defense training)
- **Test set**: Separate test set (standard dataset splits)
- **Cross-validation**: Not used (standard train/test split)

### Evaluation Protocol
- **Attack evaluation**: 9 different membership inference attacks
- **Member detection**: Binary classification accuracy
- **Utility analysis**: Accuracy comparison (defended vs undefended)
- **Statistical significance**: 5 runs with different random seeds for confidence intervals

## Implementation Details

### Hardware
- **GPU**: NVIDIA GPU (CUDA-compatible)
- **CPU**: Standard x86-64 processor
- **Memory**: 8GB+ RAM recommended

### Software Versions
- **Python**: 3.8+
- **PyTorch**: 1.9+
- **NumPy**: 1.19+
- **scikit-learn**: 0.24+
- **tqdm**: 4.60+
- **matplotlib**: 3.3+

### Tool-Specific Parameters
- **PyTorch**: Default parameters unless specified above
- **Random seed**: 42 (set_seed function)
- **Device**: CUDA if available, else CPU

### Preprocessing
- **Input normalization**: Standard normalization (mean=0.5, std=0.5 for images)
- **Output interpretation**: Softmax probabilities for classification
- **Confidence vector**: Model output probabilities (before defense)

### Scalability Techniques
- **LSH indexing**: O(1) approximate nearest neighbor search
- **Batch processing**: Efficient batch inference
- **Memory optimization**: Gradient checkpointing not used (not needed)

## Hyperparameter Tuning Policy

### Dataset-Specific Tuning
- **Defense profiles**: Per-dataset hyperparameters in Config.DEFENSE_PROFILES
- **Tuning method**: Manual grid search on validation set
- **Validation set**: 20% of training data
- **Tuning criteria**: Attack success rate minimization while maintaining utility

### Key Hyperparameters
- **swap_rate**: 0.70-0.99 (dataset-specific)
- **beta_alpha**: 0.35-0.90 (dataset-specific)
- **beta_beta**: 1.8-4.5 (dataset-specific)
- **conf_boost**: 0.10-0.60 (dataset-specific)
- **label_floor**: 0.25-0.40 (dataset-specific)
- **label_suppress**: 0.005-0.010 (dataset-specific)

## Statistical Significance

### Confidence Intervals
- **Number of runs**: 5 different random seeds
- **Metric reporting**: Mean ± standard deviation
- **Significance testing**: Paired t-test for defense vs no-defense comparison

### Results Validation
- **Reproducibility**: Code available on GitHub
- **Random seeds**: Fixed seed (42) for main results
- **Multiple runs**: Additional runs for statistical validation

## Additional Notes

### Federated Learning Extension
- **Aggregation algorithm**: FedAvg (Federated Averaging)
- **Communication rounds**: 10 (global_rounds)
- **Local epochs**: 5 (local_epochs)
- **Number of clients**: 10 (num_clients)
- **Data partitioning**: IID or Non-IID (Dirichlet alpha=0.5)

### Attack Implementations
- **Loss-based attack**: Negative log-likelihood as feature
- **Confidence-based attack**: Maximum confidence as feature
- **Entropy-based attack**: Prediction entropy as feature
- **Gradient-based attack**: Gradient norm approximation
- **Reference-based attack**: Distance to reference distribution
- **Boundary-distance attack**: Distance to decision boundary
- **Label-only attack**: Top-2 confidence ratio
- **Logit-based attack**: Logit values as features
- **Shadow-model attack**: Shadow model training
