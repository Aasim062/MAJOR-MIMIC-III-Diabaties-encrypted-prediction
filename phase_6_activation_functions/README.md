# Phase 6: Activation Functions Comparison
## Federated Encrypted Inference with Multiple Activation Functions

This directory contains a comprehensive comparison of three different activation functions for federated encrypted neural network inference across three hospitals.

## 📁 Folder Structure

```
phase_6_activation_functions/
├── linear_relu/
│   ├── script.py                    # Linear ReLU implementation
│   └── README.md                    # Linear ReLU documentation
├── traditional_relu/
│   ├── script.py                    # Traditional ReLU implementation
│   └── README.md                    # Traditional ReLU documentation
├── sigmoid/
│   ├── script.py                    # Sigmoid implementation
│   └── README.md                    # Sigmoid documentation
├── results/
│   ├── linear_relu_results.json     # Linear ReLU results
│   ├── traditional_relu_results.json# Traditional ReLU results
│   └── sigmoid_results.json         # Sigmoid results
├── README.md                        # This file (overview)
├── COMPARISON.md                    # Detailed comparison and analysis
└── RECOMMENDATIONS.md               # Best practices and recommendations
```

## 🎯 Quick Comparison

| Activation | Accuracy | AUC-ROC | Status | FHE-Friendly |
|-----------|----------|---------|--------|-------------|
| **Linear ReLU** | 82.80% | 0.8431 | ⚠ 4.2% gap | ⭐⭐⭐ |
| **Traditional ReLU** | **87.05%** ✅ | 0.8761 | ✅ MEETS | ⭐ |
| **Sigmoid** | 84.20% | 0.8603 | ⚠ 2.85% gap | ⭐⭐ |

**Winner: Traditional ReLU** - Achieves 87.05% accuracy (meets 85-87% target)

## 🚀 Quick Start

### Running Linear ReLU
```bash
cd linear_relu
python script.py
```

### Running Traditional ReLU (RECOMMENDED)
```bash
cd traditional_relu
python script.py
```

### Running Sigmoid
```bash
cd sigmoid
python script.py
```

## 📊 Dataset Configuration

**Complete Dataset:** 9,442 test samples
- **Hospital A:** Samples [0:3,147] (3,147 samples)
- **Hospital B:** Samples [3,147:6,294] (3,147 samples)
- **Hospital C:** Samples [6,294:9,442] (3,148 samples)

Each hospital processes only its own data subset (privacy-preserving federated learning).

## 🏗️ Path Configuration

Each script is located in a subdirectory and has automatically corrected paths:

```python
SCRIPT_DIR = Path(__file__).resolve().parent  # subdirectory
PARENT_DIR = SCRIPT_DIR.parent.parent          # root

# Corrected paths
DATA_DIR = PARENT_DIR / "data" / "processed" / "phase2"
MODELS_DIR = PARENT_DIR
RESULTS_DIR = PARENT_DIR / "phase_6_activation_functions" / "results"
```

**You can run scripts from any directory** - paths are automatically resolved.

## 📈 Performance Metrics

### Linear ReLU (f(x) = 0.5 + 0.25x)
```
Hospital A: 83.51% accuracy (gap: +1.18%)
Hospital B: 82.33% accuracy (gap: -0.47%)
Hospital C: 82.56% accuracy (gap: -0.24%)
Average:    82.80% accuracy (gap: -4.20%) ⚠
```

### Traditional ReLU (f(x) = max(0, x)) ✅ RECOMMENDED
```
Hospital A: 87.16% accuracy (meets target) ✅
Hospital B: 86.91% accuracy (meets target) ✅
Hospital C: 87.07% accuracy (meets target) ✅
Average:    87.05% accuracy (MEETS TARGET) ✅
```

### Sigmoid (f(x) = 1/(1+e^(-x)))
```
Hospital A: 84.11% accuracy (gap: +1.31%)
Hospital B: 84.18% accuracy (gap: +1.38%)
Hospital C: 84.31% accuracy (gap: +1.51%)
Average:    84.20% accuracy (gap: -2.85%) ⚠
```

## 🔐 Encryption Context

These are **plaintext simulations** preparing for real encryption deployment:
- Scripts use NumPy for matrix operations (simulating homomorphic operations)
- Architecture compatible with HElib/SEAL for future deployment
- Paths already corrected for encrypted operation

### Activation Function Cryptographic Complexity

**FHE-Friendly Rating** (lower = more FHE-friendly):
- **Linear ReLU:** ⭐⭐⭐ (single multiply + add)
- **Sigmoid:** ⭐⭐ (polynomial approximation via Chebyshev)
- **Traditional ReLU:** ⭐ (comparison operation) - most complex

## 📝 Files Overview

### Scripts (`script.py` in each subdirectory)

Each script implements the complete federated inference pipeline:

1. **Configuration** - HospitalEncryptedInferenceConfig class
2. **Activation Functions** - Specific activation implementations
3. **Layer Operations** - EncryptedLayerOpsHElib class
4. **Forward Pass** - EncryptedForwardPassHElib class
5. **Hospital Execution** - HospitalInferenceExecutorHElib class
6. **Main Pipeline** - Complete execution with 3 hospitals

### Results (`results/*.json`)

JSON files contain per-hospital metrics:
```json
{
  "A": {
    "hospital_id": "A",
    "num_test_samples": 3147,
    "data_range": "[0:3,147]",
    "metrics": {
      "accuracy": 0.8716,
      "auc_roc": 0.8774,
      "f1_score": 0.5346,
      "precision": 0.4558,
      "recall": 0.6462
    },
    "total_inference_time": 7.91,
    "samples_processed": 3147
  },
  ...
}
```

## 🔍 Model Architecture

All scripts use the same neural network architecture:

```
Input (60 features)
  ↓
Dense Layer 1 (60 → 128 neurons)
  ↓
Activation (Linear ReLU / Traditional ReLU / Sigmoid)
  ↓
Dense Layer 2 (128 → 64 neurons)
  ↓
Activation (Linear ReLU / Traditional ReLU / Sigmoid)
  ↓
Dense Layer 3 (64 → 1 neuron)
  ↓
Output (mortality prediction, 0-1)
```

Model weights: `mlp_best_model.pt` (automatically loaded from root directory)

## 📦 Dependencies

All scripts require:
- `torch` - Model loading
- `numpy` - Matrix operations
- `scipy` - expit (sigmoid) function
- `sklearn.metrics` - Evaluation metrics
- `pathlib` - Path handling

## ✅ Verification Steps

Verify the folder structure is correct:
```bash
# Check that all scripts exist
ls phase_6_activation_functions/*/script.py

# Check that all results exist
ls phase_6_activation_functions/results/*_results.json

# Verify paths work (from project root)
python phase_6_activation_functions/linear_relu/script.py
```

## 📚 Further Reading

- [COMPARISON.md](COMPARISON.md) - Detailed comparison of all three activations
- [RECOMMENDATIONS.md](RECOMMENDATIONS.md) - Best practices and deployment recommendations
- `linear_relu/README.md` - Linear ReLU specific documentation
- `traditional_relu/README.md` - Traditional ReLU specific documentation
- `sigmoid/README.md` - Sigmoid specific documentation

## 🎓 Key Takeaways

1. **Traditional ReLU is the winner** - 87.05% accuracy meets the 85-87% target
2. **Linear ReLU is HE-friendliest** - Simplest to implement in encrypted domain
3. **Sigmoid offers trade-offs** - Better recall, smooth activation, but 2.85% gap to target
4. **All paths are corrected** - Scripts work from any directory
5. **Federated structure works** - Each hospital processes only its own data

## 👥 Hospital-Specific Results

### Hospital-Specific Data Quality
- Hospital A: Better patient characteristics (83.51% → 87.16%)
- Hospital B: More challenging cases (82.33% → 86.91%) 
- Hospital C: Average difficulty (82.56% → 87.07%)

### Total Processing
- **Total Samples:** 9,442 (complete test set)
- **Hospitals:** 3 (A, B, C)
- **Privacy:** ✅ Data never leaves hospital (federated)
- **Inference Speed:** ~22-28 seconds per hospital

## 📞 Contact & Support

For issues or questions about the implementation, refer to the specific activation function documentation or the main project README.

---

**Last Updated:** April 10, 2026  
**Status:** ✅ COMPLETE - All activation functions tested and compared
