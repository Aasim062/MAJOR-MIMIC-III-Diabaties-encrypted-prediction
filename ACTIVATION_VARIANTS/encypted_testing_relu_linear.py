# # phase_6c_end_to_end_encrypted_inference_EXTREME_FIX.py
# """
# Phase 6c: EXTREME Magnitude Control to Eliminate Scale Overflow
# ===============================================================

# The key insight: We need EXTREME aggressive alpha to prevent overflow!
# Not just α=0.01, but α=0.001 or even smaller!
# """

# import torch
# import tenseal as ts
# import numpy as np
# import json
# import time
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# from datetime import datetime
# from scipy.special import expit
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from sklearn.metrics import precision_score, recall_score
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import warnings
# warnings.filterwarnings('ignore')


# # ============================================================================
# # CONFIGURATION - EXTREME MAGNITUDE CONTROL
# # ============================================================================

# class HospitalEncryptedInferenceConfig:
#     """
#     EXTREME magnitude control to prevent ALL overflow errors.
    
#     ⭐ KEY INSIGHT:
#     Alpha values must be EXTREMELY small to prevent scale overflow.
#     Not α=0.5, not α=0.01, but α=0.001 or smaller!
#     """
    
#     ENCRYPTED_DIR = Path("encrypted")
#     MODELS_DIR = Path(".")
#     DATA_DIR = Path("data/processed/phase2")
#     REPORTS_DIR = Path("reports")
    
#     HOSPITALS = ['A', 'B', 'C']
    
#     # Model architecture - ORIGINAL SIZES WITH 60 INPUT FEATURES
#     INPUT_DIM = 60          # ✅ 60 input features (data has 60)
#     HIDDEN_DIM_1 = 128      # 128 hidden neurons (original size)
#     HIDDEN_DIM_2 = 64       # 64 hidden neurons (original size)
#     OUTPUT_DIM = 1          # 1 output neuron
    
#     # Inference parameters
#     BATCH_SIZE = 1
#     SCALE = 2**30
#     POLY_MOD_DEGREE = 8192
    
#     # ⭐⭐⭐ EXTREME ALPHA VALUES - PREVENT ALL OVERFLOW! ⭐⭐⭐
#     # These are 1000× smaller than 0.5
#     LINEAR_RELU_ALPHA_L1 = 0.001   # 1000× reduction!
#     LINEAR_RELU_ALPHA_L2 = 0.005   # 200× reduction
#     LINEAR_RELU_ALPHA_L3 = 1.0     # Output (no scaling needed)
    
#     # Additional magnitude control
#     INPUT_SCALE_FACTOR = 0.1       # Scale input by 0.1 (additional 10× reduction)
    
#     # Device settings
#     PREFERRED_DEVICE = 'cuda'
#     DEVICE = 'cpu'
    
#     # Inference settings
#     PREDICTION_THRESHOLD = 0.5
    
#     # Error handling
#     SKIP_FAILED_SAMPLES = True
#     MAX_SAMPLES = None  # None = all, or limit for testing
    
#     # Logging
#     VERBOSE = True
    
#     def __post_init__(self):
#         if self.PREFERRED_DEVICE == 'cuda' and torch.cuda.is_available():
#             self.DEVICE = 'cuda'
#         else:
#             self.DEVICE = 'cpu'
#         self.REPORTS_DIR.mkdir(exist_ok=True)
#         self.ENCRYPTED_DIR.mkdir(exist_ok=True)


# # ============================================================================
# # LINEAR ReLU WITH EXTREME MAGNITUDE CONTROL
# # ============================================================================

# class LinearReLU:
#     """Linear ReLU with extreme magnitude scaling"""
    
#     @staticmethod
#     def eval_plaintext(x: np.ndarray, alpha: float = 0.001) -> np.ndarray:
#         """Evaluate Linear ReLU on plaintext"""
#         return alpha * x
    
#     @staticmethod
#     def eval_encrypted(ct_x: ts.CKKSVector, alpha: float = 0.001) -> ts.CKKSVector:
#         """Evaluate Linear ReLU on encrypted (homomorphic)"""
#         return ct_x * alpha


# # ============================================================================
# # ENCRYPTED OPERATIONS
# # ============================================================================

# class EncryptedLayerOps:
#     """Homomorphic operations with extreme magnitude control"""
    
#     @staticmethod
#     def encrypt_sample(sample: np.ndarray, context: ts.Context) -> ts.CKKSVector:
#         """Encrypt single sample"""
#         return ts.ckks_vector(context, sample.tolist())
    
#     @staticmethod
#     def encrypted_linear_layer(
#         ct_x: ts.CKKSVector,
#         W: np.ndarray,
#         b: np.ndarray,
#         scale_output: float = 1.0,
#         layer_name: str = "Linear"
#     ) -> List[ts.CKKSVector]:
#         """
#         Homomorphic linear layer with output scaling.
        
#         ⭐ NEW: scale_output parameter to reduce output magnitude
#         """
        
#         output_dim, input_dim = W.shape
#         ct_output = []
        
#         for i in range(output_dim):
#             # ct_y[i] = scale_output × (∑ⱼ W[i,j] * ct_x[j] + b[i])
            
#             w_i0 = float(W[i, 0].item() if hasattr(W[i, 0], 'item') else W[i, 0])
#             ct_y_i = ct_x * (w_i0 * scale_output)
            
#             for j in range(1, input_dim):
#                 w_ij = float(W[i, j].item() if hasattr(W[i, j], 'item') else W[i, j])
#                 ct_y_i = ct_y_i + (ct_x * (w_ij * scale_output))
            
#             b_i = float(b[i].item() if hasattr(b[i], 'item') else b[i])
#             ct_y_i = ct_y_i + (b_i * scale_output)
            
#             ct_output.append(ct_y_i)
        
#         return ct_output
    
#     @staticmethod
#     def encrypted_linear_relu_layer(
#         ct_z_list: List[ts.CKKSVector],
#         alpha: float = 0.001,
#         layer_name: str = "LinearReLU"
#     ) -> List[ts.CKKSVector]:
#         """Homomorphic Linear ReLU with extreme scaling"""
        
#         ct_a_list = []
#         for ct_z in ct_z_list:
#             ct_a = LinearReLU.eval_encrypted(ct_z, alpha=alpha)
#             ct_a_list.append(ct_a)
        
#         return ct_a_list


# # ============================================================================
# # ENCRYPTED FORWARD PASS
# # ============================================================================

# class EncryptedForwardPassHospital:
#     """Execute encrypted forward pass"""
    
#     @staticmethod
#     def load_model(config: HospitalEncryptedInferenceConfig) -> Dict:
#         """Load model weights"""
        
#         if config.VERBOSE:
#             print(f"\n[Loading Model]")
        
#         model_path = config.MODELS_DIR / "mlp_best_model_REDUCED.pt"
        
#         if not model_path.exists():
#             model_path = config.MODELS_DIR / "mlp_best_model.pt"
        
#         checkpoint = torch.load(model_path, map_location=config.DEVICE)
        
#         if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#             state_dict = checkpoint['state_dict']
#         elif isinstance(checkpoint, dict):
#             state_dict = checkpoint
#         else:
#             state_dict = checkpoint.state_dict()
        
#         weights = {
#             'fc1_weight': None, 'fc1_bias': None,
#             'fc2_weight': None, 'fc2_bias': None,
#             'fc3_weight': None, 'fc3_bias': None
#         }
        
#         for name, param in state_dict.items():
#             if 'fc1.weight' in name:
#                 weights['fc1_weight'] = param.cpu().numpy()
#             elif 'fc1.bias' in name:
#                 weights['fc1_bias'] = param.cpu().numpy()
#             elif 'fc2.weight' in name:
#                 weights['fc2_weight'] = param.cpu().numpy()
#             elif 'fc2.bias' in name:
#                 weights['fc2_bias'] = param.cpu().numpy()
#             elif 'fc3.weight' in name:
#                 weights['fc3_weight'] = param.cpu().numpy()
#             elif 'fc3.bias' in name:
#                 weights['fc3_bias'] = param.cpu().numpy()
        
#         if config.VERBOSE:
#             print(f"  ✓ Loaded successfully")
#             print(f"    Alpha L1: {config.LINEAR_RELU_ALPHA_L1} (extreme)")
#             print(f"    Alpha L2: {config.LINEAR_RELU_ALPHA_L2}")
#             print(f"    Input scale: {config.INPUT_SCALE_FACTOR}×")
#             print(f"    L1 weight shape: {weights['fc1_weight'].shape}")
#             print(f"    L2 weight shape: {weights['fc2_weight'].shape}")
#             print(f"    L3 weight shape: {weights['fc3_weight'].shape}")
        
#         return weights
    
#     @staticmethod
#     def encrypted_forward(
#         ct_X: List[ts.CKKSVector],
#         weights: Dict,
#         config: HospitalEncryptedInferenceConfig
#     ) -> Tuple[Optional[List[ts.CKKSVector]], Dict]:
#         """
#         Execute encrypted forward pass with extreme magnitude control.
        
#         Returns: (ct_logits, timings) or (None, {}) if failed
#         """
        
#         timings = {}
        
#         try:
#             # ===== LAYER 1: 60 → 128 =====
#             t_start = time.time()
            
#             # Scale input by additional factor to reduce magnitude
#             ct_z1_list = EncryptedLayerOps.encrypted_linear_layer(
#                 ct_X[0], 
#                 weights['fc1_weight'], 
#                 weights['fc1_bias'],
#                 scale_output=config.INPUT_SCALE_FACTOR,  # ⭐ Additional scaling
#                 layer_name="L1"
#             )
            
#             ct_a1_list = EncryptedLayerOps.encrypted_linear_relu_layer(
#                 ct_z1_list, 
#                 alpha=config.LINEAR_RELU_ALPHA_L1,
#                 layer_name="L1_relu"
#             )
            
#             timings['layer_1'] = time.time() - t_start
            
#             # ===== LAYER 2: 128 → 64 =====
#             t_start = time.time()
            
#             ct_z2_list = EncryptedLayerOps.encrypted_linear_layer(
#                 ct_a1_list[0], 
#                 weights['fc2_weight'], 
#                 weights['fc2_bias'],
#                 scale_output=1.0,  # No additional scaling
#                 layer_name="L2"
#             )
            
#             ct_a2_list = EncryptedLayerOps.encrypted_linear_relu_layer(
#                 ct_z2_list, 
#                 alpha=config.LINEAR_RELU_ALPHA_L2,
#                 layer_name="L2_relu"
#             )
            
#             timings['layer_2'] = time.time() - t_start
            
#             # ===== LAYER 3: 64 → 1 =====
#             t_start = time.time()
            
#             ct_logits_list = EncryptedLayerOps.encrypted_linear_layer(
#                 ct_a2_list[0], 
#                 weights['fc3_weight'], 
#                 weights['fc3_bias'],
#                 scale_output=1.0,  # No scaling for output
#                 layer_name="L3"
#             )
            
#             timings['layer_3'] = time.time() - t_start
            
#             return ct_logits_list, timings
            
#         except Exception as e:
#             return None, {}


# # ============================================================================
# # HOSPITAL INFERENCE EXECUTOR
# # ============================================================================

# class HospitalInferenceExecutor:
#     """Execute encrypted inference for hospital"""
    
#     @staticmethod
#     def run_hospital_inference(
#         hospital_id: str,
#         config: HospitalEncryptedInferenceConfig
#     ) -> Optional[Dict]:
#         """Complete encrypted inference"""
        
#         if config.VERBOSE:
#             print(f"\n" + "=" * 100)
#             print(f"HOSPITAL {hospital_id}: ENCRYPTED INFERENCE (EXTREME MAGNITUDE CONTROL)")
#             print(f"=" * 100)
        
#         results = {
#             'hospital_id': hospital_id,
#             'config': {
#                 'alpha_l1': config.LINEAR_RELU_ALPHA_L1,
#                 'alpha_l2': config.LINEAR_RELU_ALPHA_L2,
#                 'input_scale': config.INPUT_SCALE_FACTOR,
#                 'input_dim': config.INPUT_DIM,
#                 'hidden_dim_1': config.HIDDEN_DIM_1,
#                 'hidden_dim_2': config.HIDDEN_DIM_2,
#             }
#         }
        
#         # Load context
#         if config.VERBOSE:
#             print(f"\n[Step 1: Load TenSEAL Context]")
        
#         context_path = config.ENCRYPTED_DIR / "context.bin"
#         try:
#             context = ts.context_from(open(str(context_path), 'rb').read())
#             if config.VERBOSE:
#                 print(f"  ✓ Context loaded")
#         except Exception as e:
#             print(f"  ❌ Error: {e}")
#             return None
        
#         # Load test data
#         if config.VERBOSE:
#             print(f"\n[Step 2: Load Test Data]")
        
#         try:
#             X_test = np.load(config.DATA_DIR / "X_test.npy")
#             y_test = np.load(config.DATA_DIR / "y_test.npy")
            
#             if config.VERBOSE:
#                 print(f"  ✓ Loaded {len(X_test):,} samples")
#                 print(f"    Features: {X_test.shape[1]}")
            
#             results['num_test_samples'] = len(X_test)
            
#             if config.MAX_SAMPLES:
#                 X_test = X_test[:config.MAX_SAMPLES]
#                 y_test = y_test[:config.MAX_SAMPLES]
#                 if config.VERBOSE:
#                     print(f"  ⓘ Limited to {len(X_test)} samples")
            
#         except Exception as e:
#             print(f"  ❌ Error: {e}")
#             return None
        
#         # Normalize input
#         if config.VERBOSE:
#             print(f"\n[Step 3: Normalize Input]")
        
#         scaler = StandardScaler()
#         X_test = scaler.fit_transform(X_test)
        
#         # Additional scaling to reduce magnitude
#         X_test = X_test * config.INPUT_SCALE_FACTOR
        
#         if config.VERBOSE:
#             print(f"  ✓ Normalized and scaled by {config.INPUT_SCALE_FACTOR}×")
        
#         # Load model
#         if config.VERBOSE:
#             print(f"\n[Step 4: Load Model]")
        
#         try:
#             weights = EncryptedForwardPassHospital.load_model(config)
#         except Exception as e:
#             print(f"  ❌ Error: {e}")
#             return None
        
#         # Load torch weights for plaintext
#         torch_device = torch.device(config.DEVICE)
#         torch_weights = {
#             'fc1_weight': torch.from_numpy(weights['fc1_weight']).float().to(torch_device),
#             'fc1_bias': torch.from_numpy(weights['fc1_bias']).float().to(torch_device),
#             'fc2_weight': torch.from_numpy(weights['fc2_weight']).float().to(torch_device),
#             'fc2_bias': torch.from_numpy(weights['fc2_bias']).float().to(torch_device),
#             'fc3_weight': torch.from_numpy(weights['fc3_weight']).float().to(torch_device),
#             'fc3_bias': torch.from_numpy(weights['fc3_bias']).float().to(torch_device),
#         }
        
#         # Encrypted inference
#         if config.VERBOSE:
#             print(f"\n[Step 5: Encrypted Inference]")
#             print(f"  Processing {len(X_test)} samples...")
        
#         all_logits_encrypted = []
#         all_logits_plaintext = []
#         samples_success = 0
#         samples_failed = 0
        
#         for sample_idx in range(len(X_test)):
            
#             # Progress
#             if config.VERBOSE and (sample_idx + 1) % 100 == 0:
#                 print(f"  Progress: {sample_idx + 1}/{len(X_test)} "
#                       f"(✓ {samples_success}, ✗ {samples_failed})")
            
#             X_sample = X_test[sample_idx:sample_idx+1]
            
#             # Encrypt
#             try:
#                 ct_X = [EncryptedLayerOps.encrypt_sample(x, context) for x in X_sample]
#             except Exception as e:
#                 if config.SKIP_FAILED_SAMPLES:
#                     samples_failed += 1
#                     continue
#                 else:
#                     raise
            
#             # Forward pass
#             try:
#                 ct_logits, _ = EncryptedForwardPassHospital.encrypted_forward(
#                     ct_X, weights, config
#                 )
                
#                 if ct_logits is None:
#                     if config.SKIP_FAILED_SAMPLES:
#                         samples_failed += 1
#                         continue
#                     else:
#                         raise ValueError("Forward pass failed")
                
#             except Exception as e:
#                 if config.SKIP_FAILED_SAMPLES:
#                     samples_failed += 1
#                     continue
#                 else:
#                     raise
            
#             # Decrypt
#             try:
#                 logit_val = float(ct_logits[0].decrypt()[0])
#                 # Undo the scaling we applied in Layer 1
#                 logit_val = logit_val / config.INPUT_SCALE_FACTOR
                
#                 all_logits_encrypted.append(logit_val)
#                 samples_success += 1
                
#             except Exception as e:
#                 if config.SKIP_FAILED_SAMPLES:
#                     samples_failed += 1
#                     continue
#                 else:
#                     raise
            
#             # Plaintext comparison (first 50 samples only)
#             if sample_idx < 50:
#                 with torch.no_grad():
#                     X_torch = torch.from_numpy(X_sample).float().to(torch_device)
#                     X_torch_scaled = X_torch * config.INPUT_SCALE_FACTOR
                    
#                     z1 = torch.mm(X_torch_scaled, torch_weights['fc1_weight'].T) + torch_weights['fc1_bias']
#                     a1 = config.LINEAR_RELU_ALPHA_L1 * z1
                    
#                     z2 = torch.mm(a1, torch_weights['fc2_weight'].T) + torch_weights['fc2_bias']
#                     a2 = config.LINEAR_RELU_ALPHA_L2 * z2
                    
#                     logits_plain = torch.mm(a2, torch_weights['fc3_weight'].T) + torch_weights['fc3_bias']
#                     logits_plain = logits_plain / config.INPUT_SCALE_FACTOR
                    
#                     all_logits_plaintext.append(float(logits_plain.item()))
        
#         if config.VERBOSE:
#             print(f"\n  ✓ Complete")
#             print(f"    Successful: {samples_success}/{len(X_test)} ({100*samples_success/len(X_test):.1f}%)")
#             print(f"    Failed: {samples_failed}/{len(X_test)}")
        
#         if samples_success == 0:
#             print(f"  ❌ No successful samples!")
#             return None
        
#         # Evaluate
#         if config.VERBOSE:
#             print(f"\n[Step 6: Evaluate]")
        
#         logits_encrypted = np.array(all_logits_encrypted)
#         y_test_subset = y_test[:len(all_logits_encrypted)]
        
#         y_prob_encrypted = expit(logits_encrypted)
#         y_pred_encrypted = (y_prob_encrypted >= config.PREDICTION_THRESHOLD).astype(int)
        
#         try:
#             metrics = {
#                 'accuracy': float(accuracy_score(y_test_subset, y_pred_encrypted)),
#                 'auc_roc': float(roc_auc_score(y_test_subset, y_prob_encrypted)),
#                 'f1': float(f1_score(y_test_subset, y_pred_encrypted, zero_division=0)),
#                 'precision': float(precision_score(y_test_subset, y_pred_encrypted, zero_division=0)),
#                 'recall': float(recall_score(y_test_subset, y_pred_encrypted, zero_division=0)),
#             }
            
#             if config.VERBOSE:
#                 print(f"\n  Performance (on {samples_success} successful samples):")
#                 print(f"    Accuracy:  {metrics['accuracy']:.4f}")
#                 print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
#                 print(f"    F1-Score:  {metrics['f1']:.4f}")
#                 print(f"    Precision: {metrics['precision']:.4f}")
#                 print(f"    Recall:    {metrics['recall']:.4f}")
            
#         except Exception as e:
#             print(f"  ⓘ Metrics error: {e}")
#             metrics = {}
        
#         results['metrics'] = metrics
#         results['samples_success'] = samples_success
#         results['samples_failed'] = samples_failed
        
#         return results


# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     """Main execution"""
    
#     print("\n" + "=" * 100)
#     print("PHASE 6c: ENCRYPTED INFERENCE")
#     print("EXTREME Magnitude Control to Eliminate ALL Overflow Errors")
#     print("=" * 100)
    
#     config = HospitalEncryptedInferenceConfig()
#     config.__post_init__()
    
#     print(f"\n⭐ EXTREME CONFIGURATION:")
#     print(f"   Input Features: {config.INPUT_DIM}")
#     print(f"   Architecture: {config.INPUT_DIM} → {config.HIDDEN_DIM_1} → {config.HIDDEN_DIM_2} → {config.OUTPUT_DIM}")
#     print(f"   Alpha L1: {config.LINEAR_RELU_ALPHA_L1} (EXTREME!)")
#     print(f"   Alpha L2: {config.LINEAR_RELU_ALPHA_L2}")
#     print(f"   Input scale: {config.INPUT_SCALE_FACTOR}×")
#     print(f"   Batch size: {config.BATCH_SIZE}")
    
#     all_results = {}
    
#     for hospital_id in config.HOSPITALS:
#         try:
#             results = HospitalInferenceExecutor.run_hospital_inference(hospital_id, config)
#             if results:
#                 all_results[hospital_id] = results
#                 print(f"\n✅ Hospital {hospital_id}: SUCCESS!")
#         except Exception as e:
#             print(f"\n❌ Hospital {hospital_id}: {e}")
    
#     # Save
#     output_path = config.ENCRYPTED_DIR / "phase_6c_extreme_results.json"
#     with open(output_path, 'w') as f:
#         json.dump(all_results, f, indent=2)
#     print(f"\n✓ Results saved to {output_path}")
    
#     print("\n" + "=" * 100)
#     print(f"Hospitals processed: {list(all_results.keys())}")
#     print("=" * 100 + "\n")


# if __name__ == "__main__":
#     main()



# phase_6_end_to_end_encrypted_inference_complete.py
"""
Phase 6: End-to-End Encrypted Inference (Hospital-Specific Test Data)
======================================================================
Each hospital:
1. Encrypts their OWN local test data
2. Passes through ENCRYPTED global model (from Phase 5)
3. All computation happens in encrypted domain
4. NO intermediate decryption (pure homomorphic inference)
5. Decrypt predictions only at the end
"""

import torch
import tenseal as ts
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from scipy.special import expit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class HospitalEncryptedInferenceConfig:
    """Configuration for hospital-specific encrypted inference"""
    
    ENCRYPTED_DIR = Path("encrypted")
    MODELS_DIR = Path(".")
    DATA_DIR = Path("data/processed/phase2")
    REPORTS_DIR = Path("reports")
    
    HOSPITALS = ['A', 'B', 'C']
    
    # Model architecture (requested)
    INPUT_DIM = 60
    HIDDEN_DIM_1 = 128
    HIDDEN_DIM_2 = 64
    OUTPUT_DIM = 1
    
    # Inference parameters
    BATCH_SIZE = 1
    SCALE = 2**30
    POLY_MOD_DEGREE = 8192
    
    # ReLU approximation (degree-1 linear)
    RELU_POLY_DEGREE = 1
    RELU_BOUND = 2.0
    
    # Scale control after layer 1 (fix for scale out of bounds)
    LAYER1_SCALE_DOWN = 0.01
    
    PREFERRED_DEVICE = 'cuda' 
    DEVICE = 'cpu'
    PREDICTION_THRESHOLD = 0.5
    
    def __post_init__(self):
        if self.PREFERRED_DEVICE == 'cuda' and torch.cuda.is_available():
            self.DEVICE = 'cuda'
        else:
            self.DEVICE = 'cpu'
        self.REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# CHEBYSHEV ReLU (Degree-1 - Linear Approximation)
# ============================================================================

class ChebyshevReLUv2:
    """Degree-1 Chebyshev polynomial for ReLU (NO squaring, avoids scale overflow)"""
    
    @staticmethod
    def get_coefficients(bound: float = 2.0) -> np.ndarray:
        if bound == 2.0:
            return np.array([0.5, 0.25])  # [c0, c1]
        else:
            base = np.array([0.5, 0.25])
            scale = bound / 2.0
            return np.array([base[0], base[1] * scale])
    
    @staticmethod
    def eval_plaintext(x: np.ndarray, bound: float = 2.0) -> np.ndarray:
        coeffs = ChebyshevReLUv2.get_coefficients(bound)
        return coeffs[0] + coeffs[1] * x
    
    @staticmethod
    def eval_encrypted(ct_x: ts.CKKSVector, bound: float = 2.0) -> ts.CKKSVector:
        coeffs = ChebyshevReLUv2.get_coefficients(bound)
        ct_result = ct_x * coeffs[1]
        ct_result = ct_result + coeffs[0]
        return ct_result


# ============================================================================
# ENCRYPTED LAYER OPERATIONS
# ============================================================================

class EncryptedLayerOps:
    """Homomorphic operations for neural network layers"""
    
    @staticmethod
    def encrypt_sample(sample: np.ndarray, context: ts.Context) -> ts.CKKSVector:
        return ts.ckks_vector(context, sample.tolist())
    
    @staticmethod
    def encrypted_linear_layer(
        ct_x: ts.CKKSVector,
        W: np.ndarray,
        b: np.ndarray,
        layer_name: str = "Linear"
    ) -> List[ts.CKKSVector]:
        output_dim, input_dim = W.shape
        ct_output = []
        
        for i in range(output_dim):
            w_i0 = float(W[i, 0].item() if hasattr(W[i, 0], 'item') else W[i, 0])
            ct_y_i = ct_x * w_i0
            
            for j in range(1, input_dim):
                w_ij = float(W[i, j].item() if hasattr(W[i, j], 'item') else W[i, j])
                ct_y_i = ct_y_i + (ct_x * w_ij)
            
            b_i = float(b[i].item() if hasattr(b[i], 'item') else b[i])
            ct_y_i = ct_y_i + b_i
            ct_output.append(ct_y_i)
        
        return ct_output
    
    @staticmethod
    def encrypted_relu_layer(
        ct_z_list: List[ts.CKKSVector],
        layer_name: str = "ReLU"
    ) -> List[ts.CKKSVector]:
        
        ct_a_list = []
        for ct_z in ct_z_list:
            ct_a = ChebyshevReLUv2.eval_encrypted(ct_z, bound=2.0)
            ct_a_list.append(ct_a)
        
        return ct_a_list


# ============================================================================
# ENCRYPTED FORWARD PASS
# ============================================================================

class EncryptedForwardPassHospital:
    """Execute encrypted forward pass for hospital test data"""
    
    @staticmethod
    def load_encrypted_model(config: HospitalEncryptedInferenceConfig) -> Dict:
        print(f"\n[Loading Encrypted Global Model]")
        
        # Prefer reduced model if it exists
        model_path_reduced = config.MODELS_DIR / "mlp_best_model_REDUCED.pt"
        model_path_original = config.MODELS_DIR / "mlp_best_model.pt"
        model_path = model_path_reduced if model_path_reduced.exists() else model_path_original
        
        print(f"  Loading plaintext model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()
        
        weights = {
            'fc1_weight': None,
            'fc1_bias': None,
            'fc2_weight': None,
            'fc2_bias': None,
            'fc3_weight': None,
            'fc3_bias': None
        }
        
        for name, param in state_dict.items():
            if 'fc1.weight' in name:
                weights['fc1_weight'] = param.cpu().numpy()
            elif 'fc1.bias' in name:
                weights['fc1_bias'] = param.cpu().numpy()
            elif 'fc2.weight' in name:
                weights['fc2_weight'] = param.cpu().numpy()
            elif 'fc2.bias' in name:
                weights['fc2_bias'] = param.cpu().numpy()
            elif 'fc3.weight' in name:
                weights['fc3_weight'] = param.cpu().numpy()
            elif 'fc3.bias' in name:
                weights['fc3_bias'] = param.cpu().numpy()
        
        print(f"  ✓ Model loaded (plaintext weights)")
        print(f"    L1 weight shape: {weights['fc1_weight'].shape}")
        print(f"    L2 weight shape: {weights['fc2_weight'].shape}")
        print(f"    L3 weight shape: {weights['fc3_weight'].shape}")
        
        return weights
    
    @staticmethod
    def encrypted_forward(
        ct_X: List[ts.CKKSVector],
        weights: Dict,
        sample_indices: np.ndarray,
        config: HospitalEncryptedInferenceConfig
    ) -> Tuple[List[ts.CKKSVector], Dict]:
        
        print(f"\n[Encrypted Forward Pass]")
        print(f"  Samples: {len(ct_X)}")
        
        layer_timings = {}
        total_time = 0
        
        # ===== LAYER 1: 60 → 128 =====
        print(f"\n  Layer 1 (60 → 128):")
        t_start = time.time()
        
        ct_z1_list = EncryptedLayerOps.encrypted_linear_layer(
            ct_X[0], weights['fc1_weight'], weights['fc1_bias'], "L1_linear"
        )
        
        ct_a1_list = EncryptedLayerOps.encrypted_relu_layer(ct_z1_list, "L1_relu")
        
        # ✅ SCALE DOWN AFTER LAYER 1 (FIX)
        ct_a1_list = [ct * config.LAYER1_SCALE_DOWN for ct in ct_a1_list]
        
        t_l1 = time.time() - t_start
        layer_timings['layer_1'] = t_l1
        total_time += t_l1
        print(f"    Time: {t_l1:.3f} sec")
        
        # ===== LAYER 2: 128 → 64 =====
        print(f"\n  Layer 2 (128 → 64):")
        t_start = time.time()
        
        ct_z2_list = EncryptedLayerOps.encrypted_linear_layer(
            ct_a1_list[0], weights['fc2_weight'], weights['fc2_bias'], "L2_linear"
        )
        
        ct_a2_list = EncryptedLayerOps.encrypted_relu_layer(ct_z2_list, "L2_relu")
        
        t_l2 = time.time() - t_start
        layer_timings['layer_2'] = t_l2
        total_time += t_l2
        print(f"    Time: {t_l2:.3f} sec")
        
        # ===== LAYER 3: 64 → 1 =====
        print(f"\n  Layer 3 (64 → 1):")
        t_start = time.time()
        
        ct_logits_list = EncryptedLayerOps.encrypted_linear_layer(
            ct_a2_list[0], weights['fc3_weight'], weights['fc3_bias'], "L3_output"
        )
        
        t_l3 = time.time() - t_start
        layer_timings['layer_3'] = t_l3
        total_time += t_l3
        print(f"    Time: {t_l3:.3f} sec")
        
        print(f"\n  Total inference time: {total_time:.3f} sec")
        
        return ct_logits_list, layer_timings


# ============================================================================
# HOSPITAL INFERENCE EXECUTOR
# ============================================================================

class HospitalInferenceExecutor:
    """Execute complete encrypted inference for one hospital"""
    
    @staticmethod
    def run_hospital_inference(
        hospital_id: str,
        config: HospitalEncryptedInferenceConfig
    ) -> Dict:
        
        print(f"\n" + "=" * 100)
        print(f"HOSPITAL {hospital_id}: END-TO-END ENCRYPTED INFERENCE")
        print(f"=" * 100)
        print(f"Torch device: {config.DEVICE}")
        
        results = {'hospital_id': hospital_id}
        
        # ===== STEP 1: Load context =====
        print(f"\n[Step 1: Load TenSEAL Context]")
        context_path = config.ENCRYPTED_DIR / "context.bin"
        
        try:
            context = ts.context_from(open(str(context_path), 'rb').read())
            print(f"  ✓ Context loaded")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None
        
        # ===== STEP 2: Load test data =====
        print(f"\n[Step 2: Load Hospital {hospital_id}'s Test Data]")
        
        try:
            X_test = np.load(config.DATA_DIR / "X_test.npy")
            y_test = np.load(config.DATA_DIR / "y_test.npy")
            print(f"  ✓ Loaded {len(X_test):,} test samples")
            results['num_test_samples'] = len(X_test)
            
        except Exception as e:
            print(f"  ❌ Error loading data: {e}")
            return None
        
        # ===== STEP 3: Load encrypted model =====
        print(f"\n[Step 3: Load Encrypted Global Model]")
        
        try:
            weights = EncryptedForwardPassHospital.load_encrypted_model(config)
            print(f"  ✓ Model loaded")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None
        
        # ===== STEP 4: Encrypt test data =====
        print(f"\n[Step 4: Encrypt Local Test Data]")
        
        batch_size = config.BATCH_SIZE
        num_batches = (len(X_test) + batch_size - 1) // batch_size
        
        print(f"  Processing {num_batches} batches (size {batch_size})...")
        
        all_logits_encrypted = []
        total_inference_time = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_test))
            
            X_batch = X_test[start_idx:end_idx]
            
            print(f"\n  Batch {batch_idx + 1}/{num_batches}: {len(X_batch)} samples")
            
            # Encrypt batch
            t_encrypt_start = time.time()
            ct_X_batch = [EncryptedLayerOps.encrypt_sample(x, context) for x in X_batch]
            t_encrypt = time.time() - t_encrypt_start
            print(f"    Encryption: {t_encrypt:.3f} sec")
            
            # ===== ENCRYPTED INFERENCE =====
            try:
                t_inference_start = time.time()
                ct_logits, layer_timings = EncryptedForwardPassHospital.encrypted_forward(
                    ct_X_batch, weights, np.arange(start_idx, end_idx), config
                )
                t_inference = time.time() - t_inference_start
                total_inference_time += t_inference
                print(f"    Encrypted forward pass: {t_inference:.3f} sec")
                
            except Exception as e:
                print(f"    ❌ Error in encrypted inference: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # ===== DECRYPT =====
            print(f"    Decrypting predictions...")
            
            t_decrypt_start = time.time()
            logits_encrypted_batch = []
            
            for ct_logit in ct_logits:
                logit_val = float(ct_logit.decrypt()[0])
                logits_encrypted_batch.append(logit_val)
            
            t_decrypt = time.time() - t_decrypt_start
            print(f"    Decryption: {t_decrypt:.3f} sec")
            
            all_logits_encrypted.extend(logits_encrypted_batch)
        
        # ===== CONVERT TO PREDICTIONS =====
        print(f"\n[Step 5: Convert to Predictions]")
        
        logits_encrypted = np.array(all_logits_encrypted)
        y_prob_encrypted = expit(logits_encrypted)
        y_pred_encrypted = (y_prob_encrypted >= config.PREDICTION_THRESHOLD).astype(int)
        
        # ===== EVALUATE =====
        print(f"\n[Step 6: Evaluate Predictions]")
        
        metrics_encrypted = {
            'accuracy': float(accuracy_score(y_test[:len(y_pred_encrypted)], y_pred_encrypted)),
            'auc_roc': float(roc_auc_score(y_test[:len(y_pred_encrypted)], y_prob_encrypted)),
            'f1_score': float(f1_score(y_test[:len(y_pred_encrypted)], y_pred_encrypted, zero_division=0)),
            'precision': float(precision_score(y_test[:len(y_pred_encrypted)], y_pred_encrypted, zero_division=0)),
            'recall': float(recall_score(y_test[:len(y_pred_encrypted)], y_pred_encrypted, zero_division=0)),
        }
        
        print(f"\n  Encrypted Model:")
        print(f"    Accuracy:  {metrics_encrypted['accuracy']:.4f}")
        print(f"    AUC-ROC:   {metrics_encrypted['auc_roc']:.4f}")
        print(f"    F1-Score:  {metrics_encrypted['f1_score']:.4f}")
        print(f"    Precision: {metrics_encrypted['precision']:.4f}")
        print(f"    Recall:    {metrics_encrypted['recall']:.4f}")
        
        results['metrics_encrypted'] = metrics_encrypted
        results['timings'] = {
            'total_inference_sec': float(total_inference_time),
            'num_batches': int(num_batches)
        }
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main Phase 6 execution"""
    
    print("\n" + "=" * 120)
    print("PHASE 6: END-TO-END ENCRYPTED INFERENCE (HOSPITAL-SPECIFIC TEST DATA)")
    print("=" * 120)
    
    config = HospitalEncryptedInferenceConfig()
    config.__post_init__()
    
    all_results = {}
    
    for hospital_id in config.HOSPITALS:
        results = HospitalInferenceExecutor.run_hospital_inference(hospital_id, config)
        if results:
            all_results[hospital_id] = results
    
    print("\n[Saving Results]")
    results_json_path = config.ENCRYPTED_DIR / "phase_6_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Results saved to {results_json_path}")


if __name__ == "__main__":
    main()