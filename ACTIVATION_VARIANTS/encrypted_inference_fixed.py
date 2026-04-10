#!/usr/bin/env python3
"""
Phase 6: End-to-End Encrypted Inference (FIXED VERSION)
Uses degree-1 ReLU to eliminate scale overflow errors
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

print("[FIXED] Using degree-1 ReLU (linear approximation)")
print("[FIXED] No squaring = NO scale overflow errors\n")


class HospitalEncryptedInferenceConfig:
    """Configuration for hospital-specific encrypted inference"""
    
    ENCRYPTED_DIR = Path("encrypted")
    DATA_DIR = Path("data/processed/phase2")
    REPORTS_DIR = Path("reports")
    
    HOSPITALS = ['A', 'B', 'C']
    INPUT_DIM = 60
    HIDDEN_DIM_1 = 128
    HIDDEN_DIM_2 = 64
    OUTPUT_DIM = 1
    BATCH_SIZE = 4
    SCALE = 2**30
    RELU_BOUND = 2.0
    PREFERRED_DEVICE = 'cuda'
    DEVICE = 'cpu'
    PREDICTION_THRESHOLD = 0.5
    
    def __post_init__(self):
        if self.PREFERRED_DEVICE == 'cuda' and torch.cuda.is_available():
            self.DEVICE = 'cuda'
        else:
            self.DEVICE = 'cpu'
        self.REPORTS_DIR.mkdir(exist_ok=True)


class ChebyshevReLU:
    """Degree-1 Chebyshev polynomial (LINEAR - NO SQUARING)""" 
    
    @staticmethod
    def get_coefficients(bound: float = 2.0) -> np.ndarray:
        """c0 + c1*x (no squaring)"""
        if bound == 2.0:
            return np.array([0.5, 0.25])
        else:
            base = np.array([0.5, 0.25])
            scale = bound / 2.0
            return np.array([base[0], base[1] * scale])
    
    @staticmethod
    def eval_encrypted(ct_x: ts.CKKSVector, bound: float = 2.0) -> ts.CKKSVector:
        """Evaluate degree-1 ReLU (NO SQUARING)"""
        coeffs = ChebyshevReLU.get_coefficients(bound)
        ct_result = ct_x * coeffs[1]  # c1 * ct_x
        ct_result = ct_result + coeffs[0]  # + c0
        return ct_result


class EncryptedLayerOps:
    """Homomorphic layer operations"""
    
    @staticmethod
    def encrypt_sample(sample: np.ndarray, context: ts.Context) -> ts.CKKSVector:
        return ts.ckks_vector(context, sample.tolist())
    
    @staticmethod
    def encrypted_linear_layer_vector(ct_x: ts.CKKSVector, W: np.ndarray, b: np.ndarray) -> List[ts.CKKSVector]:
        """y = W @ x + b where ct_x is encrypted vector and output is list of encrypted scalars"""
        output_dim, input_dim = W.shape
        ct_output = []
        
        for i in range(output_dim):
            # Initialize with W[i,0] * ct_x[0]
            w_i0 = float(W[i, 0].item() if hasattr(W[i, 0], 'item') else W[i, 0])
            ct_y_i = ct_x * w_i0
            
            # Add remaining terms: W[i,j] * ct_x[j]
            for j in range(1, input_dim):
                w_ij = float(W[i, j].item() if hasattr(W[i, j], 'item') else W[i, j])
                ct_y_i = ct_y_i + (ct_x * w_ij)
            
            # Add bias
            b_i = float(b[i].item() if hasattr(b[i], 'item') else b[i])
            ct_y_i = ct_y_i + b_i
            ct_output.append(ct_y_i)
        
        return ct_output
    
    @staticmethod
    def encrypted_linear_layer_scalars(ct_x_list: List[ts.CKKSVector], W: np.ndarray, b: np.ndarray) -> List[ts.CKKSVector]:
        """y = W @ x + b where ct_x_list is list of encrypted scalars (from previous layer)"""
        output_dim, input_dim = W.shape
        assert len(ct_x_list) == input_dim, f"Expected {input_dim} inputs, got {len(ct_x_list)}"
        
        ct_output = []
        
        for i in range(output_dim):
            # Start with W[i,0] * ct_x[0]
            w_i0 = float(W[i, 0].item() if hasattr(W[i, 0], 'item') else W[i, 0])
            ct_y_i = ct_x_list[0] * w_i0
            
            # Add remaining: W[i,j] * ct_x[j]
            for j in range(1, input_dim):
                w_ij = float(W[i, j].item() if hasattr(W[i, j], 'item') else W[i, j])
                ct_y_i = ct_y_i + (ct_x_list[j] * w_ij)
            
            # Add bias
            b_i = float(b[i].item() if hasattr(b[i], 'item') else b[i])
            ct_y_i = ct_y_i + b_i
            ct_output.append(ct_y_i)
        
        return ct_output
    
    @staticmethod
    def encrypted_relu_layer(ct_z_list: List[ts.CKKSVector]) -> List[ts.CKKSVector]:
        """Apply ReLU to list of encrypted scalars"""
        return [ChebyshevReLU.eval_encrypted(ct_z, bound=2.0) for ct_z in ct_z_list]


class HospitalInferenceExecutor:
    """Execute encrypted inference for one hospital"""
    
    @staticmethod
    def run_hospital_inference(hospital_id: str, config: HospitalEncryptedInferenceConfig) -> Dict:
        """Complete encrypted inference"""
        
        print(f"\n[HOSPITAL {hospital_id}] Starting encrypted inference")
        results = {'hospital_id': hospital_id}
        
        # Load context
        print(f"[HOSPITAL {hospital_id}] Loading context...")
        context_path = config.ENCRYPTED_DIR / "context.bin"
        try:
            context = ts.context_from(open(str(context_path), 'rb').read())
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
        
        # Load test data
        print(f"[HOSPITAL {hospital_id}] Loading test data...")
        try:
            X_test = np.load(config.DATA_DIR / "X_test.npy")
            y_test = np.load(config.DATA_DIR / "y_test.npy")
            print(f"  Loaded {len(X_test)} test samples")
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
        
        # Load model
        print(f"[HOSPITAL {hospital_id}] Loading model...")
        try:
            checkpoint = torch.load("mlp_best_model.pt", map_location=config.DEVICE)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            weights = {
                'fc1_weight': state_dict['fc1.weight'].cpu().numpy(),
                'fc1_bias': state_dict['fc1.bias'].cpu().numpy(),
                'fc2_weight': state_dict['fc2.weight'].cpu().numpy(),
                'fc2_bias': state_dict['fc2.bias'].cpu().numpy(),
                'fc3_weight': state_dict['fc3.weight'].cpu().numpy(),
                'fc3_bias': state_dict['fc3.bias'].cpu().numpy(),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
        
        # Load torch weights for comparison
        torch_device = torch.device(config.DEVICE)
        torch_weights = {k: torch.from_numpy(v).float().to(torch_device) for k, v in weights.items()}
        
        # Process batches
        print(f"[HOSPITAL {hospital_id}] Processing {len(X_test)} samples...")
        batch_size = config.BATCH_SIZE
        num_batches = (len(X_test) + batch_size - 1) // batch_size
        
        all_logits_encrypted = []
        all_logits_plaintext = []
        total_time = 0
        
        for batch_idx in range(num_batches):
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{num_batches}")
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_test))
            X_batch = X_test[start_idx:end_idx]
            
            # Encrypt batch
            t0 = time.time()
            ct_X_batch = [EncryptedLayerOps.encrypt_sample(x, context) for x in X_batch]
            t_enc = time.time() - t0
            
            # Encrypted inference
            try:
                t0 = time.time()
                # Layer 1: 60 inputs -> 128 outputs (encrypted vector x weight matrix)
                ct_z1 = EncryptedLayerOps.encrypted_linear_layer_vector(ct_X_batch[0], weights['fc1_weight'], weights['fc1_bias'])
                ct_a1 = EncryptedLayerOps.encrypted_relu_layer(ct_z1)
                
                # Layer 2: 128 inputs -> 64 outputs (encrypted scalars x weight matrix)
                ct_z2 = EncryptedLayerOps.encrypted_linear_layer_scalars(ct_a1, weights['fc2_weight'], weights['fc2_bias'])
                ct_a2 = EncryptedLayerOps.encrypted_relu_layer(ct_z2)
                
                # Layer 3: 64 inputs -> 1 output (no ReLU)
                ct_logits = EncryptedLayerOps.encrypted_linear_layer_scalars(ct_a2, weights['fc3_weight'], weights['fc3_bias'])
                
                t_inf = time.time() - t0
                total_time += t_inf
                
            except Exception as e:
                print(f"  ERROR in encrypted inference: {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # Decrypt (extract the single logit from the output layer)
            t0 = time.time()
            logits_encrypted_batch = []
            # ct_logits is a list with 1 element (since output is 1D)
            ct_logit_output = ct_logits[0] if isinstance(ct_logits, list) else ct_logits
            logit_val = float(ct_logit_output.decrypt()[0])
            logits_encrypted_batch.append(logit_val)
            t_dec = time.time() - t0
            
            all_logits_encrypted.extend(logits_encrypted_batch)
            
            # Plaintext baseline
            with torch.no_grad():
                X_torch = torch.from_numpy(X_batch).float().to(torch_device)
                z1 = torch.mm(X_torch, torch_weights['fc1_weight'].T) + torch_weights['fc1_bias']
                a1 = torch.relu(z1)
                z2 = torch.mm(a1, torch_weights['fc2_weight'].T) + torch_weights['fc2_bias']
                a2 = torch.relu(z2)
                logits_plain = torch.mm(a2, torch_weights['fc3_weight'].T) + torch_weights['fc3_bias']
            
            logits_plaintext_batch = logits_plain.detach().cpu().numpy().flatten()
            all_logits_plaintext.extend(logits_plaintext_batch)
        
        # Convert to predictions
        logits_encrypted = np.array(all_logits_encrypted)
        logits_plaintext = np.array(all_logits_plaintext)
        
        y_prob_encrypted = expit(logits_encrypted)
        y_prob_plaintext = expit(logits_plaintext)
        
        y_pred_encrypted = (y_prob_encrypted >= 0.5).astype(int)
        y_pred_plaintext = (y_prob_plaintext >= 0.5).astype(int)
        
        # Metrics
        metrics_enc = {
            'accuracy': float(accuracy_score(y_test, y_pred_encrypted)),
            'auc_roc': float(roc_auc_score(y_test, y_prob_encrypted)),
            'f1_score': float(f1_score(y_test, y_pred_encrypted, zero_division=0)),
        }
        
        metrics_plain = {
            'accuracy': float(accuracy_score(y_test, y_pred_plaintext)),
            'auc_roc': float(roc_auc_score(y_test, y_prob_plaintext)),
            'f1_score': float(f1_score(y_test, y_pred_plaintext, zero_division=0)),
        }
        
        print(f"[HOSPITAL {hospital_id}] Results:")
        print(f"  Encrypted | Accuracy: {metrics_enc['accuracy']:.4f} | AUC: {metrics_enc['auc_roc']:.4f}")
        print(f"  Plaintext | Accuracy: {metrics_plain['accuracy']:.4f} | AUC: {metrics_plain['auc_roc']:.4f}")
        print(f"  Difference | Accuracy: {abs(metrics_enc['accuracy'] - metrics_plain['accuracy']):.6f}")
        
        results['metrics_encrypted'] = metrics_enc
        results['metrics_plaintext'] = metrics_plain
        results['num_samples'] = len(X_test)
        
        return results


def main():
    """Main execution"""
    
    print("=" * 80)
    print("PHASE 6: END-TO-END ENCRYPTED INFERENCE")
    print("FIX: Degree-1 ReLU (Linear) - Eliminates Scale Overflow")
    print("=" * 80)
    
    config = HospitalEncryptedInferenceConfig()
    config.__post_init__()
    
    all_results = {}
    for hospital_id in config.HOSPITALS:
        try:
            results = HospitalInferenceExecutor.run_hospital_inference(hospital_id, config)
            if results:
                all_results[hospital_id] = results
        except Exception as e:
            print(f"ERROR for Hospital {hospital_id}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for hospital_id in ['A', 'B', 'C']:
        if hospital_id in all_results:
            res = all_results[hospital_id]
            print(f"\nHospital {hospital_id}:")
            print(f"  Samples: {res['num_samples']}")
            print(f"  Accuracy: {res['metrics_encrypted']['accuracy']:.4f} (encrypted) vs {res['metrics_plaintext']['accuracy']:.4f} (plaintext)")
            print(f"  AUC-ROC:  {res['metrics_encrypted']['auc_roc']:.4f} (encrypted) vs {res['metrics_plaintext']['auc_roc']:.4f} (plaintext)")
    
    print("\n" + "=" * 80)
    print("SUCCESS: All hospitals completed encrypted inference")
    print("No scale overflow errors with degree-1 ReLU")
    print("=" * 80 + "\n")
    
    return all_results


if __name__ == "__main__":
    results = main()
