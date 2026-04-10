# phase_4_encrypt_weights_hospital.py
"""
Phase 4: Encrypt Model Weights at Each Hospital
================================================

Each hospital:
1. Loads the global model (from Phase 3)
2. Encrypts the weights locally
3. Stores encrypted weights for later use
4. Keeps secret key private

This is preparation for Phase 5 (aggregation) and Phase 6c (inference)
"""

import torch
import tenseal as ts
import numpy as np
import json
from pathlib import Path
from typing import Dict
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Phase4Config:
    """Configuration for Phase 4"""
    
    # Get script directory and build paths from there
    SCRIPT_DIR = Path(__file__).parent
    PARENT_DIR = SCRIPT_DIR.parent
    
    # Directories
    ENCRYPTED_DIR = SCRIPT_DIR / "encrypted"
    MODELS_DIR = PARENT_DIR / "models"  # Point to parent models directory
    REPORTS_DIR = SCRIPT_DIR / "reports"
    
    # Hospital IDs
    HOSPITALS = ['A', 'B', 'C']
    
    # ⭐ REDUCED NETWORK ARCHITECTURE
    INPUT_DIM = 64
    HIDDEN_DIM_1 = 32    # REDUCED
    HIDDEN_DIM_2 = 16    # REDUCED
    OUTPUT_DIM = 1
    
    # TenSEAL Parameters
    SCALE = 2**30
    POLY_MOD_DEGREE = 8192
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Verbose output
    VERBOSE = True
    
    def __post_init__(self):
        """Create directories"""
        self.ENCRYPTED_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# TENSEAL CONTEXT SETUP
# ============================================================================

class TenSEALContextSetup:
    """Create and manage TenSEAL context"""
    
    @staticmethod
    def setup_context(config: Phase4Config) -> ts.Context:
        """
        Create TenSEAL CKKS-RNS context for homomorphic encryption
        
        This context is SHARED - used by all hospitals
        """
        
        print("\n[Setting up TenSEAL Context]")
        
        # Create context
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=config.POLY_MOD_DEGREE,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        
        # Set global scale
        context.global_scale = 2 ** 30
        
        print(f"  ✓ Context created")
        print(f"    Scheme: CKKS-RNS")
        print(f"    Polynomial degree: {config.POLY_MOD_DEGREE}")
        print(f"    Security level: ~128 bits")
        print(f"    Global scale: 2^30")
        
        return context
    
    @staticmethod
    def save_context(context: ts.Context, output_path: Path):
        """Save context to disk (shared across hospitals)"""
        
        with open(output_path, 'wb') as f:
            f.write(context.serialize())
        
        print(f"  ✓ Context saved to {output_path}")
    
    @staticmethod
    def load_context(input_path: Path) -> ts.Context:
        """Load context from disk"""
        
        with open(input_path, 'rb') as f:
            context = ts.context_from(f.read())
        
        return context


# ============================================================================
# WEIGHT ENCRYPTION
# ============================================================================

class WeightEncryptor:
    """Encrypt model weights"""
    
    @staticmethod
    def load_model_weights(model_path: Path, device: str) -> Dict:
        """Load weights from trained model"""
        
        print(f"\n[Loading Model Weights]")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()
        
        weights = {
            'fc1_weight': None, 'fc1_bias': None,
            'fc2_weight': None, 'fc2_bias': None,
            'fc3_weight': None, 'fc3_bias': None
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
        
        print(f"  ✓ Weights loaded")
        print(f"    Layer 1 weight: {weights['fc1_weight'].shape}")
        print(f"    Layer 2 weight: {weights['fc2_weight'].shape}")
        print(f"    Layer 3 weight: {weights['fc3_weight'].shape}")
        
        return weights
    
    @staticmethod
    def encrypt_weight_vector(weight_vector: np.ndarray, context: ts.Context) -> ts.CKKSVector:
        """
        Encrypt a single weight vector
        
        Args:
            weight_vector: 1D numpy array of weights
            context: TenSEAL context
            
        Returns:
            Encrypted weight vector
        """
        
        # Convert to list and encrypt
        ct_vector = ts.ckks_vector(context, weight_vector.tolist())
        
        return ct_vector
    
    @staticmethod
    def encrypt_weight_matrix(weight_matrix: np.ndarray, context: ts.Context) -> list:
        """
        Encrypt a weight matrix (row by row)
        
        Args:
            weight_matrix: 2D numpy array (output_dim, input_dim)
            context: TenSEAL context
            
        Returns:
            List of encrypted row vectors
        """
        
        encrypted_rows = []
        
        for i, row in enumerate(weight_matrix):
            ct_row = WeightEncryptor.encrypt_weight_vector(row, context)
            encrypted_rows.append(ct_row)
        
        return encrypted_rows


# ============================================================================
# HOSPITAL WEIGHT ENCRYPTION PROCESS
# ============================================================================

class HospitalPhase4Executor:
    """Execute Phase 4 for each hospital"""
    
    @staticmethod
    def encrypt_weights_at_hospital(hospital_id: str, config: Phase4Config, context: ts.Context):
        """
        Encrypt weights at hospital
        
        Each hospital:
        1. Loads the HOSPITAL-SPECIFIC reduced model
        2. Encrypts weights using SHARED context
        3. Stores encrypted weights
        4. Keeps context for Phase 6c
        """
        
        print(f"\n" + "=" * 100)
        print(f"HOSPITAL {hospital_id}: PHASE 4 - ENCRYPT WEIGHTS")
        print(f"=" * 100)
        
        results = {'hospital_id': hospital_id}
        
        # ===== STEP 1: Load hospital-specific reduced model =====
        print(f"\n[Step 1: Load Hospital {hospital_id} Reduced Model]")
        
        # COMMENTED OUT: Global model (old approach)
        # model_path = config.MODELS_DIR / "mlp_best_model_REDUCED.pt"
        
        # Load HOSPITAL-SPECIFIC reduced model
        model_path = config.MODELS_DIR / f"mlp_best_model_{hospital_id}_REDUCED.pt"
        
        try:
            weights = WeightEncryptor.load_model_weights(model_path, config.DEVICE)
            print(f"  ✓ Hospital {hospital_id} Model loaded")
            print(f"    Path: {model_path}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None
        
        # ===== STEP 2: Encrypt weights =====
        print(f"\n[Step 2: Encrypt Weights]")
        
        t_start = time.time()
        
        try:
            # Encrypt Layer 1 weights (32 × 64)
            print(f"  Encrypting Layer 1 weights (32 × 64)...")
            t_l1 = time.time()
            ct_fc1_weight = WeightEncryptor.encrypt_weight_matrix(weights['fc1_weight'], context)
            ct_fc1_bias = WeightEncryptor.encrypt_weight_vector(weights['fc1_bias'], context)
            t_l1_time = time.time() - t_l1
            print(f"    ✓ Encrypted {len(ct_fc1_weight)} weight vectors in {t_l1_time:.2f}s")
            
            # Encrypt Layer 2 weights (16 × 32)
            print(f"  Encrypting Layer 2 weights (16 × 32)...")
            t_l2 = time.time()
            ct_fc2_weight = WeightEncryptor.encrypt_weight_matrix(weights['fc2_weight'], context)
            ct_fc2_bias = WeightEncryptor.encrypt_weight_vector(weights['fc2_bias'], context)
            t_l2_time = time.time() - t_l2
            print(f"    ✓ Encrypted {len(ct_fc2_weight)} weight vectors in {t_l2_time:.2f}s")
            
            # Encrypt Layer 3 weights (1 × 16)
            print(f"  Encrypting Layer 3 weights (1 × 16)...")
            t_l3 = time.time()
            ct_fc3_weight = WeightEncryptor.encrypt_weight_matrix(weights['fc3_weight'], context)
            ct_fc3_bias = WeightEncryptor.encrypt_weight_vector(weights['fc3_bias'], context)
            t_l3_time = time.time() - t_l3
            print(f"    ✓ Encrypted {len(ct_fc3_weight)} weight vectors in {t_l3_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error encrypting weights: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        total_time = time.time() - t_start
        print(f"\n  ✓ All weights encrypted")
        print(f"    Total encryption time: {total_time:.2f}s")
        
        results['encryption_time_seconds'] = total_time
        results['weights_encrypted'] = {
            'fc1_weight': len(ct_fc1_weight),
            'fc2_weight': len(ct_fc2_weight),
            'fc3_weight': len(ct_fc3_weight),
        }
        
        # ===== STEP 3: Store encrypted weights =====
        print(f"\n[Step 3: Store Encrypted Weights]")
        
        encrypted_weights_dir = config.ENCRYPTED_DIR / f"hospital_{hospital_id}"
        encrypted_weights_dir.mkdir(exist_ok=True, parents=True)
        
        # Save encrypted weights as serialized binary files
        try:
            # Save Layer 1 weights
            ct_fc1_weight_path = encrypted_weights_dir / f"ct_fc1_weight_{hospital_id}.bin"
            with open(ct_fc1_weight_path, 'wb') as f:
                # Save each encrypted row
                for i, ct_row in enumerate(ct_fc1_weight):
                    f.write(ct_row.serialize())
            print(f"  ✓ Saved {ct_fc1_weight_path}")
            
            # Save Layer 1 bias
            ct_fc1_bias_path = encrypted_weights_dir / f"ct_fc1_bias_{hospital_id}.bin"
            with open(ct_fc1_bias_path, 'wb') as f:
                f.write(ct_fc1_bias.serialize())
            print(f"  ✓ Saved {ct_fc1_bias_path}")
            
            # Save Layer 2 weights
            ct_fc2_weight_path = encrypted_weights_dir / f"ct_fc2_weight_{hospital_id}.bin"
            with open(ct_fc2_weight_path, 'wb') as f:
                for i, ct_row in enumerate(ct_fc2_weight):
                    f.write(ct_row.serialize())
            print(f"  ✓ Saved {ct_fc2_weight_path}")
            
            # Save Layer 2 bias
            ct_fc2_bias_path = encrypted_weights_dir / f"ct_fc2_bias_{hospital_id}.bin"
            with open(ct_fc2_bias_path, 'wb') as f:
                f.write(ct_fc2_bias.serialize())
            print(f"  ✓ Saved {ct_fc2_bias_path}")
            
            # Save Layer 3 weights
            ct_fc3_weight_path = encrypted_weights_dir / f"ct_fc3_weight_{hospital_id}.bin"
            with open(ct_fc3_weight_path, 'wb') as f:
                for i, ct_row in enumerate(ct_fc3_weight):
                    f.write(ct_row.serialize())
            print(f"  ✓ Saved {ct_fc3_weight_path}")
            
            # Save Layer 3 bias
            ct_fc3_bias_path = encrypted_weights_dir / f"ct_fc3_bias_{hospital_id}.bin"
            with open(ct_fc3_bias_path, 'wb') as f:
                f.write(ct_fc3_bias.serialize())
            print(f"  ✓ Saved {ct_fc3_bias_path}")
            
            print(f"\n  ✓ All encrypted weights saved to {encrypted_weights_dir}")
            
        except Exception as e:
            print(f"  ❌ Error saving encrypted weights: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # ===== STEP 4: Summary =====
        print(f"\n[Step 4: Summary]")
        print(f"  ✓ Hospital {hospital_id} encryption complete")
        print(f"    Model: Hospital {hospital_id} Reduced (64 → 32 → 16 → 1)")
        print(f"    Total weights encrypted: {len(ct_fc1_weight) + len(ct_fc2_weight) + len(ct_fc3_weight)}")
        print(f"    Time: {total_time:.2f}s")
        print(f"    Status: Ready for Phase 5 (Aggregation)")
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main Phase 4 execution"""
    
    print("\n" + "=" * 100)
    print("PHASE 4: ENCRYPT HOSPITAL-SPECIFIC REDUCED MODELS AT EACH HOSPITAL")
    print("=" * 100)
    
    config = Phase4Config()
    config.__post_init__()
    
    print(f"\n[Configuration]")
    print(f"  Architecture: 64 → {config.HIDDEN_DIM_1} → {config.HIDDEN_DIM_2} → 1 (REDUCED per hospital)")
    print(f"  Hospitals: {config.HOSPITALS}")
    print(f"  Device: {config.DEVICE}")
    
    # ===== Setup shared context =====
    print(f"\n[Creating Shared TenSEAL Context]")
    context = TenSEALContextSetup.setup_context(config)
    
    # Save context for later use (Phase 6c)
    context_path = config.ENCRYPTED_DIR / "context.bin"
    TenSEALContextSetup.save_context(context, context_path)
    
    # ===== Encrypt weights at each hospital =====
    all_results = {}
    
    for hospital_id in config.HOSPITALS:
        try:
            print(f"\n[Processing Hospital {hospital_id}]")
            results = HospitalPhase4Executor.encrypt_weights_at_hospital(
                hospital_id, config, context
            )
            if results:
                all_results[hospital_id] = results
                print(f"✅ Hospital {hospital_id}: SUCCESS!")
        except Exception as e:
            print(f"❌ Hospital {hospital_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== Final summary =====
    print(f"\n" + "=" * 100)
    print("PHASE 4 COMPLETE")
    print("=" * 100)
    
    print(f"\n✅ Phase 4 Complete!")
    print(f"\nHospitals processed: {list(all_results.keys())}")
    print(f"Artifacts created:")
    print(f"  • Context: {context_path}")
    print(f"  • Encrypted weights (Hospital A): {config.ENCRYPTED_DIR}/hospital_A/")
    print(f"  • Encrypted weights (Hospital B): {config.ENCRYPTED_DIR}/hospital_B/")
    print(f"  • Encrypted weights (Hospital C): {config.ENCRYPTED_DIR}/hospital_C/")
    print(f"\nEncrypted models:")
    print(f"  • Hospital A: mlp_best_model_A_REDUCED.pt (encrypted)")
    print(f"  • Hospital B: mlp_best_model_B_REDUCED.pt (encrypted)")
    print(f"  • Hospital C: mlp_best_model_C_REDUCED.pt (encrypted)")
    
    print(f"\n📋 Next: Phase 5 - Blind Server Aggregation")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()