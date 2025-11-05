"""
Run this diagnostic script FIRST to identify the exact cause
Save as: diagnostic_check.py
"""

import sys
import torch
import gc

print("=" * 60)
print("DIAGNOSTIC CHECK FOR MEMORY ERROR")
print("=" * 60)

# 1. Check Python version
print(f"\n1. Python Version: {sys.version}")

# 2. Check PyTorch installation
print(f"\n2. PyTorch Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Check GPU memory
    try:
        print(f"\n3. GPU Memory Status:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"   Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    except Exception as e:
        print(f"   Error checking GPU memory: {e}")

# 3. Check transformers library
try:
    import transformers
    print(f"\n4. Transformers Version: {transformers.__version__}")
    
    # Try to import the specific model
    try:
        from transformers import TrajectoryTransformerModel, TrajectoryTransformerConfig
        print("   ✓ TrajectoryTransformerModel available")
    except ImportError:
        print("   ✗ TrajectoryTransformerModel NOT available")
        print("   → Run: pip install transformers>=4.40.0")
except ImportError:
    print("\n4. Transformers NOT installed")
    print("   → Run: pip install transformers")

# 4. Test basic CUDA operations
if torch.cuda.is_available():
    print("\n5. Testing CUDA Operations:")
    try:
        # Test 1: Simple tensor creation
        x = torch.randn(100, 100).cuda()
        print("   ✓ CUDA tensor creation works")
        
        # Test 2: Simple operation
        y = x @ x.T
        print("   ✓ CUDA matrix multiplication works")
        
        # Test 3: Memory cleanup
        del x, y
        torch.cuda.empty_cache()
        gc.collect()
        print("   ✓ CUDA memory cleanup works")
        
    except Exception as e:
        print(f"   ✗ CUDA operation failed: {e}")
        print("   → This is likely causing your memory error!")

# 5. Check for conflicting libraries
print("\n6. Checking for Conflicts:")
try:
    import tensorflow as tf
    print(f"   ⚠ TensorFlow installed: {tf.__version__}")
    print("   → May conflict with PyTorch. Consider using separate environments.")
except ImportError:
    print("   ✓ No TensorFlow conflicts")

# 6. Test transformers model creation
print("\n7. Testing Transformer Model Creation:")
try:
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=64, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config)
    
    if torch.cuda.is_available():
        model = model.cuda()
        test_input = torch.randint(0, 100, (2, 10)).cuda()
        output = model(test_input)
        print("   ✓ Transformer model works on GPU")
    else:
        test_input = torch.randint(0, 100, (2, 10))
        output = model(test_input)
        print("   ✓ Transformer model works on CPU")
    
    del model, test_input, output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
except Exception as e:
    print(f"   ✗ Transformer model failed: {e}")
    print("   → This might be causing your error!")
    import traceback
    traceback.print_exc()

# 7. Check stable-baselines3
print("\n8. Checking stable-baselines3:")
try:
    import stable_baselines3
    print(f"   Version: {stable_baselines3.__version__}")
    
    from stable_baselines3 import TD3
    print("   ✓ TD3 import successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

# Recommendations
print("\nRECOMMENDATIONS:")
print("1. If CUDA operations failed: Reinstall PyTorch")
print("   pip uninstall torch torchvision torchaudio")
print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("\n2. If transformer model failed: Update transformers")
print("   pip install --upgrade transformers")
print("\n3. If GPU memory is low: Reduce batch_size and fine_tune_steps")
print("\n4. If all tests pass: The error is in your training script logic")