import torch
import omegaconf
from models.dit import DIT

# 1. Mock the Configuration
# We match the structure expected by DIT.__init__
conf_dict = {
    "algo": {
        "causal_attention": True,  # Test the Causal Block (most complex part)
    },
    "model": {
        "hidden_size": 768,       # Standard size
        "cond_dim": 256,          # For timestep embedding
        "n_heads": 12,            # 768 / 12 = 64 head_dim (optimal for V100)
        "n_blocks": 2,            # Keep it small for testing
        "dropout": 0.1,
        "scale_by_sigma": True
    }
}
config = omegaconf.OmegaConf.create(conf_dict)

def test_model():
    print("--- Starting DIT Test on V100 ---")
    
    # Check for GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. Running on CPU (xFormers might not trigger).")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Running on: {torch.cuda.get_device_name(0)}")

    # 2. Initialize Model
    vocab_size = 1000
    try:
        model = DIT(config, vocab_size=vocab_size).to(device)
        print("\n[Success] Model initialized.")
    except Exception as e:
        print(f"\n[Fail] Model initialization failed: {e}")
        return

    # 3. Create Dummy Inputs
    batch_size = 2
    seq_len = 32
    
    # Input tokens (indices)
    torch.manual_seed(0)
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Input sigma/timesteps (random floats)
    sigma = torch.randn(batch_size).to(device)

    # 4. Run Forward Pass
    print(f"Input shape: {x.shape}")
    try:
        # Enable autocast to match training conditions
        with torch.amp.autocast('cuda'):
            output = model(x, sigma)
            
        print(f"Output shape: {output.shape}")
        
        # Verify Output Shape: (Batch, SeqLen, VocabSize)
        expected_shape = (batch_size, seq_len, vocab_size)
        assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
        print('\n', output.argmax(-1))
        print("\n[PASSED] Forward pass completed successfully.")
        print("If you saw 'Flash Attention 2 not found' above, the fallback is working!")
        
    except ImportError as e:
        print(f"\n[Fail] Dependency Error: {e}")
        print("Did you install xformers?")
    except RuntimeError as e:
        print(f"\n[Fail] Runtime Error: {e}")
        print("Check if your head_dim is compatible (e.g. 64) or if JIT flags are removed.")

if __name__ == "__main__":
    test_model()