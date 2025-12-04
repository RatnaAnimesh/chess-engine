import sys
import os
import time
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.resnet import ChessResNet
from src.model.model_utils import get_device, print_model_summary

def verify_mps():
    print("--- Verifying MPS Support ---")
    
    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available. Using CPU.")
    else:
        print("SUCCESS: MPS is available.")

    device = get_device()
    print(f"Selected Device: {device}")

    # Instantiate Model
    print("\n--- Instantiating Model ---")
    model = ChessResNet(num_blocks=6, num_filters=64) # Tiny config for quick test
    model.to(device)
    print_model_summary(model)

    # Create Dummy Input (Batch Size 8, 119 Channels, 8x8 Board)
    batch_size = 8
    dummy_input = torch.randn(batch_size, 119, 8, 8).to(device)

    # Warmup
    print("\n--- Running Warmup ---")
    _ = model(dummy_input)

    # Benchmark
    print("\n--- Benchmarking Inference ---")
    start_time = time.time()
    iterations = 100
    with torch.no_grad():
        for _ in range(iterations):
            p, v = model(dummy_input)
            # Sync MPS to get accurate timing
            if device.type == 'mps':
                torch.mps.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f"Average Inference Time (Batch {batch_size}): {avg_time*1000:.2f} ms")
    print(f"Throughput: {batch_size / avg_time:.2f} positions/sec")

    print(f"\nOutput Shapes: Policy {p.shape}, Value {v.shape}")
    print("Verification Complete.")

if __name__ == "__main__":
    verify_mps()
