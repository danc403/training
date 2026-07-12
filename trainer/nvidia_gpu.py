#!/usr/bin/env python3

import torch
import argparse
import sys

class NVIDIAGPU:
    """
    Module for NVIDIA GPU telemetry and MFU baseline lookups.
    Encapsulates the peak TFLOPS logic previously hardcoded in train.py.
    """

    # Source of Truth for MFU baselines (Peak FP16/BF16 TFLOPS)
    LOOKUP = {
        "3060": 51.2, "3070": 81.3, "3080": 119.0, "3090": 142.0,
        "4060": 15.1, "4070": 29.0, "4080": 97.5, "4090": 165.2,
        "5070": 125.0, "5080": 180.0, "5090": 209.5
    }

    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system.")
            
        self.name = torch.cuda.get_device_name(gpu_id).upper()

    @property
    def peak_tflops(self):
        """Returns the peak TFLOPS for the detected card as a float."""
        for key, val in self.LOOKUP.items():
            if key in self.name:
                return val # Return as TFLOPS float
        return 142.0 # Fallback (3090 baseline)

def get_memory(self):
        """Returns memory usage in a structure compatible with the AMDGPU logic."""
        # torch.cuda returns bytes; we represent them as a dictionary to match AMD's API
        allocated = torch.cuda.memory_allocated(self.gpu_id)
        reserved = torch.cuda.memory_reserved(self.gpu_id)
        total = torch.cuda.get_device_properties(self.gpu_id).total_memory
        
        return {
            "used_visible_vram": {"value": allocated / (1024**2)}, # Converting to MB
            "total_visible_vram": {"value": total / (1024**2)}    # Converting to MB
        }

    def print_info(self):
        print(f"Device: {self.name}")
        print(f"Peak Baseline: {self.peak_tflops:.1f} TFLOPS")

def main():
    parser = argparse.ArgumentParser(description="NVIDIA GPU Telemetry Tool")
    parser.add_argument("-g", "--gpu-id", type=int, default=0, help="GPU index (default: 0)")
    args = parser.parse_args()

    try:
        gpu = NVIDIAGPU(gpu_id=args.gpu_id)
        gpu.print_info()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
