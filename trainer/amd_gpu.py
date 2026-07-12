#!/usr/bin/env python3

import subprocess
import json
import argparse
import sys

class AMDGPU:
    """
    Module for AMD GPU telemetry and MFU baseline lookups.
    """
    
    LOOKUP = {
        "gfx908":  {"name": "MI100",           "fp8": 0,    "fp16": 184.6, "fp32": 46.1},
        "gfx90a":  {"name": "MI250X",          "fp8": 0,    "fp16": 383.0, "fp32": 95.7},
        "gfx942":  {"name": "MI300X",          "fp8": 2614, "fp16": 1307,  "fp32": 163.4},
        "gfx1030": {"name": "RDNA 2 (Navi 21)", "fp8": 0,    "fp16": 48.0,  "fp32": 24.0},
        "gfx1100": {"name": "RDNA 3 (Navi 31)", "fp8": 0,    "fp16": 123.0, "fp32": 61.5},
        "gfx1102": {"name": "RDNA 3 (Navi 33)", "fp8": 0,    "fp16": 45.1,  "fp32": 22.5},
        "gfx1200": {"name": "RDNA 4 (Navi 44)", "fp8": 0,    "fp16": 50.0,  "fp32": 25.0}
    }

    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        # Cache static data once to keep the module lightweight during training loops
        self._static_data = self._fetch_static()

    def _fetch_static(self):
        """Internal call for one-time setup."""
        cmd = ["amd-smi", "static", "-g", str(self.gpu_id), "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)["gpu_data"][0]

    def get_telemetry(self):
        """Call this from your training loop for fresh real-time metrics."""
        cmd = ["amd-smi", "metric", "-m", "-p", "-t", "-f", "-g", str(self.gpu_id), "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)["gpu_metrics"][0]

    @property
    def info(self):
        """Returns the cached static identity of the GPU."""
        asic = self._static_data["asic"]
        bus = self._static_data["bus"]
        arch = asic["target_graphics_version"]
        return {
            "model": asic["market_name"],
            "arch": arch,
            "bdf": bus["bdf"],
            "baselines": self.LOOKUP.get(arch, {"fp8": 0, "fp16": 0, "fp32": 0})
        }

    def print_report(self):
        """CLI tool method to display both static identity and live metrics."""
        i = self.info
        m = self.get_telemetry()
        
        print(f"GPU {self.gpu_id} | {i['model']} ({i['arch']})")
        print(f"VRAM: {m['mem_usage']['used_visible_vram']['value']} MB used")
        print(f"Power: {m['power']['socket_power']['value']} W")
        print(f"Temp: {m['temperature']['edge']['value']} C")

def main():
    parser = argparse.ArgumentParser(description="AMD GPU Telemetry")
    parser.add_argument("-g", "--gpu-id", type=int, default=0)
    args = parser.parse_args()

    # The instance is the link: it holds the gpu_id and provides the methods
    gpu = AMDGPU(gpu_id=args.gpu_id)
    gpu.print_report()

if __name__ == "__main__":
    main()
