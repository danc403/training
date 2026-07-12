#!/usr/bin/env python3

import subprocess
import json
import argparse
import sys

class AMDGPU:
    """
    Module for AMD GPU telemetry and MFU baseline lookups.
    Importable into larger packages or executable as a standalone CLI tool.
    """
    
    # Source of Truth for MFU baselines (TFLOPS)
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
        self.data = self._refresh()

    def _refresh(self):
        """Fetches static hardware data from amd-smi."""
        try:
            cmd = ["amd-smi", "static", "-g", str(self.gpu_id), "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)["gpu_data"][0]
        except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to query amd-smi static for GPU {self.gpu_id}: {e}")

    def get_telemetry(self):
        """Fetches dynamic real-time metrics for VRAM, Power, Temp, and Fan."""
        try:
            # Aggregate command to get all metrics in one call
            cmd = ["amd-smi", "metric", "-m", "-p", "-t", "-f", "-g", str(self.gpu_id), "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metrics = json.loads(result.stdout)["gpu_metrics"][0]
            
            return {
                "vram": {
                    "used": metrics["mem_usage"]["used_visible_vram"],
                    "total": metrics["mem_usage"]["total_visible_vram"],
                    "free": metrics["mem_usage"]["free_visible_vram"]
                },
                "power": metrics["power"]["socket_power"],
                "temp": metrics["temperature"]["edge"],
                "fan": metrics["fan"]["usage"]
            }
        except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to query amd-smi metrics for GPU {self.gpu_id}: {e}")

    @property
    def info(self):
        """Returns the full dictionary of hardware state and baselines."""
        asic = self.data["asic"]
        bus = self.data["bus"]
        arch = asic["target_graphics_version"]
        
        return {
            "gpu_id": self.gpu_id,
            "model": asic["market_name"],
            "arch": arch,
            "bdf": bus["bdf"],
            "vram_gb": round(self.data["vram"]["size"]["value"] / 1024, 2),
            "pcie_link": f"{bus['pcie_interface_version']} @ {bus['max_pcie_speed']['value']} GT/s",
            "baselines": self.LOOKUP.get(arch, {"fp8": 0, "fp16": 0, "fp32": 0})
        }

    @property
    def peak_tflops(self):
        return self.info["baselines"].get("fp16", 0.0)

    def print_report(self):
        """Prints a human-readable summary including static data and real-time telemetry."""
        i = self.info
        b = i["baselines"]
        m = self.get_telemetry()
        
        print("==========================================")
        print(f"          AMD GPU STATUS REPORT           ")
        print("==========================================")
        print(f"GPU ID:       {i['gpu_id']}")
        print(f"Model:        {i['model']}")
        print(f"Architecture: {i['arch']}")
        print(f"Bus (BDF):    {i['bdf']}")
        print(f"VRAM:         {m['vram']['used']['value']} / {m['vram']['total']['value']} {m['vram']['used']['unit']} (Free: {m['vram']['free']['value']})")
        print(f"Power:        {m['power']['value']} {m['power']['unit']}")
        print(f"Temp (Edge):  {m['temp']['value']} {m['temp']['unit']}")
        print(f"PCIe Link:    {i['pcie_link']}")
        print("------------------------------------------")
        print(f"MFU Baselines (TFLOPS):")
        print(f"  FP8:        {b['fp8']}")
        print(f"  FP16:       {b['fp16']}")
        print(f"  FP32:       {b['fp32']}")
        print("==========================================")

def main():
    parser = argparse.ArgumentParser(description="AMD GPU Telemetry and MFU Tool")
    parser.add_argument("-g", "--gpu-id", type=int, default=0, help="GPU index to query (default: 0)")
    args = parser.parse_args()

    try:
        gpu = AMDGPU(gpu_id=args.gpu_id)
        gpu.print_report()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
