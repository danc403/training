import os
import time
import torch

def print_banner():
    """Print ASCII art banner with dragonfly motif."""
    banner = r"""
               __      __
              (  \0__0/  )
               \__ \/ __/
         ______(____)______
        (______(_  _)______)
                 (  )
                  ||
                  ||
                  ||
                  \/

     NYMPH FAMILY: Hybrid Associative Recall
     Qwen 3 Hyper-Parameters
    """
    print(banner)
