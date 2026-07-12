import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MaskedBinaryShardDataset(Dataset):
    def __init__(self, data_dir, context_length, eos_id=0, device="cuda"):
        super().__init__()
        self.data_dir = data_dir
        self.context_length = context_length
        self.read_length = context_length + 1
        self.eos_id = eos_id
        self.device = device
        
        self.extra_mask_chance = 0.0
        
        data_paths = sorted(glob.glob(os.path.join(data_dir, "*_data.bin")))
        mask_paths = sorted(glob.glob(os.path.join(data_dir, "*_mask.bin")))
        
        if not data_paths or len(data_paths) != len(mask_paths):
            raise FileNotFoundError(f"Mismatched or missing .bin shards in {data_dir}")
            
        # Pre-load all data into VRAM tensors
        self.all_data = []
        self.all_masks = []
        self.shard_block_counts = []
        self.shard_offsets = []
        total_blocks = 0
        
        print(f"--- Pre-loading {len(data_paths)} shards into VRAM ---")
        for dp, mp in zip(data_paths, mask_paths):
            # Load from disk to CPU RAM
            data_np = np.fromfile(dp, dtype=np.uint16)
            mask_np = np.fromfile(mp, dtype=np.uint8)
            
            # Transfer to VRAM
            self.all_data.append(torch.from_numpy(data_np).to(self.device))
            self.all_masks.append(torch.from_numpy(mask_np).to(self.device).float())
            
            num_tokens = data_np.size
            blocks_in_shard = num_tokens // 2048
            
            self.shard_offsets.append(total_blocks)
            self.shard_block_counts.append(blocks_in_shard)
            total_blocks += blocks_in_shard
            
        self.num_samples = total_blocks
        print(f"--- All data loaded into VRAM. Total blocks: {self.num_samples} ---")

    def __len__(self):
        return self.num_samples

    def set_mask_chance(self, chance):
        self.extra_mask_chance = float(chance)

    def __getitem__(self, idx):
        # Locate shard
        shard_idx = -1
        for i, offset in enumerate(self.shard_offsets):
            if idx >= offset and idx < offset + self.shard_block_counts[i]:
                shard_idx = i
                break
        
        local_block_idx = idx - self.shard_offsets[shard_idx]
        start_idx = local_block_idx * 2048
        end_idx = start_idx + 2048
        
        # Slicing directly from VRAM tensors
        raw_tokens = self.all_data[shard_idx][start_idx:end_idx]
        raw_mask = self.all_masks[shard_idx][start_idx:end_idx]
        
        if self.extra_mask_chance > 0:
            noise = torch.rand(2048, device=self.device)
            kill_mask = torch.where(
                (noise < self.extra_mask_chance) & 
                (raw_mask == 1.0) & 
                (raw_tokens != self.eos_id), 
                0.0, 1.0
            )
            raw_mask = raw_mask * kill_mask
        
        # Shift targets
        x = raw_tokens[:-1].long()
        y = raw_tokens[1:].long()
        mask = raw_mask[1:]
        
        return x, y, mask

def get_dataloader(data_dir, context_length, batch_size, eos_id=0, num_workers=0, device="cuda"):
    """
    Returns a DataLoader. num_workers must be 0 because data is already in VRAM.
    """
    dataset = MaskedBinaryShardDataset(data_dir, context_length, eos_id=eos_id, device=device)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, # Multi-processing is redundant and slows down VRAM access
        pin_memory=False, # Data is already on the GPU
        drop_last=True
    )
