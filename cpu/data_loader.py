import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MaskedBinaryShardDataset(Dataset):
    def __init__(self, data_dir, context_length):
        super().__init__()
        self.data_dir = data_dir
        # We need context_length + 1 to create the shifted Y targets
        self.context_length = context_length
        self.read_length = context_length + 1
        
        self.data_paths = sorted(glob.glob(os.path.join(data_dir, "*_data.bin")))
        self.mask_paths = sorted(glob.glob(os.path.join(data_dir, "*_mask.bin")))
        
        if not self.data_paths or len(self.data_paths) != len(self.mask_paths):
            raise FileNotFoundError(f"Mismatched or missing .bin shards in {data_dir}")
            
        self.shard_offsets = []
        self.shard_block_counts = []
        total_blocks = 0
        
        for dp in self.data_paths:
            file_size = os.path.getsize(dp)
            num_tokens = file_size // 2 # uint16
            # We calculate blocks based on the original packing CONTEXT_SIZE
            blocks_in_shard = num_tokens // 2048 
            
            self.shard_offsets.append(total_blocks)
            self.shard_block_counts.append(blocks_in_shard)
            total_blocks += blocks_in_shard
                
        self.num_samples = total_blocks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        shard_idx = -1
        # Quick binary search or simple loop to find the shard
        for i, offset in enumerate(self.shard_offsets):
            if idx >= offset and idx < offset + self.shard_block_counts[i]:
                shard_idx = i
                break
        
        if shard_idx == -1:
            raise IndexError(f"Block index {idx} out of range.")
                
        data_path = self.data_paths[shard_idx]
        mask_path = self.mask_paths[shard_idx]
        local_block_idx = idx - self.shard_offsets[shard_idx]
        
        # Read tokens (uint16 = 2 bytes)
        # We use 2048 as the stride because that's how they were packed
        raw_tokens = np.fromfile(
            data_path, 
            dtype=np.uint16, 
            count=2048, 
            offset=local_block_idx * 2048 * 2
        )
        
        # Read mask (uint8 = 1 byte)
        raw_mask = np.fromfile(
            mask_path, 
            dtype=np.uint8, 
            count=2048, 
            offset=local_block_idx * 2048 * 1
        )
        
        # CASTING & SHIFTING
        # x: indices 0 to 2046 (The inputs provided to the model)
        # y: indices 1 to 2047 (The targets the model is asked to predict)
        # mask: indices 1 to 2047 (The mask applied to the prediction of y)
        # This alignment ensures the EOS token (at index n) is supervised 
        # by the mask originally associated with that token in the shard.
        x = torch.from_numpy(raw_tokens[:-1].astype(np.int64))
        y = torch.from_numpy(raw_tokens[1:].astype(np.int64))
        mask = torch.from_numpy(raw_mask[1:].astype(np.float32))
        
        return x, y, mask

def get_dataloader(data_dir, context_length, batch_size, num_workers=2):
    """
    CPU Optimized Dataloader for i5-7300HQ.
    Set num_workers=2 to prevent oversubscription of the 4 physical cores.
    pin_memory=False as there is no GPU device to transfer to.
    """
    dataset = MaskedBinaryShardDataset(data_dir, context_length)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=False, # Disabled for CPU-only training
        drop_last=True
    )
