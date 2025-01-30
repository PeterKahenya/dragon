import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import torch
import os
import numpy as np

class BeeHummingBirdLMDataset(TorchDataset):

    def __init__(self, shards_path, T, split, total_tokens):
        self.T = T
        self.total_tokens = total_tokens
        assert split in {'train', 'val'}, f"split {split} must be either 'train' or 'val'"
        shards = os.listdir(shards_path)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(shards_path, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])

    def __len__(self):
        return self.total_tokens//(self.T+1)

    def _load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def __getitem__(self, idx):
        while len(self.tokens) < (self.T + 1): # not enough tokens left in this shard
            self.current_shard = (self.current_shard + 1) % len(self.shards) # move to next shard
            self.tokens = torch.cat([self.tokens, self._load_tokens(self.shards[self.current_shard])], dim=0) # load next shard and append to remaining tokens
        buf = self.tokens[:self.T+1]
        X = buf[:-1]
        Y = buf[1:]
        self.tokens = self.tokens[self.T+1:] # move window
        return X, Y

# Dataset for distributed training
class BeeHummingBirdLMDatasetDistrib(TorchDataset):
    def __init__(self, shards_path, B, T, process_rank, num_processes, split, master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(shards_path)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(shards_path, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
        
    def load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt
    
    def reset(self):
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y