import torch
import numpy as np
import math
import pickle

att_list = torch.load('att_openweb_small.pt')
breakpoint()
att = torch.stack(att_list)



att_block_size = att.shape[-1]
att_block_size_new = math.ceil(math.sqrt(att_block_size))**2

if att_block_size_new>att_block_size:
    att = torch.nn.functional.pad(att, (att_block_size_new-att_block_size, 0))
att_block_size = att_block_size_new


num_buckets = int(math.sqrt(att_block_size))
bucket_size = num_buckets

att_bucketized = torch.reshape(att, att.shape[:-1]+(num_buckets, bucket_size))
att_bucketized = torch.sum(att_bucketized, dim=-1)



seq_len = att.shape[0]
layers = att.shape[2]
heads = att.shape[3]

sparsity = 0.3

results = []
budget =  round(sparsity*num_buckets)
fixed_window = 3
budget -= 2*fixed_window
for l in range(layers):
    print(l)
    results.append([])
    for h in range(heads):
        results[l].append([])
        for i in range(0, seq_len):
            
            att_bucketized[i, 0, l, h,:fixed_window]+=1
            att_bucketized[i, 0, l, h,-fixed_window:]+=1
            
            _, ind_curr = torch.topk(att_bucketized[i, 1, l, h],budget, dim=-1)
            _, ind_prev = torch.topk(att_bucketized[i, 0, l, h], budget, dim=-1)
            
            att_bucketized[i, 0, l, h,:4]-=1
            att_bucketized[i, 0, l, h,-4:]-=1

            results[l][h].append(torch.sum(att_bucketized[i, 1, l, h][ind_prev]).item())
            


with open('bucketing_accuracy.pickle', 'wb') as handle:
    pickle.dump(results, handle)
