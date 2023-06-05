import torch
import numpy as np
import math


(att_top_probs_list, att_top_indices_list) = torch.load('att.pt')
att = [[att_top_probs_list[i][j].gather(dim=-1, index = att_top_indices_list[i][j].argsort(dim=-1)) for j in range(len(att_top_probs_list[0]))] for i in range(len(att_top_probs_list))]
att = [torch.stack(att[i]) for i in range(int(len(att)/2))]
att = torch.stack(att)

att_block_size = att.shape[-1]
att_block_size_new = math.ceil(math.sqrt(att_block_size))**2


att = torch.nn.functional.pad(att, (att_block_size_new-att_block_size, 0))
att_block_size = att_block_size_new


num_buckets = int(math.sqrt(att_block_size))
bucket_size = num_buckets

att_bucketized = torch.reshape(att, att.shape[:-1]+(num_buckets, bucket_size))
att_bucketized = torch.sum(att_bucketized, dim=-1)



seq_len = att.shape[0]
layers = att.shape[1]
heads = att.shape[3]

sparsity = 0.5

results = {}
budget =  round(sparsity*num_buckets)
for l in range(layers):
    print(l)
    results[l] = {}
    for h in range(heads):
        results[l][h]=[]
        for i in range(1, seq_len):
            att_bucketized[i-1, l, 0, h,:4]+=1
            att_bucketized[i-1, l, 0, h,-4:]+=1
            _, ind_curr = torch.topk(att_bucketized[i, l, 0, h],budget, dim=-1)
            _, ind_prev = torch.topk(att_bucketized[i-1, l, 0, h], budget, dim=-1)
            
            att_bucketized[i-1, l, 0, h,:4]-=1
            att_bucketized[i-1, l, 0, h,-4:]-=1

            results[l][h].append(torch.sum(att_bucketized[i, l, 0, h][ind_prev]).item())
            

breakpoint()
