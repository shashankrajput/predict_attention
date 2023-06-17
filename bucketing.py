import torch
import numpy as np
import math
import pickle

att_list = torch.load('att_openweb_small.pt')
# att = torch.stack(att_list)
conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=1, bias=False, padding_mode='reflect').cuda()
positional_bias = torch.zeros(att_list[0].shape[-1], requires_grad=True).cuda()
seq_len = len(att_list)
results = {}
for i in range(seq_len):
    print(i)
    att = att_list[i]

    att_probs = torch.nn.functional.softmax(att,dim=-1)

    att_block_size = att.shape[-1]
    att_block_size_new = math.ceil(math.sqrt(att_block_size))**2

    if att_block_size_new>att_block_size:
        att = torch.nn.functional.pad(att, (att_block_size_new-att_block_size, 0))
    att_block_size = att_block_size_new


    num_buckets = int(math.sqrt(att_block_size))
    bucket_size = num_buckets

    
    att_bucketized = torch.reshape(att, att.shape[:-1]+(num_buckets, bucket_size))

    att_probs_bucketized = torch.reshape(att_probs, att_probs.shape[:-1]+(num_buckets, bucket_size))
    att_probs_bucketized_norm = torch.norm(att_probs_bucketized, dim=-1)
    



    layers = att.shape[1]
    heads = att.shape[2]

    
        
    for l in range(layers):
        if l not in results:
            results[l] = {}
        for h in range(heads):
            if h not in results[l]:
                results[l][h]=[]

            att_probs_bucketized_norm[0, l, h, 0,:3] +=1
            att_probs_bucketized_norm[0, l, h, 0,-3:] +=1

            _, ind_prev = torch.topk(att_probs_bucketized_norm[0, l, h, 0], num_buckets, dim=-1)

            ind_prev_top = ind_prev[:num_buckets//3]
            ind_prev_mid = ind_prev[num_buckets//3:2*num_buckets//3]
            # ind_prev_bot = ind_prev[2*num_buckets//3:]
            
            approx_attn = torch.full(att_bucketized[1, l, h, 0].shape, fill_value=float('-inf')).to(torch.bfloat16).cuda()
            approx_attn[ind_prev_top] = att_bucketized[1, l, h, 0][ind_prev_top]
            approx_attn[ind_prev_mid] = att_bucketized[1, l, h, 1][ind_prev_mid]

            
            approx_attn_prob = torch.nn.functional.softmax(torch.flatten(approx_attn,start_dim=-2),dim=-1)
            

            error = torch.norm(att_probs[1, l, h, 0] - approx_attn_prob)
            
            results[l][h].append(error.item())

            # err_mid = att_bucketized[1, l, h, 0][ind_prev_mid]-att_bucketized[1, l, h, 1][ind_prev_mid]
            # err_bot = att_bucketized[1, l, h, 0][ind_prev_bot]-att_bucketized[1, l, h, 2][ind_prev_bot]
            # results[l][h].append(torch.norm(torch.tensor([torch.norm(err_mid), torch.norm(err_bot)])).item()/ torch.norm(att_bucketized[1, l, h, 0]).item())


breakpoint()
            


with open('bucketing_accuracy.pickle', 'wb') as handle:
    pickle.dump(results, handle)
