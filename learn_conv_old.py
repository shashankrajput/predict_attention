import torch
import numpy as np
import math
import pickle

att_list = torch.load('att_openweb_small.pt')
# att = torch.stack(att_list)

seq_len = len(att_list)

 #window conclusion: both sides saturate, left sides goes like 1/x, right side goes like 1/x^2


# class conv_module(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, window, stride=1, groups=1, positional_bias_size=10):
#         super(conv_module, self).__init__()
#         self.conv_left = torch.nn.Conv1d(in_channels, out_channels, groups=groups, kernel_size=2*window+1, stride=1, padding=window, bias=True)
        
#         left_decay = [1/(window-i) if i<window else 0 for i in range(2*window + 1)]

#         self.conv_left.weight = torch.nn.Parameter(torch.tensor([left_decay for _ in range(in_channels)]).unsqueeze(1).unsqueeze(1), requires_grad=False)
#         left_bias = [1/(window-i) if i<window else 0 for i in range(2*window + 1)]
#         self.conv_left.weight.requires_grad=False
        
        
#         self.conv_center = torch.nn.Conv1d(in_channels, out_channels, 3, stride, 1, dilation, groups, bias, padding_mode, device, dtype)
        

        
#         self.positional_bias_beg = torch.nn.Parameter(torch.zeros(positional_bias_size, requires_grad=True))
#         self.positional_bias_end = torch.nn.Parameter(torch.zeros(positional_bias_size, requires_grad=True))


class Conv_Model(torch.nn.Module):
    def __init__(self, att_block_size, window = 0):
        super(Conv_Model, self).__init__()
        self.groups= 1200
        if self.groups==1:
            self.conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2*window+1, stride=1, padding=window, bias=False).cuda()
        else:
            self.conv = torch.nn.Conv1d(in_channels=self.groups, out_channels=self.groups, kernel_size=2*window+1, stride=1, padding=window, bias=False, groups=self.groups).cuda()
        with torch.no_grad():
            self.conv.weight = torch.nn.Parameter(torch.zeros_like(self.conv.weight))
        if self.groups==1:
            self.positional_bias = torch.nn.Parameter(torch.zeros(att_block_size, requires_grad=True).cuda())
        else:
            self.positional_bias = torch.nn.Parameter(torch.zeros((1, self.groups, att_block_size), requires_grad=True).cuda())

    def forward(self, x):
        if self.groups==1:
            return self.conv(x) + self.positional_bias
            # return self.positional_bias
        else:
            x = torch.transpose(x, 0, 1)
            
            output = self.conv(x) + self.positional_bias
            # output = self.positional_bias

            return torch.transpose(output, 0, 1)


att_block_size=att_list[0].shape[-1]

#
num_buckets = int(math.sqrt(att_block_size))
bucket_size = num_buckets
att_block_size = num_buckets
#

conv_model = Conv_Model(att_block_size)




optimizer = torch.optim.SGD(conv_model.parameters(), lr=0.1)

sparsity=0.3

loss_history=[]
acc_history=[]
for epoch in range(10):
    print(epoch)
    for i in range(seq_len):
        optimizer.zero_grad()

        att = att_list[i].to(torch.float)
        att = att[:,:,:,0,:]
        att_probs = torch.nn.functional.softmax(att,dim=-1)

        #
        att_bucketized = torch.reshape(att, att.shape[:-1]+(num_buckets, bucket_size))

        att_probs_bucketized = torch.reshape(att_probs, att_probs.shape[:-1]+(num_buckets, bucket_size))
        att_probs = torch.norm(att_probs_bucketized, dim=-1)
        #

        flattened = torch.reshape(att_probs, (2, -1, att_probs.shape[-1]))
        flattened = torch.transpose(flattened, 0, 1)
        
        pred = torch.squeeze(conv_model(flattened[:,:1,:]))
        true = flattened[:,1,:]
        
        true_thresholds = torch.unsqueeze(torch.quantile(true, 1-sparsity, dim=-1), dim=-1)
        pred_thresholds = torch.unsqueeze(torch.quantile(pred, 1-sparsity, dim=-1), dim=-1)
        
        true_thresh = torch.sigmoid((epoch+1)*(true - true_thresholds))
        pred_thresh = torch.sigmoid((epoch+1)*(pred - pred_thresholds))
        
        # loss = torch.mean(torch.square(torch.norm((true_thresh - pred_thresh)*(true**((epoch+1)/10)), dim=-1))) # + torch.mean(torch.square(torch.sum(pred, dim=-1) - 1))
        loss = torch.mean(torch.square(torch.norm(true - pred, dim=-1)))

        _, true_top_k = torch.topk(true, k=round(sparsity*true.shape[-1]), dim=-1)
        _, pred_top_k = torch.topk(pred, k=round(sparsity*pred.shape[-1]), dim=-1)
        
        

        true_top_k = torch.zeros_like(true).scatter_(dim=-1, index=true_top_k, src=torch.ones_like(true))
        pred_top_k = torch.zeros_like(pred).scatter_(dim=-1, index=pred_top_k, src=torch.ones_like(pred))

        acc = torch.mean(torch.sum(true*pred_top_k,dim=-1)/ torch.sum(true*true_top_k,dim=-1))
        acc_history.append(acc.item())
        # acc = torch.sum(torch.abs(pred_top_k-true_top_k), dim=-1)/2
        # acc = 1- (acc/round(sparsity*true_thresh.shape[-1]))
        # acc_history.append(torch.mean(acc).item())
        
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
    # optimizer.param_groups[0]['lr'] *= (1+epoch)/(2+epoch)


import numpy as np
print("np.mean(acc_history[-900:])", np.mean(acc_history[-900:]))
print("np.mean(acc_history[:900])", np.mean(acc_history[:900]))
print("np.mean(loss_history[-900:])", np.mean(loss_history[-900:]))
print("np.mean(loss_history[:900])", np.mean(loss_history[:900]))
# print("conv_model.conv.weight", conv_model.conv.weight)
# print("conv_model.positional_bias", conv_model.positional_bias)
breakpoint()

    