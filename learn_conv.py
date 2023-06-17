import torch
import numpy as np
import math
import pickle

att_list = torch.load('att_openweb_small.pt')
# att = torch.stack(att_list)

seq_len = len(att_list)

window = 200 # conclusion: both sides saturate, left sides goes like 1/x, right side goes like 1/x^2

class Conv_Model(torch.nn.Module):
    def __init__(self, att_block_size):
        super(Conv_Model, self).__init__()
        self.groups=1
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
        else:
            x = torch.transpose(x, 0, 1)
            
            output = self.conv(x) + self.positional_bias

            return torch.transpose(output, 0, 1)

conv_model = Conv_Model(att_list[0].shape[-1])
optimizer = torch.optim.SGD(conv_model.parameters(), lr=0.001)

sparsity=0.2

loss_history=[]
acc_history=[]
for epoch in range(10):
    print(epoch)
    for i in range(seq_len):
        optimizer.zero_grad()

        att = att_list[i].to(torch.float)
        att = att[:,:,:,0,:]
        att_probs = torch.nn.functional.softmax(att,dim=-1)
        flattened = torch.reshape(att_probs, (2, -1, att_probs.shape[-1]))
        flattened = torch.transpose(flattened, 0, 1)
        
        pred = torch.squeeze(conv_model(flattened[:,:1,:]))
        true = flattened[:,1,:]
        
        true_thresholds = torch.unsqueeze(torch.quantile(true, 1-sparsity, dim=-1), dim=-1)
        pred_thresholds = torch.unsqueeze(torch.quantile(pred, 1-sparsity, dim=-1), dim=-1)
        
        true_thresh = torch.sigmoid(5*(true - true_thresholds))
        pred_thresh = torch.sigmoid(5*(pred - pred_thresholds))
        
        loss = torch.mean(torch.square(torch.norm((true_thresh - pred_thresh), dim=-1))) # + torch.mean(torch.square(torch.sum(pred, dim=-1) - 1))
        # loss = torch.mean(torch.square(torch.norm(true - pred, dim=-1)))

        _, true_top_k = torch.topk(true_thresh, k=round(sparsity*true_thresh.shape[-1]), dim=-1)
        _, pred_top_k = torch.topk(pred_thresh, k=round(sparsity*pred_thresh.shape[-1]), dim=-1)
        
        

        true_top_k = torch.zeros_like(true_thresh).scatter_(dim=-1, index=true_top_k, src=torch.ones_like(true_thresh))
        pred_top_k = torch.zeros_like(pred_thresh).scatter_(dim=-1, index=pred_top_k, src=torch.ones_like(pred_thresh))

        acc = torch.mean(torch.sum(true*pred_top_k,dim=-1)/ torch.sum(true*true_top_k,dim=-1))
        acc_history.append(acc.item())
        # acc = torch.sum(torch.abs(pred_top_k-true_top_k), dim=-1)/2
        # acc = 1- (acc/round(sparsity*true_thresh.shape[-1]))
        # acc_history.append(torch.mean(acc).item())
        
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()


import numpy as np
print("np.mean(acc_history[-900:])", np.mean(acc_history[-900:]))
print("np.mean(acc_history[:900])", np.mean(acc_history[:900]))
print("np.mean(loss_history[-900:])", np.mean(loss_history[-900:]))
print("np.mean(loss_history[:900])", np.mean(loss_history[:900]))
# print("conv_model.conv.weight", conv_model.conv.weight)
# print("conv_model.positional_bias", conv_model.positional_bias)
breakpoint()

    