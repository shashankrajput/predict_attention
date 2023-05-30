import torch
import numpy as np

(att_top_probs_list, att_top_indices_list) = torch.load('att.pt')

n_layer = len(att_top_probs_list[0])
n_head = 1 # att_top_probs_list[0][0].shape[1]
block_size = 1024

class Att_Pred_Module(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.w_list = torch.nn.ModuleDict({str(i): torch.nn.Linear(1, 1, bias=False) for i in range(-2, 2)})
        for i in self.w_list.keys():
            self.w_list[i].weight.data = torch.zeros_like(self.w_list[i].weight.data)
        # self.b = torch.nn.Parameter(torch.zeros([block_size, 1],  requires_grad = True))

    def forward(self, prev_probs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        
        output = None        
        for i in self.w_list.keys():
            i = int(i)
            prev_probs_shifted = torch.roll(prev_probs, shifts=i, dims=-1)
            if i<0:
                prev_probs_shifted[..., i:] = 0.0
            elif i>0:
                prev_probs_shifted[..., :i] = 0.0
        
            prev_probs_shifted = prev_probs_shifted.unsqueeze(-1)
            if output is None:
                output = self.w_list[str(i)](prev_probs_shifted)
            else:
                output = output + self.w_list[str(i)](prev_probs_shifted)
            
        return output # + self.b


model = [[Att_Pred_Module().cuda() for j in range(n_head)] for i in range(n_layer)]



results_1={}
results_2={}
for l in range(1, n_layer):
    results_1[l] = {}
    results_2[l] = {}
    for h in range(n_head):
        critereon = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model[l][h].parameters(), lr=0.01)
        for epoch in range(10):
            results_1[l][h] = []
            results_2[l][h] = []
            for i in range(1, len(att_top_probs_list)):
                probs_curr = torch.zeros(block_size, requires_grad=False).cuda()
                probs_curr[att_top_indices_list[i][l][0][h]] = att_top_probs_list[i][l][0][h]
                probs_curr = probs_curr.unsqueeze(-1)
                                    
                probs_prev = torch.zeros(block_size, requires_grad=False).cuda()
                probs_prev[att_top_indices_list[i-1][l][0][h]] = att_top_probs_list[i-1][l][0][h]
                
                
                optimizer.zero_grad()
                probs_pred = model[l][h](probs_prev)
                
                loss = critereon(probs_pred, probs_curr)
                
                loss.backward()
                
                optimizer.step()
                
                if i % 1000 == 0:
                    print(model[l][h].w_1.weight.item(), model[l][h].w_2.weight.item())
                
                
                _, pred_indices = torch.topk(probs_pred.squeeze(), 100)
                true_indices = att_top_indices_list[i][l][0][h].detach().clone().cpu().numpy()
                pred_indices = pred_indices.detach().clone().cpu().numpy()
                results_1[l][h].append(probs_curr[np.intersect1d(pred_indices, true_indices)].sum().item() / probs_curr.sum().item())
                results_2[l][h].append(probs_curr[np.intersect1d(pred_indices, true_indices)].sum().item())
                           


breakpoint()