from torch import nn
import torch.nn as nn
from utils.utils import to_cuda
import torch

class C_BHCC(object):
    def __init__(self, num_layers, kernel_num, kernel_mul, 
                 num_classes, intra_only=False, **kwargs):

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.num_classes = num_classes
        self.intra_only = intra_only or (self.num_classes==1)
        self.num_layers = num_layers
    
    def split_classwise(self, dist, nums):
        num_classes = len(nums)
        start = end = 0
        dist_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            dist_c = dist[start:end, start:end]
            dist_list += [dist_c]
        return dist_list

    def gamma_estimation(self, dist):
        dist_sum = torch.sum(dist['ss']) + torch.sum(dist['tt']) + 2 * torch.sum(dist['st'])
        bs_S = dist['ss'].size(0)
        bs_T = dist['tt'].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N 
        return gamma

    def patch_gamma_estimation(self, nums_S, nums_T, dist):
        assert(len(nums_S) == len(nums_T))
        num_classes = len(nums_S)
        patch = {}
        gammas = {}
        gammas['st'] = to_cuda(torch.zeros_like(dist['st'], requires_grad=False))
        gammas['ss'] = [] 
        gammas['tt'] = [] 
        for c in range(num_classes):
            gammas['ss'] += [to_cuda(torch.zeros([num_classes], requires_grad=False))]
            gammas['tt'] += [to_cuda(torch.zeros([num_classes], requires_grad=False))]

        source_start = source_end = 0
        for ns in range(num_classes):
            source_start = source_end
            source_end = source_start + nums_S[ns]
            patch['ss'] = dist['ss'][ns]
            target_start = target_end = 0
            for nt in range(num_classes):
                target_start = target_end 
                target_end = target_start + nums_T[nt] 
                patch['tt'] = dist['tt'][nt]

                patch['st'] = dist['st'].narrow(0, source_start, 
                       nums_S[ns]).narrow(1, target_start, nums_T[nt])
                gamma = self.gamma_estimation(patch)
                gammas['ss'][ns][nt] = gamma
                gammas['tt'][nt][ns] = gamma
                gammas['st'][source_start:source_end, target_start:target_end] = gamma
        return gammas

    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul):
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul**i) for i in range(kernel_num)]
        gamma_tensor = to_cuda(torch.stack(gamma_list, dim=0))

        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).to(dtype=torch.float, device='cuda')
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps
        gamma_tensor = gamma_tensor.detach()

        for i in range(len(gamma_tensor.size()) - len(dist.size())):
            dist = dist.unsqueeze(0)

        dist = dist / gamma_tensor
        upper_mask = (dist > 1e5).to(dtype=torch.float, device='cuda').detach()
        lower_mask = (dist < 1e-5).to(dtype=torch.float, device='cuda').detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dist = normal_mask * dist + upper_mask * 1e5 + lower_mask * 1e-5
        kernel_val = torch.sum(torch.exp(-1.0 * dist), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers, key, category=None):
        num_layers = self.num_layers
        kernel_dist = None
        for i in range(num_layers):

            dist = dist_layers[i][key] if category is None else dist_layers[i][key][category]

            gamma = gamma_layers[i][key] if category is None else gamma_layers[i][key][category]

            cur_kernel_num = self.kernel_num[i]
            cur_kernel_mul = self.kernel_mul[i]

            if kernel_dist is None:
                kernel_dist = self.compute_kernel_dist(dist, gamma, cur_kernel_num, cur_kernel_mul)

                continue

            kernel_dist += self.compute_kernel_dist(dist, gamma, cur_kernel_num, cur_kernel_mul)
        
        kernel_dist = kernel_dist / num_layers
        
        return kernel_dist

    def patch_mean(self, nums_row, nums_col, dist):
        assert(len(nums_row) == len(nums_col))
        num_classes = len(nums_row)

        mean_tensor = to_cuda(torch.zeros([num_classes, num_classes]))
        row_start = row_end = 0
        for row in range(num_classes):
            row_start = row_end
            row_end = row_start + nums_row[row]

            col_start = col_end = 0
            for col in range(num_classes):
                col_start = col_end
                col_end = col_start + nums_col[col]
                val = torch.mean(dist.narrow(0, row_start,
                           nums_row[row]).narrow(1, col_start, nums_col[col]))
                mean_tensor[row, col] = val
        return mean_tensor

    def compute_paired_dist(self, A, B):
        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dist = (((A_expand - B_expand))**2).sum(2)
        return dist

    def forward(self, source, target, nums_S, nums_T):
        assert(len(nums_S) == len(nums_T)), "The number of classes for source (%d) and target (%d) should be the same." % (len(nums_S), len(nums_T))

        num_classes = len(nums_S)

        # compute the dist 
        dist_layers = []
        gamma_layers = []

        for i in range(self.num_layers):

            cur_source = source[i]
            cur_target = target[i]

            dist = {}
            dist['ss'] = self.compute_paired_dist(cur_source, cur_source)
            dist['tt'] = self.compute_paired_dist(cur_target, cur_target)
            dist['st'] = self.compute_paired_dist(cur_source, cur_target)

            dist['ss'] = self.split_classwise(dist['ss'], nums_S)
            dist['tt'] = self.split_classwise(dist['tt'], nums_T)
            dist_layers += [dist]

            gamma_layers += [self.patch_gamma_estimation(nums_S, nums_T, dist)]
        # compute the kernel dist
        for i in range(self.num_layers):
            for c in range(num_classes):
                gamma_layers[i]['ss'][c] = gamma_layers[i]['ss'][c].view(num_classes, 1, 1)
                gamma_layers[i]['tt'][c] = gamma_layers[i]['tt'][c].view(num_classes, 1, 1)
        kernel_dist_st = self.kernel_layer_aggregation(dist_layers, gamma_layers, 'st')
        kernel_dist_st = self.patch_mean(nums_S, nums_T, kernel_dist_st)
        kernel_dist_ss = []
        kernel_dist_tt = []
        for c in range(num_classes):
            kernel_dist_ss += [torch.mean(self.kernel_layer_aggregation(dist_layers, gamma_layers, 'ss', c).view(num_classes, -1), dim=1)] # dist_layers['ss']:3x3，gamma_layers['ss']:10x1x1
            kernel_dist_tt += [torch.mean(self.kernel_layer_aggregation(dist_layers, gamma_layers, 'tt', c).view(num_classes, -1), dim=1)]
        kernel_dist_ss = torch.stack(kernel_dist_ss, dim=0)
        kernel_dist_tt = torch.stack(kernel_dist_tt, dim=0).transpose(1, 0)

        mmds = kernel_dist_ss + kernel_dist_tt - 2 * kernel_dist_st
        
        
        intra_distances = torch.diagonal(mmds)
        max_intra_idx = torch.argmax(intra_distances)
        max_intra_value = mmds[max_intra_idx, max_intra_idx]
        
        row_distances = mmds[max_intra_idx, :].clone()
        row_distances[max_intra_idx] = float('inf')
        min_inter_from_row = torch.min(row_distances)
        
        col_distances = mmds[:, max_intra_idx].clone()
        col_distances[max_intra_idx] = float('inf')
        min_inter_from_col = torch.min(col_distances)
        
        c_hcc_row = max_intra_value - min_inter_from_row
        c_hcc_col = max_intra_value - min_inter_from_col
        c_bhcc = (c_hcc_row + c_hcc_col) / 2.0
        
        return {
            'c_bhcc': c_bhcc,
        }