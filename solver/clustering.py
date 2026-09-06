import torch
from torch.nn import functional as F
from utils.utils import to_cuda, to_onehot
from scipy.optimize import linear_sum_assignment
from math import ceil

class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type 

    def get_dist(self, pointA, pointB, cross=False): # pointA:feats(498x2048)   pointB:self.centers(31x2048)
        return getattr(self, self.dist_type)(pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert(pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))

class Clustering(object):
    def __init__(self, eps, feat_key, max_len=1000, dist_type='cos'):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = None
        self.stop = False
        self.feat_key = feat_key
        self.max_len = max_len

    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def align_centers(self):
        cost = self.Dist.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def feature_clustering(self, net, loader, threshold=None, current_iteration=0):
        centers = None
        self.stop = False

        self.collect_samples(net, loader)
        feature = self.samples['feature']

        use_pure_center = (threshold is not None) and (current_iteration >= 200)

        refs = to_cuda(torch.LongTensor(range(self.num_classes)).unsqueeze(1))
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)

        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop: break

            centers = 0
            count = 0

            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_labels(cur_feature)
                labels_onehot = to_onehot(labels, self.num_classes)

                if use_pure_center:
                    assigned_dists = dist2center[torch.arange(cur_len, device=cur_feature.device), labels]  # (cur_len,)
                    pass_mask = (assigned_dists < threshold)  # (cur_len,) bool

                    pure_feature = cur_feature[pass_mask]   # (M, feat_dim)
                    pure_labels  = labels[pass_mask]        # (M,)

                    if pure_feature.size(0) > 0:
                        pure_labels_onehot = to_onehot(pure_labels, self.num_classes)  # (M, num_classes)
                        count += torch.sum(pure_labels_onehot, dim=0)  # (num_classes,)
                        pure_labels_u = pure_labels.unsqueeze(0)  # (1, M)
                        mask_c = (pure_labels_u == refs).unsqueeze(2).to(dtype=torch.float, device='cuda')  # (num_classes, M, 1)
                        reshaped_pure = pure_feature.unsqueeze(0)  # (1, M, feat_dim)
                        centers += torch.sum(reshaped_pure * mask_c, dim=1)  # (num_classes, feat_dim)
                else:
                    count += torch.sum(labels_onehot, dim=0) # (31,)
                    labels = labels.unsqueeze(0) # 1x498
                    mask = (labels == refs).unsqueeze(2).to(dtype=torch.float, device='cuda') # refs:31x1
                    reshaped_feature = cur_feature.unsqueeze(0)   # reshaped_feature:1x498x2048
                    # update centers
                    centers += torch.sum(reshaped_feature * mask, dim=1)  # reshaped_feature:1x498x2048，mask：31x498x1
                start += cur_len

            mask = (count.unsqueeze(1) > 0).to(dtype=torch.float, device='cuda')   # mask：31x1
            centers = mask * centers + (1 - mask) * self.init_centers

        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_labels(cur_feature)

            labels_onehot = to_onehot(cur_labels, self.num_classes)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['label'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()
        self.centers = self.centers[cluster2label, :]
        for k in range(num_samples):
            self.samples['label'][k] = cluster2label[self.samples['label'][k]].item()

        self.center_change = torch.mean(self.Dist.get_dist(self.centers, self.init_centers))

        for i in range(num_samples):
            self.path2label[self.samples['data'][i]] = self.samples['label'][i].item()

        del self.samples['feature']

    def _get_multi_source_predictions(self, unlabeled_features):
        source_predictions = {}
        
        for source_name, source_centers in self.source_centers.items():
            dists = self.Dist.get_dist(unlabeled_features, source_centers, cross=True)
            _, predictions = torch.min(dists, dim=1)
            
            min_dists = torch.min(dists, dim=1)[0]
            confidences = 1.0 - min_dists
            
            source_predictions[source_name] = {
                'predictions': predictions,
                'confidences': confidences,
                'distances': min_dists
            }
        
        return source_predictions

    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers)
            dist = torch.mean(dist, dim=0)
            self.stop = dist.item() < self.eps

    def assign_labels(self, feats):
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels

    def collect_samples(self, net, loader):
        data_feat, data_gt, data_paths = [], [], []
        for sample in iter(loader):
            data = sample['Img'].cuda()
            data_paths += sample['Path']
            if 'Label' in sample.keys():
                data_gt += [sample['Label'].cuda()]

            output = net.forward(data)
            feature = output[self.feat_key].data
            data_feat += [feature]

        self.samples['data'] = data_paths
        self.samples['gt'] = torch.cat(data_gt, dim=0) if len(data_gt) > 0 else None
        self.samples['feature'] = torch.cat(data_feat, dim=0)