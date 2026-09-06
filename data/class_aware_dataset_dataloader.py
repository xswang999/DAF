import torch.utils.data
from .categorical_dataset import CategoricalSUDataset, CategoricalSSDataset, CategoricalUUDataset, ClassBalancedDataset
from math import ceil as ceil

def collate_fn(data):
    data_collate = {}
    num_classes = len(data)
    keys = data[0].keys()
    for key in keys:
        if key.find('Label') != -1:
            data_collate[key] = [torch.tensor(data[i][key]) for i in range(num_classes)]
        if key.find('Img') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]
        if key.find('Path') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]

    return data_collate

class ClassAwareDataLoader(object):
    def name(self):
        return 'ClassAwareDataLoader'

    def __init__(self, source_batch_size, unlabeled_batch_size,
                source_dataset_root="", unlabeled_paths=[], 
                transform=None, classnames=[], 
                class_set=[], num_selected_classes=0, 
                seed=None, num_workers=0, drop_last=True, 
                sampler='RandomSampler', **kwargs):
        
        # dataset type
        self.dataset = CategoricalSUDataset()

        # dataset parameters
        self.source_dataset_root = source_dataset_root
        self.unlabeled_paths = unlabeled_paths
        self.classnames = classnames
        self.class_set = class_set
        self.source_batch_size = source_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.seed = seed
        self.transform = transform

        # loader parameters
        self.num_selected_classes = min(num_selected_classes, len(class_set))
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.kwargs = kwargs

    def construct(self):
        self.dataset.initialize(source_root=self.source_dataset_root,
                  unlabeled_paths=self.unlabeled_paths,
                  classnames=self.classnames, class_set=self.class_set, 
                  source_batch_size=self.source_batch_size, 
                  unlabeled_batch_size=self.unlabeled_batch_size, 
                  seed=self.seed, transform=self.transform, 
                  **self.kwargs)

        drop_last = self.drop_last
        sampler = getattr(torch.utils.data, self.sampler)(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, 
                                 self.num_selected_classes, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                         batch_sampler=batch_sampler,
                         collate_fn=collate_fn,
                         num_workers=int(self.num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        dataset_len = 0.0
        cid = 0
        for c in self.class_set:
            c_len = max([len(self.dataset.data_paths[d][cid]) // self.dataset.batch_sizes[d][cid] for d in ['source', 'unlabeled']])
            dataset_len += c_len
            cid += 1

        dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
        return dataset_len


class ClassBalancedDataLoader(object):
    def name(self):
        return 'ClassBalancedDataLoader'

    def __init__(self, dataset_root="", transform=None, classnames=[], 
                class_set=[], batch_size=3, num_selected_classes=6,
                seed=None, num_workers=0, drop_last=True, 
                sampler='RandomSampler', **kwargs):
        
        # dataset type
        self.dataset = ClassBalancedDataset()

        # dataset parameters
        self.dataset_root = dataset_root
        self.classnames = classnames
        self.class_set = class_set
        self.batch_size = batch_size
        self.seed = seed
        self.transform = transform

        # loader parameters
        self.num_selected_classes = min(num_selected_classes, len(class_set))
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.kwargs = kwargs

    def construct(self):
        self.dataset.initialize(root=self.dataset_root,
                  classnames=self.classnames, class_set=self.class_set, 
                  batch_size=self.batch_size, 
                  seed=self.seed, transform=self.transform, 
                  **self.kwargs)

        drop_last = self.drop_last
        sampler = getattr(torch.utils.data, self.sampler)(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, 
                                 self.num_selected_classes, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                         batch_sampler=batch_sampler,
                         collate_fn=collate_fn,
                         num_workers=int(self.num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        dataset_len = 0.0
        cid = 0
        for c in self.class_set:
            c_len = len(self.dataset.data_paths[cid]) // self.dataset.batch_size
            dataset_len += c_len
            cid += 1

        dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
        return dataset_len


class SourcePairClassAwareDataLoader(object):
    def name(self):
        return 'SourcePairClassAwareDataLoader'

    def __init__(self, source1_batch_size, source2_batch_size,
                source1_dataset_root="", source2_dataset_root="", 
                transform=None, classnames=[], 
                class_set=[], num_selected_classes=0, 
                seed=None, num_workers=0, drop_last=True, 
                sampler='RandomSampler', **kwargs):
        
        # dataset type
        self.dataset = CategoricalSSDataset()

        # dataset parameters
        self.source1_dataset_root = source1_dataset_root
        self.source2_dataset_root = source2_dataset_root
        self.classnames = classnames
        self.class_set = class_set
        self.source1_batch_size = source1_batch_size
        self.source2_batch_size = source2_batch_size
        self.seed = seed
        self.transform = transform

        # loader parameters
        self.num_selected_classes = min(num_selected_classes, len(class_set))
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.kwargs = kwargs

    def construct(self):
        self.dataset.initialize(source1_root=self.source1_dataset_root,
                  source2_root=self.source2_dataset_root,
                  classnames=self.classnames, class_set=self.class_set, 
                  source1_batch_size=self.source1_batch_size, 
                  source2_batch_size=self.source2_batch_size, 
                  seed=self.seed, transform=self.transform, 
                  **self.kwargs)

        drop_last = self.drop_last
        sampler = getattr(torch.utils.data, self.sampler)(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, 
                                 self.num_selected_classes, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                         batch_sampler=batch_sampler,
                         collate_fn=collate_fn,
                         num_workers=int(self.num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        dataset_len = 0.0
        cid = 0
        for c in self.class_set:
            c_len = max([len(self.dataset.data_paths[d][cid]) // self.dataset.batch_sizes[d][cid] for d in ['source1', 'source2']])
            dataset_len += c_len
            cid += 1

        dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
        return dataset_len


class UnlabeledPairClassAwareDataLoader(object):
    def name(self):
        return 'UnlabeledPairClassAwareDataLoader'

    def __init__(self, unlabeled1_batch_size, unlabeled2_batch_size,
                unlabeled1_paths={}, unlabeled2_paths={}, 
                transform=None, classnames=[], 
                class_set=[], num_selected_classes=0, 
                seed=None, num_workers=0, drop_last=True, 
                sampler='RandomSampler', **kwargs):
        
        # dataset type
        self.dataset = CategoricalUUDataset()

        # dataset parameters
        self.unlabeled1_paths = unlabeled1_paths
        self.unlabeled2_paths = unlabeled2_paths
        self.classnames = classnames
        self.class_set = class_set
        self.unlabeled1_batch_size = unlabeled1_batch_size
        self.unlabeled2_batch_size = unlabeled2_batch_size
        self.seed = seed
        self.transform = transform

        # loader parameters
        self.num_selected_classes = min(num_selected_classes, len(class_set))
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.kwargs = kwargs

    def construct(self):
        self.dataset.initialize(
                  unlabeled1_paths=self.unlabeled1_paths,
                  unlabeled2_paths=self.unlabeled2_paths,
                  classnames=self.classnames, class_set=self.class_set, 
                  unlabeled1_batch_size=self.unlabeled1_batch_size, 
                  unlabeled2_batch_size=self.unlabeled2_batch_size, 
                  seed=self.seed, transform=self.transform, 
                  **self.kwargs)

        drop_last = self.drop_last
        sampler = getattr(torch.utils.data, self.sampler)(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, 
                                 self.num_selected_classes, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                         batch_sampler=batch_sampler,
                         collate_fn=collate_fn,
                         num_workers=int(self.num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        dataset_len = 0.0
        cid = 0
        for c in self.class_set:
            c_len = max([len(self.dataset.data_paths[d][cid]) // self.dataset.batch_sizes[d][cid] for d in ['unlabeled1', 'unlabeled2']])
            dataset_len += c_len
            cid += 1

        dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
        return dataset_len
