import os
from .image_folder import make_dataset_with_labels, make_dataset_classwise
from PIL import Image
from torch.utils.data import Dataset
import random
from math import ceil
import torch

class CategoricalDataset(Dataset):
    def __init__(self):
        super(CategoricalDataset, self).__init__()

    def initialize(self, root, classnames, class_set, 
                  batch_size, seed=None, transform=None, 
                  **kwargs):

        self.root = root
        self.transform = transform
        self.class_set = class_set
        
        self.data_paths = {}
        self.data_paths[self.root] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths[self.root][cid] = make_dataset_classwise(self.root, c)
            cid += 1

        self.seed = seed
        self.classnames = classnames
        
        self.rng = random.Random(seed) if seed is not None else random.Random()

        self.batch_sizes = {}
        self.batch_sizes[self.root] = {}
        cid = 0
        for c in self.class_set:
            batch_size = batch_size
            self.batch_sizes[self.root][cid] = min(batch_size, len(self.data_paths[self.root][cid]))
            cid += 1

    def __getitem__(self, index):
        data = {}
        root = self.root
        cur_paths = self.data_paths[root]
        
        inds = self.rng.sample(range(len(cur_paths[index])), self.batch_sizes[root][index])

        path = [cur_paths[index][ind] for ind in inds]
        data['Path'] = path
        assert(len(path) > 0)
        for p in path:
            img = Image.open(p).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)

            if 'Img' not in data:
                data['Img'] = [img]
            else:
                data['Img'] += [img]

        data['Label'] = [self.classnames.index(self.class_set[index])] * len(data['Img'])
        data['Img'] = torch.stack(data['Img'], dim=0)
        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'CategoricalDataset'

class CategoricalSUDataset(Dataset):
    def __init__(self):
        super(CategoricalSUDataset, self).__init__()

    def initialize(self, source_root, unlabeled_paths,
                  classnames, class_set, 
                  source_batch_size, 
                  unlabeled_batch_size, seed=None, 
                  transform=None, **kwargs):

        self.source_root = source_root
        self.unlabeled_paths = unlabeled_paths

        self.transform = transform
        self.class_set = class_set
        
        self.data_paths = {}
        self.data_paths['source'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['source'][cid] = make_dataset_classwise(self.source_root, c)
            cid += 1

        self.data_paths['unlabeled'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['unlabeled'][cid] = self.unlabeled_paths[c]
            cid += 1

        self.seed = seed
        self.classnames = classnames
        
        source_name = os.path.basename(source_root)
        domain_seed = seed + hash(source_name) % 1000 if seed is not None else None
        self.rng = random.Random(domain_seed) if domain_seed is not None else random.Random()

        self.batch_sizes = {}
        for d in ['source', 'unlabeled']:
            self.batch_sizes[d] = {}
            cid = 0
            for c in self.class_set:
                batch_size = source_batch_size if d == 'source' else unlabeled_batch_size
                self.batch_sizes[d][cid] = min(batch_size, len(self.data_paths[d][cid]))
                cid += 1


    def __getitem__(self, index):
        data = {}
        for d in ['source', 'unlabeled']:
            cur_paths = self.data_paths[d]
            
            inds = self.rng.sample(range(len(cur_paths[index])), self.batch_sizes[d][index])

            path = [cur_paths[index][ind] for ind in inds]
            data['Path_'+d] = path
            assert(len(path) > 0)
            for p in path:
                img = Image.open(p).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)

                if 'Img_'+d not in data:
                    data['Img_'+d] = [img]
                else:
                    data['Img_'+d] += [img]

            data['Label_'+d] = [self.classnames.index(self.class_set[index])] * len(data['Img_'+d])
            data['Img_'+d] = torch.stack(data['Img_'+d], dim=0)

        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'CategoricalSUDataset'


class CategoricalSSDataset(Dataset):
    def __init__(self):
        super(CategoricalSSDataset, self).__init__()

    def initialize(self, source1_root, source2_root,
                  classnames, class_set, 
                  source1_batch_size, 
                  source2_batch_size, seed=None, 
                  transform=None, **kwargs):

        self.source1_root = source1_root
        self.source2_root = source2_root

        self.transform = transform
        self.class_set = class_set
        
        self.data_paths = {}
        self.data_paths['source1'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['source1'][cid] = make_dataset_classwise(self.source1_root, c)
            cid += 1

        self.data_paths['source2'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['source2'][cid] = make_dataset_classwise(self.source2_root, c)
            cid += 1

        self.seed = seed
        self.classnames = classnames
        
        source1_name = os.path.basename(source1_root)
        source2_name = os.path.basename(source2_root)
        pair_seed = seed + hash(source1_name + source2_name) % 1000 if seed is not None else None
        self.rng = random.Random(pair_seed) if pair_seed is not None else random.Random()

        self.batch_sizes = {}
        for d in ['source1', 'source2']:
            self.batch_sizes[d] = {}
            cid = 0
            for c in self.class_set:
                batch_size = source1_batch_size if d == 'source1' else source2_batch_size
                self.batch_sizes[d][cid] = min(batch_size, len(self.data_paths[d][cid]))
                cid += 1


    def __getitem__(self, index):
        data = {}
        for d in ['source1', 'source2']:
            cur_paths = self.data_paths[d]
            
            inds = self.rng.sample(range(len(cur_paths[index])), self.batch_sizes[d][index])

            path = [cur_paths[index][ind] for ind in inds]
            data['Path_'+d] = path
            assert(len(path) > 0)
            for p in path:
                img = Image.open(p).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)

                if 'Img_'+d not in data:
                    data['Img_'+d] = [img]
                else:
                    data['Img_'+d] += [img]

            data['Label_'+d] = [self.classnames.index(self.class_set[index])] * len(data['Img_'+d])
            data['Img_'+d] = torch.stack(data['Img_'+d], dim=0)

        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'CategoricalSSDataset'


class CategoricalUUDataset(Dataset):
    def __init__(self):
        super(CategoricalUUDataset, self).__init__()

    def initialize(self, unlabeled1_paths, unlabeled2_paths,
                  classnames, class_set, 
                  unlabeled1_batch_size, 
                  unlabeled2_batch_size, seed=None, 
                  transform=None, **kwargs):
        self.unlabeled1_paths = unlabeled1_paths
        self.unlabeled2_paths = unlabeled2_paths

        self.transform = transform
        self.class_set = class_set
        
        self.data_paths = {}
        self.data_paths['unlabeled1'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['unlabeled1'][cid] = self.unlabeled1_paths[c]
            cid += 1

        self.data_paths['unlabeled2'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['unlabeled2'][cid] = self.unlabeled2_paths[c]
            cid += 1

        self.seed = seed
        self.classnames = classnames
        
        pair_seed = seed + 9999 if seed is not None else None
        self.rng = random.Random(pair_seed) if pair_seed is not None else random.Random()

        self.batch_sizes = {}
        for d in ['unlabeled1', 'unlabeled2']:
            self.batch_sizes[d] = {}
            cid = 0
            for c in self.class_set:
                batch_size = unlabeled1_batch_size if d == 'unlabeled1' else unlabeled2_batch_size
                self.batch_sizes[d][cid] = min(batch_size, len(self.data_paths[d][cid]))
                cid += 1


    def __getitem__(self, index):
        data = {}
        for d in ['unlabeled1', 'unlabeled2']:
            cur_paths = self.data_paths[d]
            
            inds = self.rng.sample(range(len(cur_paths[index])), self.batch_sizes[d][index])

            path = [cur_paths[index][ind] for ind in inds]
            data['Path_'+d] = path
            assert(len(path) > 0)
            for p in path:
                img = Image.open(p).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)

                if 'Img_'+d not in data:
                    data['Img_'+d] = [img]
                else:
                    data['Img_'+d] += [img]

            data['Label_'+d] = [self.classnames.index(self.class_set[index])] * len(data['Img_'+d])
            data['Img_'+d] = torch.stack(data['Img_'+d], dim=0)

        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'CategoricalUUDataset'


class PseudoLabelDataset(Dataset):
    def __init__(self):
        super(PseudoLabelDataset, self).__init__()

    def initialize(self, samples_dict, classnames, transform=None):
        self.samples_paths = samples_dict['data']
        self.samples_labels = samples_dict['label']
        self.transform = transform
        self.classnames = classnames
        
        self.class_samples = {}
        for path, label in zip(self.samples_paths, self.samples_labels):
            label_item = label.item() if hasattr(label, 'item') else label
            if label_item not in self.class_samples:
                self.class_samples[label_item] = []
            self.class_samples[label_item].append(path)

    def __getitem__(self, index):
        path = self.samples_paths[index]
        label = self.samples_labels[index]
        label_item = label.item() if hasattr(label, 'item') else label
        
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return {
            'Img': img,
            'Label': label_item,
            'Path': path
        }
    
    def __len__(self):
        return len(self.samples_paths)
    
    def name(self):
        return 'PseudoLabelDataset'


class ClassBalancedDataset(Dataset):
    def __init__(self):
        super(ClassBalancedDataset, self).__init__()

    def initialize(self, root, classnames, class_set, 
                  batch_size, seed=None, transform=None, 
                  **kwargs):
        self.root = root
        self.transform = transform
        self.class_set = class_set
        self.classnames = classnames
        
        self.data_paths = {}
        for cid, c in enumerate(self.class_set):
            self.data_paths[cid] = make_dataset_classwise(self.root, c)
        
        self.rng = random.Random(seed) if seed is not None else random.Random()
        
        self.batch_size = batch_size

    def __getitem__(self, index):
        data = {}
        cur_paths = self.data_paths[index]
        
        num_samples = min(self.batch_size, len(cur_paths))
        inds = self.rng.sample(range(len(cur_paths)), num_samples)
        
        path = [cur_paths[ind] for ind in inds]
        data['Path'] = path
        
        for p in path:
            img = Image.open(p).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
            
            if 'Img' not in data:
                data['Img'] = [img]
            else:
                data['Img'] += [img]
        
        data['Label'] = [self.classnames.index(self.class_set[index])] * len(data['Img'])
        data['Img'] = torch.stack(data['Img'], dim=0)
        
        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'ClassBalancedDataset'


class ClassBalancedPseudoLabelDataset(Dataset):
    def __init__(self):
        super(ClassBalancedPseudoLabelDataset, self).__init__()

    def initialize(self, samples_dict, classnames, class_set,
                  batch_size, seed=None, transform=None):
        self.classnames = classnames
        self.class_set = class_set
        self.transform = transform
        self.batch_size = batch_size
        
        self.data_paths = {}
        for cid, class_idx in enumerate(self.class_set):
            self.data_paths[cid] = []
        
        for path, label in zip(samples_dict['data'], samples_dict['label']):
            label_item = label.item() if hasattr(label, 'item') else label
            if label_item in self.class_set:
                cid = self.class_set.index(label_item)
                self.data_paths[cid].append(path)
        
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def __getitem__(self, index):
        data = {}
        cur_paths = self.data_paths[index]
        
        num_samples = min(self.batch_size, len(cur_paths))
        inds = self.rng.sample(range(len(cur_paths)), num_samples)
        
        path = [cur_paths[ind] for ind in inds]
        data['Path'] = path
        
        for p in path:
            img = Image.open(p).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            
            if 'Img' not in data:
                data['Img'] = [img]
            else:
                data['Img'] += [img]
        
        class_idx = self.class_set[index]
        data['Label'] = [class_idx] * len(data['Img'])
        data['Img'] = torch.stack(data['Img'], dim=0)
        
        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'ClassBalancedPseudoLabelDataset'

