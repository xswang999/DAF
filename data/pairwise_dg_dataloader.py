import torch
import random
from PIL import Image
from torch.utils.data import Dataset


class PairwiseDomainGeneralizationDataset(Dataset):
    
    def __init__(self, source_pairs, labeled_sources, unlabeled_sources,
                 labeled_datasets, filtered_samples_dict,
                 num_classes_per_batch=6, samples_per_class=3,
                 transform=None):
        self.source_pairs = source_pairs
        self.labeled_sources = labeled_sources
        self.unlabeled_sources = unlabeled_sources
        self.labeled_datasets = labeled_datasets
        self.filtered_samples_dict = filtered_samples_dict
        self.num_classes = num_classes_per_batch
        self.samples_per_class = samples_per_class
        self.transform = transform
        
        self._class_to_indices = {}
        self._filtered_class_to_indices = {}

        for source_name in labeled_sources:
            self._build_class_to_indices(source_name)
        
        for unlabeled_source in unlabeled_sources:
            self._build_filtered_class_to_indices(unlabeled_source)
        
        self.pair_common_classes = {}
        for domain1, domain2 in source_pairs:
            common_classes = self._compute_pair_common_classes(domain1, domain2)
            self.pair_common_classes[(domain1, domain2)] = common_classes

    def _compute_pair_common_classes(self, domain1, domain2):
        if domain1 in self.labeled_sources:
            domain1_classes = {cls for cls, indices in self._class_to_indices[domain1].items()
                             if len(indices) >= self.samples_per_class}
        else:
            domain1_classes = set(self._filtered_class_to_indices[domain1].keys())
        
        if domain2 in self.labeled_sources:
            domain2_classes = {cls for cls, indices in self._class_to_indices[domain2].items()
                             if len(indices) >= self.samples_per_class}
        else:
            domain2_classes = set(self._filtered_class_to_indices[domain2].keys())
        
        common_classes = sorted(list(domain1_classes & domain2_classes))
        return common_classes
    
    def _build_class_to_indices(self, source_name):
        dataset = self.labeled_datasets[source_name]
        self._class_to_indices[source_name] = {}
        
        for idx in range(len(dataset)):
            label = None
            if hasattr(dataset, 'data_labels'):
                label = dataset.data_labels[idx]
            elif hasattr(dataset, 'samples'):
                _, label = dataset.samples[idx]
            elif hasattr(dataset, 'imgs'):
                _, label = dataset.imgs[idx]
            elif hasattr(dataset, 'targets'):
                label = dataset.targets[idx]
            else:
                data = dataset[idx]
                if isinstance(data, dict):
                    label = data.get('Label')
                elif len(data) == 2:
                    _, label = data
                elif len(data) == 3:
                    _, label, _ = data
                else:
                    raise ValueError(f"Unexpected dataset output")
            
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            if not isinstance(label, int):
                continue
            
            if label not in self._class_to_indices[source_name]:
                self._class_to_indices[source_name][label] = []
            self._class_to_indices[source_name][label].append(idx)
    
    def _build_filtered_class_to_indices(self, unlabeled_source):
        if unlabeled_source not in self.filtered_samples_dict:
            self._filtered_class_to_indices[unlabeled_source] = {}
            return
        
        filtered_labels = self.filtered_samples_dict[unlabeled_source]['label']
        self._filtered_class_to_indices[unlabeled_source] = {}
        
        for idx, label in enumerate(filtered_labels):
            label_item = label.item() if torch.is_tensor(label) else label
            if label_item not in self._filtered_class_to_indices[unlabeled_source]:
                self._filtered_class_to_indices[unlabeled_source][label_item] = []
            self._filtered_class_to_indices[unlabeled_source][label_item].append(idx)
    
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        batch_data = {}
        
        for domain1, domain2 in self.source_pairs:
            common_classes = self.pair_common_classes[(domain1, domain2)]
            
            if len(common_classes) < self.num_classes:
                selected_classes = common_classes
            else:
                selected_classes = random.sample(common_classes, self.num_classes)
            
            if domain1 in self.labeled_sources:
                images1, labels1 = self._sample_from_labeled_source(domain1, selected_classes)
            else:
                images1, labels1 = self._sample_from_filtered_samples(domain1, selected_classes)

            if domain2 in self.labeled_sources:
                images2, labels2 = self._sample_from_labeled_source(domain2, selected_classes)
            else:
                images2, labels2 = self._sample_from_filtered_samples(domain2, selected_classes)

            pair_key = f"{domain1}-{domain2}"
            pair_entry = {
                'domain1': domain1,
                'domain2': domain2,
                'selected_classes': selected_classes,
                f'{domain1}_images': images1,
                f'{domain1}_labels': labels1,
                f'{domain2}_images': images2,
                f'{domain2}_labels': labels2,
            }
            batch_data[pair_key] = pair_entry
        
        return batch_data
    
    def _sample_from_labeled_source(self, source_name, selected_classes):
        images = []
        labels = []
        dataset = self.labeled_datasets[source_name]
        
        if source_name not in self._class_to_indices:
            self._build_class_to_indices(source_name)
        
        for cls in selected_classes:
            if cls not in self._class_to_indices[source_name]:
                raise ValueError(f"{source_name}: 类别 {cls} 不在索引映射中！")
            
            class_indices = self._class_to_indices[source_name][cls]
            if len(class_indices) < self.samples_per_class:
                raise ValueError(f"{source_name}: 类别 {cls} 样本不足！")
            
            sampled_indices = random.sample(class_indices, self.samples_per_class)
            
            for idx in sampled_indices:
                data = dataset[idx]
                
                if isinstance(data, dict):
                    img = data['Img']
                    label = data['Label']
                    images.append(img)
                    labels.append(label)
                else:
                    if len(data) == 2:
                        img, label = data
                    elif len(data) == 3:
                        img, label, _ = data
                    else:
                        raise ValueError(f"Unexpected dataset output")
                    
                    if self.transform:
                        img = self.transform(img)
                    images.append(img)
                    labels.append(label)
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        for cls in selected_classes:
            count = (labels_tensor == cls).sum().item()
            if count != self.samples_per_class:
                raise ValueError(f"{source_name}: 类别 {cls} 采样数量错误！")
        
        return torch.stack(images), labels_tensor
    
    def _sample_from_filtered_samples(self, unlabeled_source, selected_classes):
        images = []
        labels = []

        if unlabeled_source not in self.filtered_samples_dict:
            raise ValueError(f"{unlabeled_source}: 不在 filtered_samples_dict 中！")

        filtered_data = self.filtered_samples_dict[unlabeled_source]['data']

        if unlabeled_source not in self._filtered_class_to_indices:
            self._build_filtered_class_to_indices(unlabeled_source)

        for cls in selected_classes:
            if cls not in self._filtered_class_to_indices[unlabeled_source]:
                raise ValueError(f"{unlabeled_source}: 类别 {cls} 不在过滤后的索引映射中！")

            class_indices = self._filtered_class_to_indices[unlabeled_source][cls]
            if len(class_indices) < self.samples_per_class:
                raise ValueError(f"{unlabeled_source}: 类别 {cls} 样本不足！")

            sampled_indices = random.sample(class_indices, self.samples_per_class)

            for idx in sampled_indices:
                img_path = filtered_data[idx]
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
                labels.append(cls)

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        for cls in selected_classes:
            count = (labels_tensor == cls).sum().item()
            if count != self.samples_per_class:
                raise ValueError(f"{unlabeled_source}: 类别 {cls} 采样数量错误！")

        return torch.stack(images), labels_tensor


def collate_fn_pairwise(batch):
    return batch[0]
