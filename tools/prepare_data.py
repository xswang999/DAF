import os
import data.utils as data_utils
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from data.class_aware_dataset_dataloader import ClassAwareDataLoader, SourcePairClassAwareDataLoader
from data.pseudo_label_loader import load_unlabeled_source_pseudo_labels
from config.config import cfg

def prepare_data_DAF():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source_names = cfg.DATASET.SOURCE_NAMES  # ['Art', 'Clipart', 'RealWorld']
    target = cfg.DATASET.TARGET_NAME  # 'Product'
    
    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    unlabeled_sources = cfg.DATASET.UNLABELED_SOURCES
    labeled_sources = [name for name in source_names if name not in unlabeled_sources]
    
    for source in labeled_sources:
        dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
        dataloaders[source] = CustomDatasetDataLoader(
                    dataset_root=dataroot_S,
                    dataset_type='SingleDataset',
                    batch_size=18,
                    transform=train_transform,
                    train=True,
                    num_workers=cfg.NUM_WORKERS,
                    prefetch_factor=cfg.DATALOADER_PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
                    persistent_workers=cfg.DATALOADER_PERSISTENT_WORKERS if cfg.NUM_WORKERS > 0 else False,
                    pin_memory=cfg.DATALOADER_PIN_MEMORY,
                    classnames=classes)


    if len(labeled_sources) >= 2:
        source_batch_size = cfg.TRAIN.SOURCE_CLASS_BATCH_SIZE
        
        for i in range(len(labeled_sources)):
            for j in range(i+1, len(labeled_sources)):
                source1 = labeled_sources[i]
                source2 = labeled_sources[j]
                pair_name = f'pair_{source1}_{source2}'
                
                dataroot_S1 = os.path.join(cfg.DATASET.DATAROOT, source1)
                dataroot_S2 = os.path.join(cfg.DATASET.DATAROOT, source2)
                
                dataloaders[pair_name] = SourcePairClassAwareDataLoader(
                            source1_batch_size=source_batch_size,
                            source2_batch_size=source_batch_size,
                            source1_dataset_root=dataroot_S1,
                            source2_dataset_root=dataroot_S2,
                            transform=train_transform,
                            classnames=classes,
                            class_set=classes,
                            num_selected_classes=cfg.TRAIN.NUM_SELECTED_CLASSES,
                            num_workers=cfg.NUM_WORKERS,
                            drop_last=True, sampler='RandomSampler')
                dataloaders[pair_name].construct()

    for unlabeled_source in unlabeled_sources:
        dataloaders[f'pseudo_label_{unlabeled_source}'] = None

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    prefetch_factor=cfg.DATALOADER_PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
                    persistent_workers=cfg.DATALOADER_PERSISTENT_WORKERS if cfg.NUM_WORKERS > 0 else False,
                    pin_memory=cfg.DATALOADER_PIN_MEMORY,
                    classnames=classes)

    return dataloaders
