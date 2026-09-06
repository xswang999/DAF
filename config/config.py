import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset options
__C.DATASET = edict()
__C.DATASET.NUM_CLASSES = 65
__C.DATASET.DATAROOT = r'D:\PyTorch\Pycharm\SLDG\HCC\experiments\dataset\Office-Home'
__C.DATASET.SOURCE_NAMES = ['Art', 'Product', 'RealWorld']
__C.DATASET.UNLABELED_SOURCES = ['Product', 'RealWorld']
__C.DATASET.TARGET_NAME = 'Clipart'
__C.DATASET.RESULTS_DIR = r'D:\PyTorch\Pycharm\SLDG\HCC\results\Office-Home'

# Model options
__C.MODEL = edict()
__C.MODEL.FEATURE_EXTRACTOR = 'resnet18'
__C.MODEL.FC_HIDDEN_DIMS = ()
__C.MODEL.PRETRAINED = True
__C.MODEL.PRETRAINED_WEIGHTS = 'ResNet50_Weights.IMAGENET1K_V2'
# data pre-processing options

__C.DATA_TRANSFORM = edict()
__C.DATA_TRANSFORM.FINESIZE = 224
__C.DATA_TRANSFORM.NORMALIZE_MEAN = (0.485, 0.456, 0.406)
__C.DATA_TRANSFORM.NORMALIZE_STD = (0.229, 0.224, 0.225)

# Training options
__C.TRAIN = edict()
__C.TRAIN.SOURCE_CLASS_BATCH_SIZE = 3
__C.TRAIN.NUM_SELECTED_CLASSES = 6
__C.TRAIN.DROPOUT_RATIO = (0.0,)
__C.TRAIN.BASE_LR = 0.001
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.LR_MULT = 10
__C.TRAIN.OPTIMIZER = 'SGD'
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.LR_SCHEDULE = 'inv'
__C.TRAIN.MAX_ITERATIONS = 10000
__C.TRAIN.MIN_SN_PER_CLASS = 3
__C.TRAIN.RECORD_INTERVAL = 250

__C.INV = edict()
__C.INV.ALPHA = 10
__C.INV.BETA = 0.75

# CC options (Contrastive Constraint)
__C.CC = edict()
__C.CC.KERNEL_NUM = (5, 5)
__C.CC.KERNEL_MUL = (2, 2)
__C.CC.C_BHCC_LOSS_WEIGHT = 0.05
__C.CC.D_BHCC_LOSS_WEIGHT = 0.1
__C.CC.UU_GRAD_AGG_WEIGHT = 0.7
__C.CC.ALIGNMENT_FEAT_KEYS = ['feat', 'probs']
__C.CC.INTRA_ONLY = False

__C.PSEUDO_DOMAIN = edict()
__C.PSEUDO_DOMAIN.NUM_DOMAINS = 3
__C.PSEUDO_DOMAIN.CLASSES_PER_DOMAIN = 2

# Testing options
__C.TEST = edict()
__C.TEST.BATCH_SIZE = 30
__C.TEST.DATASET_TYPE = 'SingleDataset'
__C.TEST.DOMAIN = ''

# MISC
__C.WEIGHTS = ''
__C.RESUME = ''
__C.EVAL_METRIC = "accuracy"
__C.SEED = 42
__C.EXP_NAME = 'exp'
__C.SAVE_DIR = r'D:\PyTorch\Pycharm\SLDG\HCC\experiments\ckpt\Office-Home'
__C.NUM_WORKERS = 4
__C.DATALOADER_PREFETCH_FACTOR = 6
__C.DATALOADER_PERSISTENT_WORKERS = True
__C.DATALOADER_PIN_MEMORY = True

def _merge_a_into_b(a, b):
    if type(a) is not edict:
        return

    for k in a:
        # a must specify keys that are in b
        v = a[k]
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif k in {'PER_DOMAIN_FILTERING'} and type(v) is edict:
                pass
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        _FREE_DICT_KEYS = {'PER_DOMAIN_FILTERING'}
        if type(v) is edict:
            if k in _FREE_DICT_KEYS:
                def _edict_to_dict(obj):
                    if isinstance(obj, dict):
                        return {kk: _edict_to_dict(vv) for kk, vv in obj.items()}
                    elif isinstance(obj, list):
                        return [_edict_to_dict(i) for i in obj]
                    return obj
                b[k] = _edict_to_dict(a[k])
            else:
                try:
                    _merge_a_into_b(a[k], b[k])
                except:
                    print('Error under config key: {}'.format(k))
                    raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r', encoding='utf-8') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            value = v
        d[subkey] = value