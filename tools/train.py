import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import random
import argparse
import numpy as np
from torch.backends import cudnn
from model import model
from config.config import cfg, cfg_from_file, cfg_from_list
from prepare_data import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark    = False
    print(f"🔒 随机种子已设置: {seed}")

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=r'D:\PyTorch\Pycharm\SLDG\DAF\experiments\config\Office-Home\Art_to_Clipart.yaml', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--method', dest='method',
                        help='set the method to use', 
                        default='DAF', type=str)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name', 
                        default='', type=str)

    args = parser.parse_args()
    return args

def train(args):
    # method-specific setting 
    if args.method == 'DAF':
        from solver.daf_solver import DAFSolver as Solver
        dataloaders = prepare_data_DAF()

    else:
        raise NotImplementedError("Currently don't support the specified method: %s."
                                 % args.method)

    # initialize model
    model_state_dict = None
    fx_pretrained = cfg.MODEL.PRETRAINED if hasattr(cfg.MODEL, 'PRETRAINED') else True
    resume_dict = None
    loaded_model = None

    if cfg.RESUME != '':
        resume_dict = torch.load(cfg.RESUME)
        model_state_dict = resume_dict['model_state_dict']
        fx_pretrained = False
    elif cfg.WEIGHTS != '':
        param_dict = torch.load(cfg.WEIGHTS, weights_only=False)
        if isinstance(param_dict, dict):
            if 'weights' in param_dict:
                model_state_dict = param_dict['weights']
            elif 'model_state_dict' in param_dict:
                model_state_dict = param_dict['model_state_dict']
            elif 'state_dict' in param_dict:
                model_state_dict = param_dict['state_dict']
            else:
                model_state_dict = param_dict
            print(f"✅ 从 {cfg.WEIGHTS} 加载模型权重（{len(model_state_dict)} 个参数）")
            fx_pretrained = False
        else:
            if hasattr(param_dict, 'module'):
                loaded_model = param_dict.module
                print(f"✅ 从 {cfg.WEIGHTS} 直接加载完整模型")
            elif hasattr(param_dict, 'state_dict'):
                model_state_dict = param_dict.state_dict()
                print(f"✅ 从 {cfg.WEIGHTS} 加载模型权重（{len(model_state_dict)} 个参数）")
                fx_pretrained = False
            else:
                model_state_dict = param_dict
                print(f"✅ 从 {cfg.WEIGHTS} 加载模型权重")
                fx_pretrained = False

    if loaded_model is not None:
        net = loaded_model
    else:
        net = model.daf_net(num_classes=cfg.DATASET.NUM_CLASSES, 
                     state_dict=model_state_dict,
                     feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR, 
                     fx_pretrained=fx_pretrained, 
                     dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
                     fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
                     weights=cfg.MODEL.PRETRAINED_WEIGHTS if hasattr(cfg.MODEL, 'PRETRAINED_WEIGHTS') else None)

    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
       net.cuda()

    # initialize solver
    train_solver = Solver(net, dataloaders, resume=resume_dict)
    # train 
    train_solver.solve()
    print('Finished!')

if __name__ == '__main__':
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.resume is not None:
        cfg.RESUME = args.resume
    if args.weights is not None:
        cfg.WEIGHTS = args.weights
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name

    set_seed(cfg.SEED)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)

    train(args)
