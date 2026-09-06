import torch
from torch.nn import init

def weights_init_he(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if 'weight' in m.state_dict().keys():
            m.weight.data.normal_(1.0, 0.02)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0)
    else:
        if 'weight' in m.state_dict().keys():
            init.kaiming_normal_(m.weight)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0)


def init_weights(model, state_dict):
    model.apply(weights_init_he)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
        print("✅ 已加载预训练权重（包括BN层参数）")

    return model
