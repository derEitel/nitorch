# Initialize weights
from torch.nn import init, Conv3d, BatchNorm3d, Linear
def xavier(x):
    return init.xavier_normal_(x)

def he(x):
    return init.kaiming_normal_(x)

def weights_init(m, func=he):
    if isinstance(m, Conv3d):
        func(m.weight.data)
    elif isinstance(m, BatchNorm3d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, Linear):
        init.constant_(m.bias, 0)