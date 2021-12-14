import torch
import numpy as np
import random
import os


def fix_seed(seed=12345, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    print(f"Fix seed: {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    import torch.nn as nn

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            return self.dropout(x)

        def infer(self, x):
            return self.dropout(x)

    fix_seed(12)
    inputs = torch.rand(2, 3)
    net = Net()
    with torch.no_grad():
        net.eval()
        print(net(inputs))

    net.train()
    print(net(inputs))

