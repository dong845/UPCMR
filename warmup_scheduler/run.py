import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from torch.optim.sgd import SGD

from scheduler import GradualWarmupScheduler


if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = SGD(model, 0.0001)

    # scheduler_warmup is chained with schduler_steplr
    # scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1)
    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                                                            T_max=15,
                                                            eta_min=1e-6)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()

    for epoch in range(1, 25):
        scheduler_warmup.step(epoch)
        print(epoch, optim.param_groups[0]['lr'])

        optim.step()    # backward pass (update network)