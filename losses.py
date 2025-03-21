import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIM(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, Xt: torch.Tensor, Yt: torch.Tensor, data_range=None, full=False):
        assert isinstance(self.w, torch.Tensor)
        Xt = (Xt / Xt.max()).unsqueeze(2)
        Yt = (Yt / Yt.max()).unsqueeze(2)
        ssims = 0.0
        for t in range(Xt.shape[1]):

            X = Xt[:, t, :, :, :].permute(0, 1, 3, 2)
            Y = Yt[:, t, :, :, :].permute(0, 1, 3, 2)

            if data_range is None:
                data_range = torch.ones_like(Y)  # * Y.max()
                p = (self.win_size - 1) // 2
                data_range = data_range[:, :, p:-p, p:-p]
            data_range = data_range[:, None, None, None]
            C1 = (self.k1 * data_range) ** 2
            C2 = (self.k2 * data_range) ** 2
            ux = F.conv2d(X, self.w)  # typing: ignore
            uy = F.conv2d(Y, self.w)  #
            uxx = F.conv2d(X * X, self.w)
            uyy = F.conv2d(Y * Y, self.w)
            uxy = F.conv2d(X * Y, self.w)
            vx = self.cov_norm * (uxx - ux * ux)
            vy = self.cov_norm * (uyy - uy * uy)
            vxy = self.cov_norm * (uxy - ux * uy)
            A1, A2, B1, B2 = (
                2 * ux * uy + C1,
                2 * vxy + C2,
                ux ** 2 + uy ** 2 + C1,
                vx + vy + C2,
            )
            D = B1 * B2
            S = (A1 * A2) / D

            if full:
                ssims += 1 - S
            else:
                ssims += 1 - S.mean()

        return ssims / Xt.shape[1]


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, Xt: torch.Tensor, Yt: torch.Tensor, data_range=None, full=False):
        assert isinstance(self.w, torch.Tensor)

        ssims = 0.0
        
        Xt = (Xt / (Xt.max()+1e-6)).float().unsqueeze(1)+1e-6
        Yt = (Yt / (Yt.max()+1e-6)).float().unsqueeze(1)+1e-6

        for t in range(Xt.shape[2]):

            X = Xt[:, :, t, :, :]
            Y = Yt[:, :, t, :, :]

            if data_range is None:
                data_range = torch.ones_like(Y)  # * Y.max()
                p = (self.win_size - 1) // 2
                data_range = data_range[:, :, p:-p, p:-p]
            data_range = data_range[:, None, None, None]
            C1 = (self.k1 * data_range) ** 2
            C2 = (self.k2 * data_range) ** 2
            ux = F.conv2d(X, self.w)  # typing: ignore
            uy = F.conv2d(Y, self.w)  #
            uxx = F.conv2d(X * X, self.w)
            uyy = F.conv2d(Y * Y, self.w)
            uxy = F.conv2d(X * Y, self.w)
            vx = self.cov_norm * (uxx - ux * ux)
            vy = self.cov_norm * (uyy - uy * uy)
            vxy = self.cov_norm * (uxy - ux * uy)
            A1, A2, B1, B2 = (
                2 * ux * uy + C1,
                2 * vxy + C2,
                ux ** 2 + uy ** 2 + C1,
                vx + vy + C2,
            )
            D = B1 * B2
            S = (A1 * A2) / D

            if full:
                ssims += 1 - S
            else:
                ssims += 1 - S.mean()

        return ssims / Xt.shape[2]
    
