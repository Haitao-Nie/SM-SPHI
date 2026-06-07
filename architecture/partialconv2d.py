import torch
import torch.nn.functional as F
from torch import nn


class PartialConv2d(nn.Conv2d):
    """
    Local partial convolution layer adapted from NVIDIA's implementation.
    It masks invalid pixels, renormalizes by the valid-count ratio, and
    optionally returns the updated validity mask.
    """

    def __init__(self, *args, multi_channel=False, return_mask=False, **kwargs):
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super().__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_mask_updater = torch.ones(
                self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
            )
        else:
            self.weight_mask_updater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = (
            self.weight_mask_updater.shape[1]
            * self.weight_mask_updater.shape[2]
            * self.weight_mask_updater.shape[3]
        )
        self.last_size = None
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4

        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            with torch.no_grad():
                if self.weight_mask_updater.device != input.device or self.weight_mask_updater.dtype != input.dtype:
                    self.weight_mask_updater = self.weight_mask_updater.to(input)

                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones_like(input)
                    else:
                        mask = torch.ones((input.shape[0], 1, input.shape[2], input.shape[3]), device=input.device, dtype=input.dtype)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_mask_updater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = self.mask_ratio * self.update_mask

        masked_input = input * mask if mask_in is not None else input
        raw_out = super().forward(masked_input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = (raw_out - bias_view) * self.mask_ratio + bias_view
            output = output * self.update_mask
        else:
            output = raw_out * self.mask_ratio

        if self.return_mask:
            return output, self.update_mask
        return output
