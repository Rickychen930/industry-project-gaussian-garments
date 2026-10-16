"""The StyleUNet of a Gaussian Garments checkpoint's appearance network
(`shadow_net`), ported to plain PyTorch.

Follows `scene/styleunet/styleunet.py` of eth-ait/Gaussian-Garments (a
StyleGAN2 port, MIT). Names match the reference so checkpoint keys load with
`strict=True`. The CUDA kernels (`upfirdn2d`, `fused_bias_act`) are plain
tensor ops here; the reference's CPU fallbacks are the test oracle. Research
and evaluation use only (see `appearance.py`).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    """Upsample by zero insertion, pad, FIR filter, downsample."""
    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])
    px0, px1, py0, py1 = pad
    batch, channel, height, width = x.shape
    x = x.reshape(batch * channel, 1, height, width)
    if up > 1:
        zeros = x.new_zeros(batch * channel, 1, height * up, width * up)
        zeros[:, :, ::up, ::up] = x
        x = zeros
    x = F.pad(x, [max(px0, 0), max(px1, 0), max(py0, 0), max(py1, 0)])
    x = x[
        :, :,
        max(-py0, 0): x.shape[2] - max(-py1, 0),
        max(-px0, 0): x.shape[3] - max(-px1, 0),
    ]
    weight = torch.flip(kernel, [0, 1])[None, None]
    x = F.conv2d(x, weight)
    if down > 1:
        x = x[:, :, ::down, ::down]
    return x.reshape(batch, channel, x.shape[2], x.shape[3])


def fused_leaky_relu(x, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest = [1] * (x.ndim - bias.ndim - 1)
        x = x + bias.view(1, bias.shape[0], *rest)
    return F.leaky_relu(x, negative_slope=negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        return fused_leaky_relu(x, self.bias, self.negative_slope, self.scale)


class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    return k / k.sum()


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)
        p = kernel.shape[0] - factor
        self.pad = ((p + 1) // 2 + factor - 1, p // 2)

    def forward(self, x):
        return upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)
        p = kernel.shape[0] - factor
        self.pad = ((p + 1) // 2, p // 2)

    def forward(self, x):
        return upfirdn2d(x, self.kernel, up=1, down=self.factor, pad=self.pad)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, x):
        return upfirdn2d(x, self.kernel, pad=self.pad)


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0.0, lr_mul=1.0, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        if self.activation:
            out = F.linear(x, self.weight * self.scale)
            return fused_leaky_relu(out, self.bias * self.lr_mul)
        return F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True,
                 upsample=False, downsample=False, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            self.blur = Blur(blur_kernel, pad=((p + 1) // 2 + factor - 1, p // 2 + 1), upsample_factor=factor)
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            self.blur = Blur(blur_kernel, pad=((p + 1) // 2, p // 2))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        if self.upsample:
            x = x.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=batch)
            out = out.view(batch, self.out_channel, out.shape[2], out.shape[3])
            return self.blur(out)
        if self.downsample:
            x = self.blur(x)
            x = x.view(1, batch * in_channel, x.shape[2], x.shape[3])
            out = F.conv2d(x, weight, padding=0, stride=2, groups=batch)
            return out.view(batch, self.out_channel, out.shape[2], out.shape[3])
        x = x.view(1, batch * in_channel, height, width)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        return out.view(batch, self.out_channel, out.shape[2], out.shape[3])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False,
                 blur_kernel=(1, 3, 3, 1), bias=True, activate=True):
        layers = []
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            layers.append(Blur(blur_kernel, pad=((p + 1) // 2, p // 2)))
            stride, padding = 2, 0
        else:
            stride, padding = 1, kernel_size // 2
        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=padding,
                                  stride=stride, bias=bias and not activate))
        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))
        super().__init__(*layers)


def _haar_wavelet():
    low = 1 / (2 ** 0.5) * torch.ones(1, 2)
    high = 1 / (2 ** 0.5) * torch.ones(1, 2)
    high[0, 0] = -high[0, 0]
    return low.T * low, high.T * low, low.T * high, high.T * high


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        ll, lh, hl, hh = _haar_wavelet()
        self.register_buffer("ll", ll)
        self.register_buffer("lh", lh)
        self.register_buffer("hl", hl)
        self.register_buffer("hh", hh)

    def forward(self, x):
        return torch.cat(
            [upfirdn2d(x, self.ll, down=2), upfirdn2d(x, self.lh, down=2),
             upfirdn2d(x, self.hl, down=2), upfirdn2d(x, self.hh, down=2)], 1)


class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        ll, lh, hl, hh = _haar_wavelet()
        self.register_buffer("ll", ll)
        self.register_buffer("lh", -lh)
        self.register_buffer("hl", -hl)
        self.register_buffer("hh", hh)

    def forward(self, x):
        ll, lh, hl, hh = x.chunk(4, 1)
        return (upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
                + upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
                + upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
                + upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0)))


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=(1, 3, 3, 1), downsample=True):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class FromRGB(nn.Module):
    """`use_wt=False` variant only - the one the checkpoint was built with."""

    def __init__(self, out_channel, in_channel, downsample=True, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        self.downsample = Downsample(blur_kernel) if downsample else None
        self.conv = ConvLayer(in_channel, out_channel, 1)

    def forward(self, x, skip=None):
        if self.downsample is not None:
            x = self.downsample(x)
        out = self.conv(x)
        if skip is not None:
            out = out + skip
        return x, out


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False,
                 blur_kernel=(1, 3, 3, 1), demodulate=True):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim,
                                    upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, x, style, noise=None):
        return self.activate(self.noise(self.conv(x, style), noise=noise))


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channel=12, upsample=True, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
            self.iwt = InverseHaarTransform(3)
            self.dwt = HaarTransform(3)
        self.out_channel = out_channel
        self.conv = ModulatedConv2d(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style) + self.bias
        if skip is not None:
            skip = self.dwt(self.upsample(self.iwt(skip)))
            out = out + skip
        return out


class SWGAN_unet(nn.Module):
    """The reference `SWGAN_unet`, single-style, inference paths only."""

    def __init__(self, inp_size, inp_ch, out_ch, out_size, style_dim, n_mlp, middle_size=8,
                 channel_multiplier=2, blur_kernel=(1, 3, 3, 1), lr_mlp=0.01):
        super().__init__()
        self.inp_size = inp_size
        self.style_dim = style_dim
        self.middle_log_size = int(math.log(middle_size, 2))

        layers = [PixelNorm(), EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu")]
        for _ in range(n_mlp - 1):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))
        self.style = nn.Sequential(*layers)

        self.channels = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * channel_multiplier,
                         128: 128 * channel_multiplier, 256: 64 * channel_multiplier,
                         512: 32 * channel_multiplier, 1024: 16 * channel_multiplier}
        self.log_size = int(math.log(out_size, 2)) - 1

        in_channel = self.channels[inp_size // 2]
        self.from_rgbs = nn.ModuleList()
        self.cond_convs = nn.ModuleList()
        self.comb_convs = nn.ModuleList()
        self.comb_convs.append(ConvLayer(in_channel * 2, in_channel, 3))
        self.conv_in = ConvLayer(inp_ch, in_channel, 3, downsample=True)
        for i in range(int(math.log(inp_size, 2)) - 2, self.middle_log_size - 1, -1):
            out_channel = self.channels[2 ** i]
            self.from_rgbs.append(FromRGB(in_channel, inp_ch, downsample=True))
            self.cond_convs.append(ConvBlock(in_channel, out_channel, blur_kernel))
            if i > self.middle_log_size:
                self.comb_convs.append(ConvLayer(out_channel * 2, out_channel, 3))
            else:
                self.comb_convs.append(ConvLayer(out_channel, out_channel, 3))
            in_channel = out_channel

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        in_channel = self.channels[middle_size]
        self.num_layers = (self.log_size - self.middle_log_size) * 2
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 8) // 2
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(1, 1, 2 ** res, 2 ** res))
        for i in range(self.middle_log_size + 1, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(in_channel=out_channel, style_dim=style_dim, out_channel=out_ch * 4))
            in_channel = out_channel
        self.iwt = InverseHaarTransform(out_ch)
        self.n_latent = self.log_size * 2 - (self.middle_log_size * 2 - 1) + 1

    def forward(self, styles, condition_img, randomize_noise=True, view_feature=None):
        styles = [self.style(s) for s in styles]
        if randomize_noise:
            noise = [None] * self.num_layers
        else:
            noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]
        assert len(styles) == 1, "single-style inference only"
        latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)

        cond_img = condition_img
        cond_out = self.conv_in(cond_img)
        cond_list = [cond_out]
        for from_rgb, cond_conv in zip(self.from_rgbs, self.cond_convs):
            cond_img, cond_out = from_rgb(cond_img, cond_out)
            cond_out = cond_conv(cond_out)
            cond_list.append(cond_out)

        i = 0
        skip = None
        out = None
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[::2], noise[1::2], self.to_rgbs):
            if i == 0:
                out = self.comb_convs[-1](cond_list[-1])
            elif i < 2 * len(self.comb_convs):
                out = torch.cat([out, cond_list[-1 - (i // 2)]], dim=1)
                out = self.comb_convs[-1 - (i // 2)](out)
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            if view_feature is not None and i == 8:
                # As in the reference; `out` is never read again, so this has no effect on `skip`.
                view_feature = F.interpolate(view_feature, out.shape[-2:], mode="bilinear")
                out = out + view_feature
            i += 2
        return self.iwt(skip), None
