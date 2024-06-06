import collections
import math
import types
import typing

import clip
import torch
import torch.nn as nn
from torchvision import models, transforms

# code lifted from CLIPasso

# For ViT
class CLIPVisualEncoder(nn.Module):
    def __init__(self, model_name, stride, device):
        super().__init__()
        self.load_model(model_name, device)
        self.old_stride = self.model.conv1.stride[0]
        self.new_stride = stride
        self.patch_vit_resolution(stride)

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.model.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    
    def load_model(self, model_name, device):
        model, preprocess = clip.load(model_name, device=device)
        self.model = model.visual
        self.mean = torch.tensor(preprocess.transforms[-1].mean, device=device)
        self.std = torch.tensor(preprocess.transforms[-1].std, device=device)

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: typing.Tuple[int, int]):
        def interpolate_pos_encoding(self, x, w, h):
            npatch = x.shape[1] - 1
            N = self.positional_embedding.shape[0] - 1
            if npatch == N and w == h:
                return self.positional_embedding
            class_pos_embed = self.positional_embedding[:1].type(x.dtype)
            patch_pos_embed = self.positional_embedding[1:].type(x.dtype)
            dim = x.shape[-1]
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch)
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = torch.nn.functional.interpolate(
                patch_pos_embed.reshape(int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(2, 0, 1).unsqueeze(0),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False,
            ).squeeze()
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(1, 2, 0).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        return interpolate_pos_encoding

    
    def patch_vit_resolution(self, stride):
        patch_size = self.model.conv1.stride[0]
        if stride == patch_size:
            return
        
        stride = (stride, stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in stride])
        self.model.conv1.stride = stride
        self.model.interpolate_pos_encoding = types.MethodType(CLIPVisualEncoder._fix_pos_enc(patch_size, stride), self.model)

    @property
    def dtype(self):
        return self.model.conv1.weight.dtype

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x, preprocess=False):
        self.featuremaps = collections.OrderedDict()
        if preprocess:
            x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        B, C, W, H = x.shape
        x = self.model.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.interpolate_pos_encoding(x, W, H)
        x = self.model.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        # remove cls
        featuremaps = [self.featuremaps[k].permute(0, 2, 1)[..., 1:] for k in range(12)]

        return featuremaps
