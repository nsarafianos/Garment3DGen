"""
    Various helper functions

    create_scene -> combines multiple nvdiffmodeling meshes in to a single mesh with mega texture
"""
import sys
import numpy as np
import torch

from math import ceil

sys.path.append("../nvdiffmodeling")

import nvdiffmodeling.src.mesh as mesh
import nvdiffmodeling.src.texture as texture
import nvdiffmodeling.src.renderutils as ru

from torchvision.models import vgg19_bn
from torchvision.models import VGG19_BN_Weights
import torch.nn as nn
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out





cosine_sim = torch.nn.CosineSimilarity()
render_loss = torch.nn.L1Loss()
mean = torch.tensor([0.485, 0.456, 0.406], device='cuda')
std = torch.tensor([0.229, 0.224, 0.225], device='cuda')

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            # cos_loss += self.weights[i] * cosine_sim(x_vgg[i], y_vgg[i].detach()).mean()
        return loss

vgg_loss = VGGLoss()
def get_vgg_loss(base_image, target_image):
    base_image = (torch.clamp(base_image, 0, 1) - mean[None, :, None, None]) / std[None, :, None, None]
    target_image = (torch.clamp(target_image, 0, 1) - mean[None, :, None, None]) / std[None, :, None, None]
    return vgg_loss(base_image, target_image)

def cosine_sum(features, targets):
    return -cosine_sim(features, targets).sum()

def cosine_avg(features, targets):
    return -cosine_sim(features, targets).mean()

def l1_avg(source, targets):
    return render_loss(source, targets).mean()
    
def _merge_attr_idx(a, b, a_idx, b_idx, scale_a=1.0, scale_b=1.0, add_a=0.0, add_b=0.0):
    if a is None and b is None:
        return None, None
    elif a is not None and b is None:
        return (a*scale_a)+add_a, a_idx
    elif a is None and b is not None:
        return (b*scale_b)+add_b, b_idx
    else:
        return torch.cat(((a*scale_a)+add_a, (b*scale_b)+add_b), dim=0), torch.cat((a_idx, b_idx + a.shape[0]), dim=0)

def create_scene(meshes, sz=1024):
    
    # Need to comment and fix code
    
    scene = mesh.Mesh()

    tot = len(meshes) if len(meshes) % 2 == 0 else len(meshes)+1

    nx = 2
    ny = ceil(tot / 2) if ceil(tot / 2) % 2 == 0 else ceil(tot / 2) + 1

    w = int(sz*ny)
    h = int(sz*nx)

    dev = meshes[0].v_pos.device

    kd_atlas = torch.ones ( (1, w, h, 4) ).to(dev)
    ks_atlas = torch.zeros( (1, w, h, 3) ).to(dev)
    kn_atlas = torch.ones ( (1, w, h, 3) ).to(dev)

    for i, m in enumerate(meshes):
        v_pos, t_pos_idx = _merge_attr_idx(scene.v_pos, m.v_pos, scene.t_pos_idx, m.t_pos_idx)
        v_nrm, t_nrm_idx = _merge_attr_idx(scene.v_nrm, m.v_nrm, scene.t_nrm_idx, m.t_nrm_idx)
        v_tng, t_tng_idx = _merge_attr_idx(scene.v_tng, m.v_tng, scene.t_tng_idx, m.t_tng_idx)

        pos_x = i % nx
        pos_y = int(i / ny)

        sc_x = 1./nx
        sc_y = 1./ny

        v_tex, t_tex_idx = _merge_attr_idx(
            scene.v_tex,
            m.v_tex,
            scene.t_tex_idx,
            m.t_tex_idx,
            scale_a=1.,
            scale_b=torch.tensor([sc_x, sc_y]).to(dev),
            add_a=0.,
            add_b=torch.tensor([sc_x*pos_x, sc_y*pos_y]).to(dev)
        )

        kd_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['kd'].data.shape[-1]] = m.material['kd'].data
        ks_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['ks'].data.shape[-1]] = m.material['ks'].data
        kn_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['normal'].data.shape[-1]] = m.material['normal'].data

        scene = mesh.Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            v_nrm=v_nrm,
            t_nrm_idx=t_nrm_idx,
            v_tng=v_tng,
            t_tng_idx=t_tng_idx,
            v_tex=v_tex,
            t_tex_idx=t_tex_idx,
            base=scene 
        )

    scene = mesh.Mesh(
        material={
            'bsdf': 'diffuse',
            'kd': texture.Texture2D(
                kd_atlas
            ),
            'ks': texture.Texture2D(
                ks_atlas
            ),
            'normal': texture.Texture2D(
                kn_atlas
            ),
        },
        base=scene # gets uvs etc from here
    )

    return scene

def get_vp_map(v_pos, mtx_in, resolution):
    device = v_pos.device
    with torch.no_grad():
        vp_mtx = torch.tensor([
            [resolution / 2, 0., 0., (resolution - 1) / 2],
            [0., resolution / 2, 0., (resolution - 1) / 2],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.,]
        ], device=device)

        v_pos_clip = ru.xfm_points(v_pos[None, ...], mtx_in)
        v_pos_div = v_pos_clip / v_pos_clip[..., -1:]

        v_vp = (vp_mtx @ v_pos_div.transpose(1, 2)).transpose(1, 2)[..., :-1]

        # don't need manual z-buffer here since we're using the rast map to do occlusion
        if False:
            v_pix = v_vp[..., :-1].int().cpu().numpy()
            v_depth = v_vp[..., -1].cpu().numpy()

            # pix_v_map = -torch.ones(len(v_pix), resolution, resolution, dtype=int)
            pix_v_map = -np.ones((len(v_pix), resolution, resolution), dtype=int)
            # v_pix_map = resolution * torch.ones(len(v_pix), len(v_pos), 2, dtype=int)
            v_pix_map = resolution * np.ones_like(v_pix, dtype=int)
            # buffer = torch.ones_like(pix_v_map) / 0
            buffer = -np.ones_like(pix_v_map) / 0
            for i, vs in enumerate(v_pix):
                for j, (y, x) in enumerate(vs):
                    if x < 0 or x > resolution - 1 or y < 0 or y > resolution - 1:
                        continue
                    else:
                        if v_depth[i, j] > buffer[i, x, y]:
                            buffer[i, x, y] = v_depth[i, j]
                            if pix_v_map[i, x, y] != -1:
                                v_pix_map[i, pix_v_map[i, x, y]] = np.array([resolution, resolution])
                            pix_v_map[i, x, y] = j
                            v_pix_map[i, j] = np.array([x, y])
            v_pix_map = torch.tensor(v_pix_map, device=device)
        v_pix_map = v_vp[..., :-1].int().flip([-1])
        v_pix_map [(v_pix_map > resolution - 1) | (v_pix_map < 0)] = resolution
    return v_pix_map.long()




    