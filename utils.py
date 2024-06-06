import pymeshlab
import torch
from nvdiffmodeling.src     import obj
from nvdiffmodeling.src     import mesh
from nvdiffmodeling.src     import texture
import numpy as np
from utilities.helpers import get_vp_map

texture_map = texture.create_trainable(np.random.uniform(size=[512] * 2 + [3], low=0.0, high=1.0), [512] * 2, True)
normal_map = texture.create_trainable(np.array([0, 0, 1]), [512] * 2, True)
specular_map = texture.create_trainable(np.array([0, 0, 0]), [512] * 2, True)

def get_mesh(mesh_path, output_path, triangulate_flag, bsdf_flag, mesh_name='mesh.obj'):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    if triangulate_flag:
        print('Retriangulating shape')
        ms.meshing_isotropic_explicit_remeshing()

    if not ms.current_mesh().has_wedge_tex_coord():
        # some arbitrarily high number
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)

    ms.save_current_mesh(str(output_path / 'tmp' / mesh_name))

    load_mesh = obj.load_obj(str(output_path / 'tmp' / mesh_name))
    load_mesh = mesh.unit_size(load_mesh)

    ms.add_mesh(
        pymeshlab.Mesh(vertex_matrix=load_mesh.v_pos.cpu().numpy(), face_matrix=load_mesh.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(str(output_path / 'tmp' / mesh_name), save_vertex_color=False)

    load_mesh = mesh.Mesh(
        material={
            'bsdf': bsdf_flag,
            'kd': texture_map,
            'ks': specular_map,
            'normal': normal_map,
        },
        base=load_mesh  # Get UVs from original loaded mesh
    )
    return load_mesh


def get_og_mesh(mesh_path, output_path, triangulate_flag, bsdf_flag, mesh_name='mesh.obj'):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    if triangulate_flag:
        print('Retriangulating shape')
        ms.meshing_isotropic_explicit_remeshing()

    if not ms.current_mesh().has_wedge_tex_coord():
        # some arbitrarily high number
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)

    ms.save_current_mesh(str(output_path / 'tmp' / mesh_name))

    load_mesh = obj.load_obj(str(output_path / 'tmp' / mesh_name))
    load_mesh = mesh.resize_mesh(load_mesh)

    ms.add_mesh(
        pymeshlab.Mesh(vertex_matrix=load_mesh.v_pos.cpu().numpy(), face_matrix=load_mesh.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(str(output_path / 'tmp' / mesh_name), save_vertex_color=False)

    load_mesh = mesh.Mesh(
        material={
            'bsdf': bsdf_flag,
            'kd': texture_map,
            'ks': specular_map,
            'normal': normal_map,
        },
        base=load_mesh  # Get UVs from original loaded mesh
    )
    return load_mesh

def compute_mv_cl(final_mesh, fe, normalized_clip_render, params_camera, train_rast_map, cfg, device):
    # Consistency loss
    # Get mapping from vertex to pixels
    curr_vp_map = get_vp_map(final_mesh.v_pos, params_camera['mvp'], 224)
    for idx, rast_faces in enumerate(train_rast_map[:, :, :, 3].view(cfg.batch_size, -1)):
        u_faces = rast_faces.unique().long()[1:] - 1
        t = torch.arange(len(final_mesh.v_pos), device=device)
        u_ret = torch.cat([t, final_mesh.t_pos_idx[u_faces].flatten()]).unique(return_counts=True)
        non_verts = u_ret[0][u_ret[1] < 2]
        curr_vp_map[idx][non_verts] = torch.tensor([224, 224], device=device)

    # Get mapping from vertex to patch
    med = (fe.old_stride - 1) / 2
    curr_vp_map[curr_vp_map < med] = med
    curr_vp_map[(curr_vp_map > 224 - fe.old_stride) & (curr_vp_map < 224)] = 223 - med
    curr_patch_map = ((curr_vp_map - med) / fe.new_stride).round()
    flat_patch_map = curr_patch_map[..., 0] * (((224 - fe.old_stride) / fe.new_stride) + 1) + curr_patch_map[..., 1]

    # Deep features
    patch_feats = fe(normalized_clip_render)
    flat_patch_map[flat_patch_map > patch_feats[0].shape[-1] - 1] = patch_feats[0].shape[-1]
    flat_patch_map = flat_patch_map.long()[:, None, :].repeat(1, patch_feats[0].shape[1], 1)

    deep_feats = patch_feats[cfg.consistency_vit_layer]
    deep_feats = torch.nn.functional.pad(deep_feats, (0, 1))
    deep_feats = torch.gather(deep_feats, dim=2, index=flat_patch_map)
    deep_feats = torch.nn.functional.normalize(deep_feats, dim=1, eps=1e-6)

    elev_d = torch.cdist(params_camera['elev'].unsqueeze(1), params_camera['elev'].unsqueeze(1)).abs() < torch.deg2rad(
        torch.tensor(cfg.consistency_elev_filter))
    azim_d = torch.cdist(params_camera['azim'].unsqueeze(1), params_camera['azim'].unsqueeze(1)).abs() < torch.deg2rad(
        torch.tensor(cfg.consistency_azim_filter))

    cosines = torch.einsum('ijk, lkj -> ilk', deep_feats, deep_feats.permute(0, 2, 1))
    cosines = (cosines * azim_d.unsqueeze(-1) * elev_d.unsqueeze(-1)).permute(2, 0, 1).triu(1)
    consistency_loss = cosines[cosines != 0].mean()
    return consistency_loss