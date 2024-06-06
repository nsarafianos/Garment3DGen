import clip
import kornia
import os
import sys
import pathlib
import torchvision
import logging
import yaml
import nvdiffrast.torch as dr
from easydict import EasyDict

from NeuralJacobianFields import SourceMesh

from nvdiffmodeling.src import render

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utilities.video import Video
from utilities.helpers import cosine_avg, create_scene, l1_avg
from utilities.camera import CameraBatch, get_camera_params
from utilities.clip_spatial import CLIPVisualEncoder
from utilities.resize_right import resize, cubic, linear, lanczos2, lanczos3
from packages.fashion_clip.fashion_clip.fashion_clip import FashionCLIP
from utils import *
from get_embeddings import *

from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
# from collisions import collision_penalty
from pytorch3d.ops import sample_points_from_meshes
# CUDA_LAUNCH_BLOCKING=1

# def triangle_area(vertices):
#     # Calculate the area of a triangle given its vertices
#     edge1 = vertices[1] - vertices[0]
#     edge2 = vertices[2] - vertices[0]
#     cross_product = torch.cross(edge1, edge2)
#     area = 0.5 * torch.norm(cross_product)
#     return area

def total_triangle_area(vertices):
    # Calculate the sum of the areas of all triangles in the mesh
    num_triangles = vertices.shape[0] // 3
    triangle_vertices = vertices.view(num_triangles, 3, 3)

    # Calculate the cross product for each triangle
    cross_products = torch.cross(triangle_vertices[:, 1] - triangle_vertices[:, 0],
                                 triangle_vertices[:, 2] - triangle_vertices[:, 0])

    # Calculate the area of each triangle
    areas = 0.5 * torch.norm(cross_products, dim=1)

    # Sum the areas of all triangles
    total_area = torch.sum(areas)
    return total_area

def triangle_size_regularization(vertices):
    # Penalize small triangles by minimizing the squared sum of triangle areas
    return total_triangle_area(vertices)**2

def loop(cfg):
    clip_flag = True
    output_path = pathlib.Path(cfg['output_path'])
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / 'config.yml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    cfg = EasyDict(cfg)
    
    print(f'Output directory {cfg.output_path} created')
    os.makedirs(output_path / 'tmp', exist_ok=True)

    device = torch.device(f'cuda:{cfg.gpu}')
    torch.cuda.set_device(device)

    text_input, image_input, fashion_image, fashion_text, use_target_mesh = False, False, False, False, True
    CLIP_embeddings = False

    if CLIP_embeddings:
        print('Loading CLIP Models')
        model, preprocess = clip.load(cfg.clip_model, device=device)
    else:
        fclip = FashionCLIP('fashion-clip')

    fe = CLIPVisualEncoder(cfg.consistency_clip_model, cfg.consistency_vit_stride, device)

    if fashion_text or fashion_image:
        target_direction_embeds, delta_direction_embeds = get_fashion_img_embeddings(fclip, cfg, device, True)
    elif text_input:
        target_direction_embeds, delta_direction_embeds = get_text_embeddings(clip, model, cfg, device)
    elif image_input:
        target_direction_embeds, delta_direction_embeds = get_img_embeddings(model, preprocess, cfg, device)

    clip_mean = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    # output video
    video = Video(cfg.output_path)
    # GL Context
    glctx = dr.RasterizeGLContext()


    load_mesh = get_mesh(cfg.mesh, output_path, cfg.retriangulate, cfg.bsdf)

    if use_target_mesh:
        target_mesh = get_mesh(cfg.target_mesh, output_path, cfg.retriangulate, cfg.bsdf, 'mesh_target.obj')
        # We construct a Meshes structure for the target mesh
        trg_mesh_p3d = Meshes(verts=[target_mesh.v_pos], faces=[target_mesh.t_pos_idx])


    jacobian_source = SourceMesh.SourceMesh(0, str(output_path / 'tmp' / 'mesh.obj'), {}, 1, ttype=torch.float)
    if len(list((output_path / 'tmp').glob('*.npz'))) > 0:
        logging.warn(f'Using existing Jacobian .npz files in {str(output_path)}/tmp/ ! Please check if this is intentional.')
    jacobian_source.load()
    jacobian_source.to(device)

    with torch.no_grad():
        gt_jacobians = jacobian_source.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0))
    gt_jacobians.requires_grad_(True)

    optimizer = torch.optim.Adam([gt_jacobians], lr=cfg.lr)
    cams_data = CameraBatch(
        cfg.train_res,
        [cfg.dist_min, cfg.dist_max],
        [cfg.azim_min, cfg.azim_max],
        [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
        [cfg.fov_min, cfg.fov_max],
        cfg.aug_loc,
        cfg.aug_light,
        cfg.aug_bkg,
        cfg.batch_size,
        rand_solid=True
    )
    cams = torch.utils.data.DataLoader(cams_data, cfg.batch_size, num_workers=0, pin_memory=True)
    best_losses = {'CLIP': np.inf, 'total': np.inf}

    for out_type in ['final', 'best_clip', 'best_total', 'target_final']:
        os.makedirs(output_path / f'mesh_{out_type}', exist_ok=True)
    os.makedirs(output_path / 'images', exist_ok=True)
    logger = SummaryWriter(str(output_path / 'logs'))

    rot_ang = 0.0
    t_loop = tqdm(range(cfg.epochs), leave=False)

    if cfg.resize_method == 'cubic':
        resize_method = cubic
    elif cfg.resize_method == 'linear':
        resize_method = linear
    elif cfg.resize_method == 'lanczos2':
        resize_method = lanczos2
    elif cfg.resize_method == 'lanczos3':
        resize_method = lanczos3

    for it in t_loop:

        # updated vertices from jacobians
        n_vert = jacobian_source.vertices_from_jacobians(gt_jacobians).squeeze()

        # TODO: More texture code required to make it work ...
        ready_texture = texture.Texture2D(
            kornia.filters.gaussian_blur2d(
                load_mesh.material['kd'].data.permute(0, 3, 1, 2),
                kernel_size=(7, 7),
                sigma=(3, 3),
            ).permute(0, 2, 3, 1).contiguous()
        )

        kd_notex = texture.Texture2D(torch.full_like(ready_texture.data, 0.5))

        ready_specular = texture.Texture2D(
            kornia.filters.gaussian_blur2d(
                load_mesh.material['ks'].data.permute(0, 3, 1, 2),
                kernel_size=(7, 7),
                sigma=(3, 3),
            ).permute(0, 2, 3, 1).contiguous()
        )

        ready_normal = texture.Texture2D(
            kornia.filters.gaussian_blur2d(
                load_mesh.material['normal'].data.permute(0, 3, 1, 2),
                kernel_size=(7, 7),
                sigma=(3, 3),
            ).permute(0, 2, 3, 1).contiguous()
        )

        # Final mesh
        m = mesh.Mesh(
            n_vert,
            load_mesh.t_pos_idx,
            material={
                'bsdf': cfg.bsdf,
                'kd': kd_notex,
                'ks': ready_specular,
                'normal': ready_normal,
            },
            base=load_mesh # gets uvs etc from here
        )

        deformed_mesh_p3d = Meshes(verts=[m.v_pos], faces=[m.t_pos_idx])

        render_mesh = create_scene([m.eval()], sz=512)
        if it == 0:
            base_mesh = render_mesh.clone()
            base_mesh = mesh.auto_normals(base_mesh)
            base_mesh = mesh.compute_tangents(base_mesh)
        render_mesh = mesh.auto_normals(render_mesh)
        render_mesh = mesh.compute_tangents(render_mesh)

        if use_target_mesh:
            # Target mesh
            m_target = mesh.Mesh(
                target_mesh.v_pos,
                target_mesh.t_pos_idx,
                material={
                    'bsdf': cfg.bsdf,
                    'kd': kd_notex,
                    'ks': ready_specular,
                    'normal': ready_normal,
                },
                base=target_mesh
            )

            render_target_mesh = create_scene([m_target.eval()], sz=512)
            if it == 0:
                base_target_mesh = render_target_mesh.clone()
                base_target_mesh = mesh.auto_normals(base_target_mesh)
                base_target_mesh = mesh.compute_tangents(base_target_mesh)
            render_target_mesh = mesh.auto_normals(render_target_mesh)
            render_target_mesh = mesh.compute_tangents(render_target_mesh)


        # Logging mesh
        if it % cfg.log_interval == 0:
            with torch.no_grad():
                params = get_camera_params(
                    cfg.log_elev,
                    rot_ang,
                    cfg.log_dist,
                    cfg.log_res,
                    cfg.log_fov,
                )
                rot_ang += 5
                log_mesh = mesh.unit_size(render_mesh.eval(params))
                log_image = render.render_mesh(
                    glctx,
                    log_mesh,
                    params['mvp'],
                    params['campos'],
                    params['lightpos'],
                    cfg.log_light_power,
                    cfg.log_res,
                    1,
                    background=torch.ones(1, cfg.log_res, cfg.log_res, 3).to(device)
                )

                log_image = video.ready_image(log_image)
                logger.add_mesh('predicted_mesh', vertices=log_mesh.v_pos.unsqueeze(0), faces=log_mesh.t_pos_idx.unsqueeze(0), global_step=it)

        if cfg.adapt_dist and it > 0:
            with torch.no_grad():
                v_pos = m.v_pos.clone()
                vmin = v_pos.amin(dim=0)
                vmax = v_pos.amax(dim=0)
                v_pos -= (vmin + vmax) / 2
                mult = torch.cat([v_pos.amin(dim=0), v_pos.amax(dim=0)]).abs().amax().cpu()
                cams.dataset.dist_min = cfg.dist_min * mult
                cams.dataset.dist_max = cfg.dist_max * mult

        params_camera = next(iter(cams))
        for key in params_camera:
            params_camera[key] = params_camera[key].to(device)

        final_mesh = render_mesh.eval(params_camera)
        train_render = render.render_mesh(
            glctx,
            final_mesh,
            params_camera['mvp'],
            params_camera['campos'],
            params_camera['lightpos'],
            cfg.light_power,
            cfg.train_res,
            spp=1,
            num_layers=1,
            msaa=False,
            background=params_camera['bkgs']
        ).permute(0, 3, 1, 2)
        train_render = resize(train_render, out_shape=(224, 224), interp_method=resize_method)

        if use_target_mesh:
            final_target_mesh = render_target_mesh.eval(params_camera)
            train_target_render = render.render_mesh(
                glctx,
                final_target_mesh,
                params_camera['mvp'],
                params_camera['campos'],
                params_camera['lightpos'],
                cfg.light_power,
                cfg.train_res,
                spp=1,
                num_layers=1,
                msaa=False,
                background=params_camera['bkgs']
            ).permute(0, 3, 1, 2)
            train_target_render = resize(train_target_render, out_shape=(224, 224), interp_method=resize_method)

        train_rast_map = render.render_mesh(
            glctx,
            final_mesh,
            params_camera['mvp'],
            params_camera['campos'],
            params_camera['lightpos'],
            cfg.light_power,
            cfg.train_res,
            spp=1,
            num_layers=1,
            msaa=False,
            background=params_camera['bkgs'],
            return_rast_map=True
        )

        if it == 0:
            params_camera = next(iter(cams))
            for key in params_camera:
                params_camera[key] = params_camera[key].to(device)
        base_render = render.render_mesh(
            glctx,
            base_mesh.eval(params_camera),
            params_camera['mvp'],
            params_camera['campos'],
            params_camera['lightpos'],
            cfg.light_power,
            cfg.train_res,
            spp=1,
            num_layers=1,
            msaa=False,
            background=params_camera['bkgs'],
        ).permute(0, 3, 1, 2)
        base_render = resize(base_render, out_shape=(224, 224), interp_method=resize_method)

        if it % cfg.log_interval_im == 0:
            log_idx = torch.randperm(cfg.batch_size)[:5]
            s_log = train_render[log_idx, :, :, :]
            s_log = torchvision.utils.make_grid(s_log)
            ndarr = s_log.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(str(output_path / 'images' / f'epoch_{it}.png'))

            if use_target_mesh:
                s_log_target = train_target_render[log_idx, :, :, :]
                s_log_target = torchvision.utils.make_grid(s_log_target)
                ndarr = s_log_target.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(str(output_path / 'images' / f'epoch_{it}_target.png'))

            obj.write_obj(
                str(output_path / 'mesh_final'),
                m.eval()
            )

        optimizer.zero_grad()

        # Diff Rendering Loss
        #
        # clip_loss = l1_avg(train_render, train_target_render)
        # logger.add_scalar('clip_loss', clip_loss, global_step=it)

        # # CLIP similarity losses
        normalized_clip_render = (train_render - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]


        # if CLIP_embeddings:
        #     image_embeds = model.encode_image(
        #         normalized_clip_render
        #     )
        #     with torch.no_grad():
        #         normalized_base_render = (base_render - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]
        #         base_embeds = model.encode_image(normalized_base_render)
        #
        #         if use_target_mesh:
        #             normalized_target_render = (train_target_render - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]
        #             image_target_embeds = model.encode_image(
        #                 normalized_target_render
        #             )
        #
        #     orig_image_embeds = image_embeds.clone() / image_embeds.norm(dim=1, keepdim=True)
        #     delta_image_embeds = image_embeds - base_embeds
        #     delta_image_embeds = delta_image_embeds / delta_image_embeds.norm(dim=1, keepdim=True)
        #
        #     clip_loss = cosine_avg(orig_image_embeds, base_image_embeds.clone() / base_image_embeds.norm(dim=1, keepdim=True))
        #     delta_clip_loss = cosine_avg(delta_image_embeds, delta_img_embeds)

        # if use_target_mesh:
        #     image_target_embeds_norm = image_target_embeds.clone() / image_target_embeds.norm(dim=1, keepdim=True)
        #     delta_image_target_embeds = image_target_embeds - image_embeds
        #     delta_image_target_embeds = delta_image_target_embeds / delta_image_target_embeds.norm(dim=1, keepdim=True)

        # clip_loss = cosine_avg(orig_image_embeds, image_target_embeds_norm)
        # delta_clip_loss = cosine_avg(delta_image_embeds, delta_image_target_embeds)

        # clip_loss = cosine_avg(orig_image_embeds, target_text_embeds)
        # delta_clip_loss = cosine_avg(delta_image_embeds, delta_text_embeds)

        # vgg_loss = get_vgg_loss(train_render, train_target_render)

        # deformed_features = resnet(resize(train_render, out_shape=(160, 160), interp_method=resize_method))
        # target_features = resnet(resize(train_target_render, out_shape=(160, 160), interp_method=resize_method))
        # face_loss = l1_avg(deformed_features, target_features)

        # deformed_features =  fclip.encode_images(['C:\\Users\\nsarafianos\\Desktop\\garments\\jacket.jpg'], batch_size=cfg.batch_size)


        # image_embeds = fclip.encode_image_tensors(train_render)
        # orig_image_embeds = image_embeds.clone() / image_embeds.norm(dim=1, keepdim=True)
        # with torch.no_grad():
        #     base_embeds = fclip.encode_image_tensors(base_render)
        # delta_image_embeds = image_embeds - base_embeds
        # delta_image_embeds = delta_image_embeds / delta_image_embeds.norm(dim=1, keepdim=True)

        # clip_loss = cosine_avg(orig_image_embeds, target_direction_embeds)
        # delta_clip_loss = cosine_avg(delta_image_embeds, delta_direction_embeds)

        # garment_loss = l1_avg(image_embeds, target_direction_embeds)

        # clip_loss = cosine_avg(orig_image_embeds, target_text_embeds)
        # delta_clip_loss = cosine_avg(delta_image_embeds, delta_text_embeds)

        # clip_loss = cosine_avg(orig_image_embeds, target_text_embeds)
        # delta_clip_loss = cosine_avg(delta_image_embeds, delta_text_embeds)

        deformed_features = fclip.encode_image_tensors(train_render)
        target_features = fclip.encode_image_tensors(train_target_render)
        garment_loss = l1_avg(deformed_features, target_features)
        l1_loss = l1_avg(train_render, train_target_render)

        # Deform the mesh
        # new_src_mesh = src_mesh.offset_verts(deform_verts)

        # We sample 5k points from the surface of each mesh
        sample_src = sample_points_from_meshes(deformed_mesh_p3d, 10000)
        sample_trg = sample_points_from_meshes(trg_mesh_p3d, 10000)
        #
        #
        # # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
        loss_chamfer *= 25.
        #
        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(deformed_mesh_p3d)

        # mesh normal consistency
        loss_normal = mesh_normal_consistency(deformed_mesh_p3d)

        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(deformed_mesh_p3d, method="uniform")

        loss_triangles = triangle_size_regularization(deformed_mesh_p3d.verts_list()[0])/100000.



        # target_features = resnet(resize(train_target_render, out_shape=(160, 160), interp_method=resize_method))
        # face_loss = l1_avg(deformed_features, target_features)

        # print('here')
        # Calculate embedding (unsqueeze to add batch dimension)
        # deformed_features = resnet(resize(train_render, out_shape=(160, 160), interp_method=resize_method))
        # target_features = resnet(resize(train_target_render, out_shape=(160, 160), interp_method=resize_method))
        # face_loss = cosine_avg(deformed_features, target_features)
        # l1_loss = l1_avg(train_render, train_target_render)
        # clip_loss = cosine_avg(orig_image_embeds, image_target_embeds_norm)
        # delta_clip_loss = cosine_avg(delta_image_embeds, delta_image_target_embeds)

        # clip_loss_target = cosine_avg(orig_image_embeds, image_target_embeds_norm)
        # delta_clip_loss_target = cosine_avg(delta_image_embeds, delta_image_target_embeds)

        # deformed_deep_features = fe(normalized_clip_render)[cfg.consistency_vit_layer]
        # target_deep_features = fe(normalized_target_render)[cfg.consistency_vit_layer]
        # orig_image_features = fe(image)[cfg.consistency_vit_layer]
        #
        # image_embedding_loss = cosine_avg(deformed_deep_features, orig_image_features)
        # target_embedding_loss = cosine_avg(deformed_deep_features, target_deep_features)

        # logger.add_scalar('vgg_loss', face_loss, global_step=it)
        logger.add_scalar('l1_loss', l1_loss, global_step=it)
        logger.add_scalar('garment_loss', garment_loss, global_step=it)
        # logger.add_scalar('clip_loss', clip_loss, global_step=it)
        # logger.add_scalar('delta_clip_loss', delta_clip_loss, global_step=it)
        # logger.add_scalar('clip_loss_target', clip_loss_target, global_step=it)
        # logger.add_scalar('delta_clip_loss_target', delta_clip_loss_target, global_step=it)

        # logger.add_scalar('image_embedding_loss', image_embedding_loss, global_step=it)
        # logger.add_scalar('target_embedding_loss', target_embedding_loss, global_step=it)

        # Jacobian regularization
        r_loss = (((gt_jacobians) - torch.eye(3, 3, device=device)) ** 2).mean()
        logger.add_scalar('jacobian_regularization', r_loss, global_step=it)

        if cfg.consistency_loss_weight != 0:
            consistency_loss = compute_mv_cl(final_mesh, fe, normalized_clip_render, params_camera, train_rast_map, cfg, device)
        else:
            consistency_loss = r_loss
        logger.add_scalar('consistency_loss', consistency_loss, global_step=it)

        logger.add_scalar('chamfer', loss_chamfer, global_step=it)
        logger.add_scalar('edge', loss_edge, global_step=it)
        logger.add_scalar('normal', loss_normal, global_step=it)
        logger.add_scalar('laplacian', loss_laplacian, global_step=it)
        logger.add_scalar('triangles', loss_triangles, global_step=it)

        # total_loss = (cfg.clip_weight * garment_loss + # cfg.clip_weight * face_loss + #cfg.delta_clip_weight * delta_clip_loss +
        #               cfg.regularize_jacobians_weight * r_loss + cfg.consistency_loss_weight * consistency_loss)

        if it > 1000 and clip_flag:
            cfg.clip_weight = 0
            cfg.consistency_loss_weight = 0
            cfg.regularize_jacobians_weight = 0.025
            clip_flag = False
            # cfg.regularize_jacobians_weight *= 0.95
        regularizers = loss_chamfer + loss_edge + loss_normal + loss_laplacian + loss_triangles
        total_loss = (cfg.clip_weight * garment_loss + cfg.delta_clip_weight * l1_loss +
                      cfg.regularize_jacobians_weight * r_loss + cfg.consistency_loss_weight * consistency_loss + regularizers)
        # total_loss = (cfg.clip_weight * clip_loss + cfg.delta_clip_weight * delta_clip_loss +
        #               cfg.regularize_jacobians_weight * r_loss + cfg.consistency_loss_weight * consistency_loss)
        # total_loss = cfg.clip_weight * garment_loss
        logger.add_scalar('total_loss', total_loss, global_step=it)

        # if best_losses['total'] > total_loss:
        #     best_losses['total'] = total_loss.detach()
        #     obj.write_obj(
        #         str(output_path / 'mesh_best_total'),
        #         m.eval()
        #     )
        #
        # if best_losses['CLIP'] > l1_loss:#_target
        #     best_losses['CLIP'] = l1_loss.detach()#_target
        #     obj.write_obj(
        #         str(output_path / 'mesh_best_clip'),
        #         m.eval()
        #     )

        total_loss.backward()
        optimizer.step()
        t_loop.set_description(
                               f'L1 = {cfg.delta_clip_weight * l1_loss.item()}, '
                               f'CLIP = {cfg.clip_weight * garment_loss.item()}, '
                               # f'CLIP Loss = {cfg.clip_weight * clip_loss.item()}, '
                               # f'Delta CLIP Loss = {cfg.delta_clip_weight * delta_clip_loss.item()}, '
                               # f'VGG Loss = {cfg.clip_weight * face_loss.item()}, '
                               # f'Face Loss = {cfg.clip_weight * face_loss.item()}, '
                               # f'Delta CLIP Loss = {cfg.delta_clip_weight * delta_clip_loss.item()}, '
                               # f'Image CLIP Loss = {cfg.clip_weight * clip_loss_target.item()}, '
                               # f'Image Delta CLIP Loss = {cfg.delta_clip_weight * delta_clip_loss_target.item()}, '
                               f'Jacb = {cfg.regularize_jacobians_weight * r_loss.item()}, '
                               f'MVC = {cfg.consistency_loss_weight * consistency_loss.item()}, '
                               f'Chamf = {loss_chamfer.item()}, '
                               f'Edge = {loss_edge.item()}, '
                               f'Normal = {loss_normal.item()}, '
                               f'Lapl = {loss_laplacian.item()}, '
                               f'Triang = {loss_triangles.item()}, '
                               f'Total = {total_loss.item()}')#_target

    video.close()
    obj.write_obj(
        str(output_path / 'mesh_final'),
        m.eval()
    )
    
    return
