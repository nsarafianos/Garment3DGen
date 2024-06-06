import os
import yaml
import torch
import random
import argparse
import numpy as np

from loop import loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', type=str, default='./example_config.yml')
    parser.add_argument('--output_path', help='Output directory (will be created)', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--gpu', help='GPU index', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--seed', help='Random seed', type=int, default=argparse.SUPPRESS)

    # CLIP-related
    parser.add_argument('--text_prompt', help='Target text prompt', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--base_text_prompt', help='Base text prompt describing input mesh', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--clip_model', help='CLIP Model for text comparison', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--consistency_clip_model', help='CLIP Model for consistency', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--consistency_vit_stride', help='New stride for ViT patch interpolation', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--consistency_vit_layer', help='Which layer to take ViT patch features from (0-11)', type=int, default=argparse.SUPPRESS)

    # Mesh
    parser.add_argument('--mesh', help='Path to input mesh', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--retriangulate', help='Use isotropic remeshing', type=int, default=argparse.SUPPRESS, choices=[0, 1])

    # Render settings
    parser.add_argument('--bsdf', help='Render technique', type=str, default=argparse.SUPPRESS, choices=['diffuse', 'pbr'])

    # Hyper-parameters
    parser.add_argument('--lr', help='Learning rate', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--epochs', help='Number of optimization steps', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--clip_weight', help='Weight for CLIP loss', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--delta_clip_weight', help='Wight for delta-CLIP loss', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--regularize_jacobians_weight', help='Weight for jacobian regularization', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--consistency_loss_weight', help='Weight for viewpoint consistency penalty', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--consistency_elev_filter', help='Elev. angle threshold for filtering out pairs of viewpoints for consistency loss', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--consistency_azim_filter', help='Azim. angle threshold for filtering out pairs of viewpoints for consistency loss', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--batch_size', help='Number of images rendered at the same time', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--train_res', help='Resolution of render before downscaling to CLIP size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--resize_method', help='Image downsampling/upsampling  method', type=str, default=argparse.SUPPRESS, choices=['cubic', 'linear', 'lanczos2', 'lanczos3'])
    ## Camera Parameters ##
    parser.add_argument('--fov_min', help='Minimum camera field of view angle during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--fov_max', help='Maximum camera field of view angle during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dist_mi  n', help= 'Minimum distance of camera from mesh during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dist_max', help='Maximum distance of camera from mesh during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--light_power', help='Light intensity', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_alpha', help='Alpha parameter for Beta distribution for elevation sampling', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_beta', help='Beta parameter for Beta distribution for elevation sampling', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_max', help='Maximum elevation anglez in degree', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--azim_min', help='Minimum azimuth angle in degree',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--azim_max', help='Maximum azimuth angle in degree', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--aug_loc', help='Offset mesh from center of image?', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--aug_light', help='Augment the direction of light around the camera', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--aug_bkg', help='Augment the background', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--adapt_dist', help='Adjust camera distance to account for scale of shape', type=int, default=argparse.SUPPRESS, choices=[0, 1])

    # Logging
    parser.add_argument('--log_inter val', help='Interval for logging, every X epochs',  type=int, default=argparse.SUPPRESS)
    parser.add_argument('--log_interval_im', help='Interval for logging renders image, every X epochs',  type=int, default=argparse.SUPPRESS)
    parser.add_argument('--log_elev', help='Logging elevation angle',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_fov', help='Logging field of view',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_dist', help='Logging distance from object',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_res', help='Logging render resolution',  type=int, default=argparse.SUPPRESS)
    parser.add_argument('--log_light_power', help='Light intensity for logging',  type=float, default=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
    
    for key in vars(args):
        cfg[key] = vars(args)[key]

    print(yaml.dump(cfg, default_flow_style=False))
    random.seed(cfg['seed'])
    os.environ['PYTHONHASHSEED'] = str(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = True

    loop(cfg)
    print('Done')

if __name__ == '__main__':
    main()

