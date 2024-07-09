# Garment3DGen: 3D Garment Stylization and Texture Generation

![](https://nsarafianos.github.io/assets/garment3dgen/teaser.png)


### [Project Page](https://nsarafianos.github.io/garment3dgen), [Paper](https://arxiv.org/abs/2403.18816), [Video](https://nsarafianos.github.io/assets/garment3dgen/video.mp4)
### **TL;DR**: Garment3DGen stylizes the geometry and textures of real and fantastical garments that we can fit on top of parametric bodies and simulate.



## Installation 
Tested with Windows 10, python 3.8, CUDA 11.8 but it should be significantly easier to set it up on Linux


### Basic Dependencies
Create a fresh virtual_env and then:
```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r .\requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### Repository Dependencies
1. First we need to install nvdiffrast
```
mkdir packages
cd packages
git clone https://github.com/NVlabs/nvdiffrast.git
cd .\nvdiffrast\
pip install .
cd ..
```
2. Then we set up PyTorch3D 
```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
set DISTUTILS_USE_SDK=1
python setup.py install
cd ..
```
3. And finally for Fashion-CLIP (already provided under packages. Make sure you have the dependences)

```
git clone https://github.com/patrickjohncyh/fashion-clip.git
cd fashion-clip
pip install appdirs boto3 annoy validators  transformers datasets
cd ..\..
```

## Data
1. Under ./meshes/ you can find a handful of garment geometries that we used for our work. 
2. Under ./meshes_target/ you can place the target geometries you'd like to deform your input mesh to. You can obtain these very easily by passing an RGB image to [InstantMesh](https://github.com/TencentARC/InstantMesh). You can choose an image you already have, or generate one with text/sketch to image generation models. 
3. Aim for your target images to not have intersections and ideally have stretching arms (not touching the other parts of the geometry)
4. Aim for your source and target geometry to be reasonably close to each other. Going from a skirt to a shirt won't work well. 


## How to Run
> python main.py

## Acknowledgements
1. This work relies heavily on [TextDeformer](https://github.com/threedle/TextDeformer) and [Neural Jacobian Fields](https://github.com/ThibaultGROUEIX/NeuralJacobianFields) so if you find our work useful please all cite these works appropriately
2. [Fashion-CLIP](https://github.com/patrickjohncyh/fashion-clip) is essential to obtain more meaningful embeddings so please cite their article
3. In the original paper we relied on [Wonder3D](https://github.com/xxlong0/Wonder3D)+[Instant-NSR](https://github.com/zhaofuq/Instant-NSR) to obtain the target geometries. We've currently switched to [InstantMesh](https://github.com/TencentARC/InstantMesh) so please cite both works. 

## Reference
```
@article{
    sarafianos2024garment3dgen,
    title={Garment3DGen: 3D Garment Stylization and Texture Generation},
    author={Sarafianos, Nikolaos and Stuyck, Tuur and Xiang, Xiaoyu and Li, Yilei and Popovic, Jovan and Ranjan, Rakesh},
    journal={arXiv preprint arXiv:2403.18816},
    year={2024}
}
```

<div align="center">
  <img width="600" src="asset_visualization/armor.gif">
</div>