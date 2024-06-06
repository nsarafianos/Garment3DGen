import torch
from PIL import Image


def get_fashion_text_embeddings(fclip, cfg, device):
    print(f'Target text prompt is {cfg.text_prompt}')
    print(f'Base text prompt is {cfg.base_text_prompt}')
    with torch.no_grad():
        text_embeds = fclip.encode_text_tensors([cfg.text_prompt]).detach()
        base_text_embeds = fclip.encode_text_tensors([cfg.base_text_prompt]).detach()
        target_text_embeds = text_embeds.clone() / text_embeds.norm(dim=1, keepdim=True)
        delta_text_embeds = text_embeds - base_text_embeds
        delta_text_embeds = delta_text_embeds / delta_text_embeds.norm(dim=1, keepdim=True)
    return target_text_embeds.to(device), delta_text_embeds.to(device)


def get_fashion_img_embeddings(fclip, cfg, device, normalize=True):
    print(f'Target image path is {cfg.image_prompt}')
    print(f'Base image path is {cfg.base_image_prompt}')
    with torch.no_grad():
        target_image_embeds = fclip.encode_images([cfg.image_prompt], 1)
        target_image_embeds = torch.tensor(target_image_embeds, device=device).detach()

        base_image_embeds = fclip.encode_images([cfg.base_image_prompt], 1)
        base_image_embeds = torch.tensor(base_image_embeds, device=device).detach()
        delta_img_embeds = target_image_embeds - base_image_embeds
    if normalize:
        delta_img_embeds = delta_img_embeds / delta_img_embeds.norm(dim=1, keepdim=True)
        target_image_embeds = target_image_embeds.clone() / target_image_embeds.norm(dim=1, keepdim=True)
    return target_image_embeds.to(device), delta_img_embeds.to(device)


def get_text_embeddings(clip, model, cfg, device):
    print(f'Target text prompt is {cfg.text_prompt}')
    print(f'Base text prompt is {cfg.base_text_prompt}')
    text_embeds = clip.tokenize(cfg.text_prompt).to(device)
    base_text_embeds = clip.tokenize(cfg.base_text_prompt).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(text_embeds).detach()
        target_text_embeds = text_embeds.clone() / text_embeds.norm(dim=1, keepdim=True)
        delta_text_embeds = text_embeds - model.encode_text(base_text_embeds)
        delta_text_embeds = delta_text_embeds / delta_text_embeds.norm(dim=1, keepdim=True)
    return target_text_embeds, delta_text_embeds


def get_img_embeddings(model, preprocess, cfg, device):
    print(f'Target image path is {cfg.image_prompt}')
    print(f'Base image path is {cfg.base_image_prompt}')

    image = preprocess(Image.open(cfg.image_prompt)).unsqueeze(0).to(device)
    base_image = preprocess(Image.open(cfg.base_image_prompt)).unsqueeze(0).to(device)
    with torch.no_grad():
        target_image_embeds = model.encode_image(image).to(device).detach()
        base_image_embeds = model.encode_image(base_image).to(device)

        delta_img_embeds = target_image_embeds - base_image_embeds
        delta_img_embeds = delta_img_embeds / delta_img_embeds.norm(dim=1, keepdim=True)
    return target_image_embeds, delta_img_embeds
