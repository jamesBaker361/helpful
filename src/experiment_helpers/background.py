from backgroundremover.bg import detect, alpha_matting_cutout,naive_cutout,get_model
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from typing import Union, Tuple



def remove_background(img:Image.Image,model_name="u2net",
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_structure_size=10,
    alpha_matting_base_size=1000,)->Image.Image:
    model=get_model(model_name)
    mask = detect.predict(model, np.array(img)).convert("L")

    if alpha_matting:
        cutout = alpha_matting_cutout(
            img,
            mask,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_structure_size,
            alpha_matting_base_size,
        )
    else:
        cutout = naive_cutout(img, mask)

    return cutout.convert("RGB")

def remove_background_birefnet(image:Image.Image,birefnet: AutoModelForImageSegmentation=None,return_mask:bool=False)-> Union[Image.Image, Tuple[Image.Image, Union[torch.Tensor, np.ndarray]]]:
    if birefnet is None:
        birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_images = transform_image(image).unsqueeze(0).to(birefnet.device)
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    black_pixel = (0, 0, 0)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) == 0:  # Mask pixel value > 0 means apply the black color
                image.putpixel((x, y), black_pixel)
    if return_mask:
        return image,preds
    return image