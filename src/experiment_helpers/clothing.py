'''import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Function to preprocess the image
def preprocess(image):
    preprocess_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess_transform(image).unsqueeze(0)

# Function to perform segmentation
def segment_image(image):
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    return output.argmax(0)

# Function to extract clothing pixels and create a new image with non-clothing pixels set to black
def create_clothing_only_image(image, segmentation):
    clothing_ids = [

    ]  # Adjust these IDs based on the dataset used
    mask = torch.zeros_like(segmentation, dtype=torch.uint8)
    for idx in clothing_ids:
        mask[segmentation == idx] = 1

    # Convert the mask to a numpy array and expand dimensions
    mask_np = mask.numpy().astype(np.uint8)
    mask_expanded = np.expand_dims(mask_np, axis=2)

    # Convert the original image to a numpy array
    image_np = np.array(image)

    # Apply the mask to the image
    clothing_only_image_np = image_np * mask_expanded

    # Convert the result back to a PIL image
    clothing_only_image = Image.fromarray(clothing_only_image_np.astype(np.uint8))
    return clothing_only_image
'''
import torch.nn.functional as F
from PIL import Image
from cloths_segmentation.pre_trained_models import create_model
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

import numpy as np
from torchvision.transforms import PILToTensor,Normalize,ToPILImage
import torch

class BetterUnetDecoder(UnetDecoder):
    def __init__(self,parent:UnetDecoder):
        for key, value in parent.__dict__.items():
            setattr(self, key, value)

class BetterUnet(Unet):
    def __init__(self, parent:Unet,device:str, dtype:torch.dtype):
        
        parent=parent.to(device)
        parent=parent.to(dtype)
        for key, value in parent.__dict__.items():
            setattr(self, key, value)
        self.device=device
        self.dtype=dtype

    @torch.no_grad()
    def predict_embeds(self,x)->list:
        self.check_input_shape(x)

        features = self.encoder(x)
        for f in features:
            print(f.size())

    @torch.no_grad()
    def decoder_embeds(self,x)->torch.Tensor:
        self.check_input_shape(x)

        features = self.encoder(x)

def get_segmentation_model(device:str,dtype:torch.dtype)->BetterUnet:
    parent = create_model("Unet_2020-10-30")
    return BetterUnet(parent,device,dtype)

def pad_to_multiple_of_32(image_tensor: torch.Tensor):
    _, _, h, w = image_tensor.size()
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
    padded_image = F.pad(image_tensor, padding, mode='constant', value=0)
    return padded_image, padding

# Function to unpad the image tensor
def unpad_image(image_tensor:torch.Tensor, padding:tuple):
    _, _, h, w = image_tensor.size()
    pad_h, pad_w = padding[3], padding[1]
    unpadded_image = image_tensor[:, :, :h - pad_h, :w - pad_w]
    return unpadded_image

def process_image(image: Image.Image,device:str,dtype:torch.dtype)->tuple[torch.Tensor,torch.Tensor,tuple]:
    tensor_img=PILToTensor()(image)
    tensor_img=torch.unsqueeze(tensor_img,0)
    tensor_img,padding=pad_to_multiple_of_32(tensor_img)
    base_image=tensor_img
    tensor_img=tensor_img.to(dtype)
    tensor_img=Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(tensor_img)
    
    #tensor_img,padding=pad_to_multiple_of_32(tensor_img)
    if device is not None:
        tensor_img=tensor_img.to(device)
    return tensor_img,base_image,padding


def get_mask(tensor_img:torch.Tensor,segmentation_model:BetterUnet,threshold:int)->torch.Tensor:
    prediction = segmentation_model(tensor_img)
    #segmentation_model.predict_embeds(tensor_img)
    #print(prediction.size())
    #print(torch.max(prediction),torch.min(prediction))
    #print(torch.mean(prediction),torch.max(prediction),torch.min(prediction))
    mask=(prediction > threshold).to(torch.uint8)
    return mask

@torch.no_grad()
def clothes_segmentation(image: Image.Image,segmentation_model:BetterUnet,threshold:int)->Image.Image:
    tensor_img,base_image,padding=process_image(image,segmentation_model.device,segmentation_model.dtype)
    mask=get_mask(tensor_img,segmentation_model,threshold)
    masked_tensor=base_image.to("cpu")*mask.to("cpu")
    unpadded_masked_tensor=unpad_image(masked_tensor,padding)
    return ToPILImage()(unpadded_masked_tensor.squeeze(0))


