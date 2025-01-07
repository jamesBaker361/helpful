import os
import sys
sys.path.append('/home/jlb638/Desktop/package')
from src.experiment_helpers.unet_hydra import *
import unittest
from PIL import Image

batch_size=1
num_metadata=2
metadata=torch.rand((batch_size,num_metadata))
canny_image=Image.open("ArcaneJinx.jpg")
class TestHydra(unittest.TestCase):
    def test_forward(self):
        for use_metadata in [True,False]:
            for use_control in [True,False]:
                for n_heads in [1,3]:
                    for use_hydra_down in [True,False]:
                            with self.subTest(use_metadata=use_metadata, use_control=use_control, n_heads=n_heads, use_hydra_down=use_hydra_down):
                                kwargs={"num_inference_steps":2}
                                if use_control:
                                    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
                                    pipe=StableDiffusionControlNetPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",controlnet=controlnet)
                                    kwargs["image"]=canny_image
                                else:
                                    pipe=DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
                                pipe.unet=HydraMetaDataUnet.from_unet(pipe.unet,use_metadata=use_metadata,n_heads=n_heads,use_hydra_down=use_hydra_down,num_metadata=num_metadata)
                                if use_metadata:
                                    kwargs["metadata"]=metadata
                                forward_hydra(pipe, **kwargs)


if __name__=="__main__":
    unittest.main()