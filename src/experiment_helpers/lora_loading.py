from trl import DefaultDDPOStableDiffusionPipeline
import os

from safetensors import safe_open
from safetensors.torch import save_file
from peft import get_peft_model_state_dict
from huggingface_hub import HfApi,snapshot_download,create_repo
from .better_ddpo_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from huggingface_hub import hf_hub_download

def save_lora_weights(pipeline:DefaultDDPOStableDiffusionPipeline,output_dir:str):
    state_dict=get_peft_model_state_dict(pipeline.sd_pipeline.unet, unwrap_compiled=True)
    weight_path=os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    print("saving to ",weight_path)
    save_file(state_dict, weight_path, metadata={"format": "pt"})

def load_lora_weights(pipeline:DefaultDDPOStableDiffusionPipeline,path:str):
    #pipeline.get_trainable_layers()
    print("loading from ",path)
    state_dict={}
    count=0
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key]=f.get_tensor(key)
    state_dict={
        k.replace("weight","default.weight"):v for k,v in state_dict.items()
    }
    param_set=set([p[0] for p in pipeline.sd_pipeline.unet.named_parameters()])
    for k in state_dict.keys():
        if k in param_set:
            count+=1
    print(f"loaded {count} params")
    pipeline.sd_pipeline.unet.load_state_dict(state_dict,strict=False)
    print("successfully loaded")

def save_pipeline_hf(pipeline:DefaultDDPOStableDiffusionPipeline, hub_model_id:str,temp_dir="/scratch/jlb638/temp_ddpo"):
    os.makedirs(temp_dir,exist_ok=True)
    save_lora_weights(pipeline,temp_dir)
    api = HfApi()
    create_repo(hub_model_id,repo_type="model",exist_ok=True)
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=hub_model_id,
        repo_type="model",
    )
    print(f"uploaded to {hub_model_id}")
    snapshot_download(hub_model_id,repo_type="model")
    print(f"downloaded snapshot {hub_model_id}")

def get_pipeline_from_hf(hf_model_id:str,train_text_encoder:bool,
                 train_text_encoder_embeddings:bool,
                 train_unet:bool,
                  use_lora_text_encoder:bool,
                  use_lora:bool,
                  pretrained_model_name:str="runwayml/stable-diffusion-v1-5",
                  device="cpu"):
    pipeline=BetterDefaultDDPOStableDiffusionPipeline(
        train_text_encoder,
        train_text_encoder_embeddings,
        train_unet,
        use_lora_text_encoder,
        use_lora=use_lora,
        pretrained_model_name=pretrained_model_name
    )
    weight_path=hf_hub_download(repo_id=hf_model_id, filename="pytorch_lora_weights.safetensors",repo_type="model")
    load_lora_weights(pipeline,weight_path)
    pipeline.sd_pipeline.unet.to(device)
    pipeline.sd_pipeline.text_encoder.to(device)
    pipeline.sd_pipeline.vae.to(device)
    print(f"loaded from {hf_model_id}")
    return pipeline
