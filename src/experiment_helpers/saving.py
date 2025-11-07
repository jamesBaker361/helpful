import torch
import json
from huggingface_hub import HfApi
from accelerate import Accelerator

WEIGHTS_NAME="diffusion_pytorch_model.safetensors"
CONFIG_NAME="config.json"

def save_state_dict(state_dict:dict,e:int,
         save_path:str,
         config_path:str,
         repo_id:str,
         weights_name=WEIGHTS_NAME,
         config_name=CONFIG_NAME,
         api:HfApi=None,
         accelerator:Accelerator=None
         ):
    
    #state_dict=???
    print("state dict len",len(state_dict))
    torch.save(state_dict,save_path)
    with open(config_path,"w+") as config_file:
        data={"start_epoch":e}
        json.dump(data,config_file, indent=4)
        pad = " " * 2048  # ~1KB of padding
        config_file.write(pad)
    print(f"saved {save_path}")
    try:
        api.upload_file(path_or_fileobj=save_path,
                                path_in_repo=weights_name,
                                repo_id=repo_id)
        api.upload_file(path_or_fileobj=config_path,path_in_repo=config_name,
                                repo_id=repo_id)
        print(f"uploaded {repo_id} to hub")
    except Exception as err:
        if accelerator is not None:
            accelerator.print("failed to upload")
            accelerator.print(err)
        else:
            print("failed to upload")
            print(err)