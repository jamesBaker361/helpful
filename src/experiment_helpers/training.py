from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
import torch
from accelerate import Accelerator
from random import randint
from PIL import Image
from torchvision.transforms import PILToTensor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import torch.nn.functional as F
from tqdm.auto import tqdm
import time

def pil_to_tensor_process(image:Image):
    tensor=PILToTensor()(image)
    if torch.max(tensor)>1.0 and torch.min(tensor)>=0:
        tensor=tensor/128.0 -1.0
    elif torch.min(tensor)>=0:
        tensor=tensor*2.0 -1.0
    if tensor.ndim==4:
        tensor=tensor[0]
    return tensor

def encode_prompt(text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,prompt:str):
    text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
    text_input_ids = text_inputs.input_ids
    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), attention_mask=attention_mask)
    #print(prompt_embeds)
    #print(dir(prompt_embeds))
    #secondary_prompt_embeds=text_encoder(text_input_ids.to(text_encoder.device), attention_mask=attention_mask,return_dict=False)
    #print(secondary_prompt_embeds)
    #prompt_embeds = prompt_embeds[0]
    return prompt_embeds.last_hidden_state

def train_unet(pipeline:StableDiffusionPipeline,
                   epochs:int,
                   training_image_list:list,
                   training_prompt_list:list,
                   optimizer:torch.optim.Optimizer,
                   use_prior_preservation:bool,
                   prior_class:str,
                   batch_size:int,
                   max_grad_norm:float,
                   entity_name:str,
                   accelerator:Accelerator,
                   num_inference_steps:int,
                   prior_loss_weight:float,
                   track_loss:bool=False)->StableDiffusionPipeline:
    '''
    prior_class= "cat" or "man" or whatever
    training_prompt_list= ["{} doing a thing"] will be formatted with entity_name or if 
        using prior entity name + prior class and prior class
    '''
    
    pipeline.safety_checker=None
    unet=pipeline.unet
    vae=pipeline.vae
    text_encoder=pipeline.text_encoder
    tokenizer=pipeline.tokenizer
    noise_scheduler=pipeline.scheduler

    _training_image_list=[i for i in training_image_list]
    _training_prompt_list=[p for p in training_prompt_list]
    index=0
    while len(_training_image_list)%batch_size!=0:
        index=index%len(training_image_list)
        _training_image_list.append(training_image_list[index])
        _training_prompt_list.append(training_prompt_list[index])
        index+=1
    training_prompt_list=_training_prompt_list
    training_image_list=_training_image_list

    initial_text_encoder_device=text_encoder.device
    text_encoder=text_encoder.to(unet.device)
    initial_vae_device=vae.device
    vae=vae.to(unet.device)
    pipeline("do this to instantiate things",num_inference_steps=1)
    width,height=training_image_list[0].size
    if use_prior_preservation:
        #prior_prompt_list=[p.format(prior_class) for p in training_prompt_list]
        prior_prompt_list=[encode_prompt(text_encoder,tokenizer,p.format(prior_class)) for p in training_prompt_list]
        training_prompt_list=[encode_prompt(text_encoder,tokenizer,p.format(f"{entity_name} {prior_class}")) for p in training_prompt_list]
        prior_image_list=[
            pil_to_tensor_process(pipeline(prompt_embeds=prompt_embeds,num_inference_steps=num_inference_steps,safety_checker=None,width=width,height=height).images[0])
            for prompt_embeds in prior_prompt_list
        ]
        training_image_list=[
            pil_to_tensor_process(image) for image in training_image_list
        ]
        training_image_list_batched=[
            training_image_list[i:i+batch_size]+prior_image_list[i:i+batch_size] for i in range(0,len(training_image_list),batch_size)
        ]
        training_prompt_list_batched=[
            training_prompt_list[i:i+batch_size]+prior_prompt_list[i:i+batch_size] for i in range(0,len(training_image_list),batch_size)
        ]
    else:
        training_image_list=[
            pil_to_tensor_process(image) for image in training_image_list
        ]
        training_prompt_list=[encode_prompt(text_encoder,tokenizer,p.format(f"{entity_name}")) for p in training_prompt_list]
        training_image_list_batched=[
            training_image_list[i:i+batch_size] for i in range(0,len(training_image_list),batch_size)
        ]
        training_prompt_list_batched=[
            training_prompt_list[i:i+batch_size] for i in range(0,len(training_image_list),batch_size)
        ]
    text_encoder=text_encoder.to(initial_text_encoder_device)
    vae=vae.to(initial_vae_device)
    training_image_data=[
        torch.stack(batch) for batch in training_image_list_batched
    ]
    training_prompts_data=[
        torch.cat(batch) for batch in training_prompt_list_batched
    ]

    progress_bar = tqdm(
        range(0, epochs),
        initial=0,
        desc="epochs",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    '''print('unet device',unet.device)
    print('vae device',vae.device)
    print('text encoder device',text_encoder.device )'''
    start=time.time()
    for e in range(epochs):
        unet.train()
        for images,prompts in zip(training_image_data, training_prompts_data):
            with accelerator.accumulate(unet):
                '''if use_prior_preservation:
                    images,prior_images=images.chunk(2)
                    prompts,prior_prompts=images.chunk(2)'''
                images=images.to(vae.device)
                model_input = vae.encode(images).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                model_input=model_input.to(unet.device)
                prompts=prompts.to(unet.device)

                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=unet.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                noisy_model_input=noisy_model_input.to(accelerator.device)
                '''print('unet device',unet.device)
                print('noisy_model_input.device',noisy_model_input.device)
                print('timesteps.device',timesteps.device)
                print('prompts.device',prompts.device)'''
                model_pred = unet(
                    noisy_model_input, timesteps, prompts, return_dict=False
                )[0]
                if use_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)
                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), noise_prior.float(), reduction="mean")
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                if use_prior_preservation:
                    loss=loss+prior_loss_weight*prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                    if track_loss:
                        try:
                            accelerator.log({
                                "loss":loss
                            })
                        except:
                            accelerator.log({
                                "loss":loss.detach().cpu().numpy()
                            })
                optimizer.step()
                optimizer.zero_grad()
        if accelerator.sync_gradients:
            progress_bar.update(1)
    end=time.time()
    print(f"trained unet! time elapsed: {end-start}")
    return pipeline