from accelerate import Accelerator
from .static_globals import *
from .better_vit_model import BetterViTModel
import ImageReward as image_reward
import torch

from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel
import numpy as np
from numpy.linalg import norm
from .aesthetic_reward import AestheticScorer
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim as cos_sim_st

def cos_sim(vector_i,vector_j)->float:
    return np.dot(vector_i,vector_j)/(norm(vector_i)*norm(vector_j))

def get_caption(image:Image,blip_processor: Blip2Processor,blip_conditional_gen: Blip2ForConditionalGeneration):
    caption_inputs = blip_processor(image, "", return_tensors="pt")
    for name in ["pixel_values","input_ids"]:
        caption_inputs[name]=caption_inputs[name].to(blip_conditional_gen.device)
    caption_out=blip_conditional_gen.generate(**caption_inputs)
    return blip_processor.decode(caption_out[0],skip_special_tokens=True).strip()

def get_vit_embeddings(vit_processor: ViTImageProcessor, vit_model: BetterViTModel, image_list:list,return_numpy:bool=True):
    vit_inputs = vit_processor(images=image_list, return_tensors="pt")
    #print("inputs :)")
    vit_inputs['pixel_values']=vit_inputs['pixel_values'].to(vit_model.device)
    vit_outputs=vit_model(**vit_inputs,output_hidden_states=True, output_past_key_values=True)
    vit_embedding_list=vit_outputs.last_hidden_state.reshape(len(image_list),-1)
    vit_style_embedding_list=[
        hidden_states[0] for hidden_states in vit_outputs.last_hidden_state #CLS token: https://github.com/google/dreambooth/issues/3
    ]
    vit_content_embedding_list=vit_outputs.past_key_values[11][0].reshape(len(image_list),-1)
    if return_numpy:
        vit_embedding_list=[v.cpu().numpy() for v in vit_embedding_list]
        vit_style_embedding_list=[v.cpu().numpy() for v in vit_style_embedding_list]
        vit_content_embedding_list=[v.cpu().numpy() for v in vit_content_embedding_list]
    return vit_embedding_list,vit_style_embedding_list, vit_content_embedding_list

@torch.no_grad()
def get_metric_dict(evaluation_prompt_list:list, evaluation_image_list:list,image_list:list,accelerator:Accelerator=None):
    metric_dict={}
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if accelerator is not None:
        clip_model.to(accelerator.device)
        clip_model=accelerator.prepare(clip_model)
    
    clip_inputs=clip_processor(text=evaluation_prompt_list, images=evaluation_image_list+image_list, return_tensors="pt", padding=True)
    clip_inputs["input_ids"]=clip_inputs["input_ids"].to(clip_model.device)
    clip_inputs["pixel_values"]=clip_inputs["pixel_values"].to(clip_model.device)

    clip_outputs = clip_model(**clip_inputs)
    src_image_n=len(image_list)
    text_embed_list=clip_outputs.text_embeds.cpu().detach().numpy()
    image_embed_list=clip_outputs.image_embeds.detach().cpu().numpy()[:-src_image_n]
    src_image_embed_list=clip_outputs.image_embeds.detach().cpu().numpy()[-src_image_n:]
    ir_model=image_reward.load("/scratch/jlb638/reward-blob",med_config="/scratch/jlb638/ImageReward/med_config.json")

    identity_consistency_list=[]
    target_similarity_list=[]
    prompt_similarity_list=[]
    for i in range(len(image_embed_list)):
        image_embed=image_embed_list[i]
        text_embed=text_embed_list[i]
        for src_image_embed in src_image_embed_list:
            target_similarity_list.append(cos_sim(image_embed,src_image_embed))
        prompt_similarity_list.append(cos_sim(image_embed, text_embed))
        for j in range(i+1, len(image_embed_list)):
            #print(i,j)
            vector_j=image_embed_list[j]
            sim=cos_sim(image_embed,vector_j)
            identity_consistency_list.append(sim)

    metric_dict[IDENTITY_CONSISTENCY]=np.mean(identity_consistency_list)
    metric_dict[TARGET_SIMILARITY]=np.mean(target_similarity_list)
    metric_dict[PROMPT_SIMILARITY]=np.mean(prompt_similarity_list)

    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_conditional_gen = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").eval()
    if accelerator is not None:
        blip_conditional_gen.to(accelerator.device)
        blip_conditional_gen=accelerator.prepare(blip_conditional_gen)
    

    src_blip_caption_list=[get_caption(src_image,blip_processor,blip_conditional_gen) for src_image in image_list]
    image_blip_caption_list=[get_caption(image,blip_processor,blip_conditional_gen) for image in evaluation_image_list]
    embedding_model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True).eval()
    if accelerator is not None:
        embedding_model.to(accelerator.device)
        embedding_model=accelerator.prepare(embedding_model)
    src_blip_embedding_list=embedding_model.encode(src_blip_caption_list)
    image_blip_embedding_list=embedding_model.encode(image_blip_caption_list)
    evaluation_blip_embedding_list=embedding_model.encode(evaluation_prompt_list)

    metric_dict[BLIP_TARGET_CAPTION_SIMILARITY]=np.mean(cos_sim_st(src_blip_embedding_list, image_blip_embedding_list).cpu().detach().numpy())
    metric_dict[BLIP_PROMPT_CAPTION_SIMILARITY]=np.mean(
        [cos_sim_st(evaluation_blip_embedding, image_blip_embedding).cpu().detach().numpy() for evaluation_blip_embedding, image_blip_embedding in zip(evaluation_blip_embedding_list, image_blip_embedding_list)]
    )

    '''blip_model=Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_inputs=blip_processor(text=evaluation_prompt_list, images=evaluation_image_list+image_list)
    blip_outputs=blip_model(**blip_inputs)'''

    vit_processor=ViTImageProcessor.from_pretrained("facebook/dino-vits16")
    vit_model=BetterViTModel.from_pretrained("facebook/dino-vits16").eval()
    if accelerator is not None:
        vit_model.to(accelerator.device)
        vit_model=accelerator.prepare(vit_model)

    image_vit_embedding_list,image_vit_style_embedding_list,image_vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,image_list)
    evaluation_image_vit_embedding_list,evaluation_image_vit_style_embedding_list,evaluation_image_vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,evaluation_image_list)

    vit_consistency_list=[]
    vit_target_similarity_list=[]
    vit_style_consistency_list=[]
    vit_style_target_similarity_list=[]
    vit_content_consistency_list=[]
    vit_content_similarity_list=[]

    for i in range(len(evaluation_image_list)):
        evaluation_vit_embedding=evaluation_image_vit_embedding_list[i]
        evaluation_vit_content_embedding=evaluation_image_vit_content_embedding_list[i]
        evaluation_vit_style_embedding=evaluation_image_vit_style_embedding_list[i]
        for k in range(len(image_list)):
            image_vit_embedding=image_vit_embedding_list[k]
            image_vit_style_embedding=image_vit_style_embedding_list[k]
            image_vit_content_embedding=image_vit_content_embedding_list[k]
            vit_target_similarity_list.append(cos_sim(image_vit_embedding, evaluation_vit_embedding))
            vit_style_target_similarity_list.append(cos_sim(evaluation_vit_style_embedding,image_vit_style_embedding))
            vit_content_similarity_list.append(cos_sim(evaluation_vit_content_embedding,image_vit_content_embedding))
        for j in range(i+1, len(evaluation_image_list)):
            secondary_evaluation_vit_embedding=evaluation_image_vit_embedding_list[j]
            secondary_evaluation_vit_content_embedding=evaluation_image_vit_content_embedding_list[j]
            secondary_evaluation_vit_style_embedding=evaluation_image_vit_style_embedding_list[j]
            vit_consistency_list.append(cos_sim(evaluation_vit_embedding,secondary_evaluation_vit_embedding))
            vit_style_consistency_list.append(cos_sim(evaluation_vit_style_embedding, secondary_evaluation_vit_style_embedding))
            vit_content_consistency_list.append(cos_sim(evaluation_vit_content_embedding,secondary_evaluation_vit_content_embedding))

    metric_dict[VIT_TARGET_SIMILARITY]=np.mean(vit_target_similarity_list)
    metric_dict[VIT_IDENTITY_CONSISTENCY]=np.mean(vit_consistency_list)
    metric_dict[VIT_STYLE_TARGET_SIMILARITY]=np.mean(vit_style_target_similarity_list)
    metric_dict[VIT_STYLE_CONSISTENCY]=np.mean(vit_style_consistency_list)
    metric_dict[VIT_CONTENT_TARGET_SIMILARITY]=np.mean(vit_content_similarity_list)
    metric_dict[VIT_CONTENT_CONSISTENCY]=np.mean(vit_content_consistency_list)

    metric_dict[IMAGE_REWARD]=np.mean(
        [ir_model.score(evaluation_prompt,evaluation_image) for evaluation_prompt,evaluation_image in zip(evaluation_prompt_list, evaluation_image_list) ]
    )
    aesthetic_scorer=AestheticScorer()
    if accelerator is not None:
        aesthetic_scorer.to(accelerator.device)
        aesthetic_scorer=accelerator.prepare(aesthetic_scorer)
    metric_dict[AESTHETIC_SCORE]=np.mean(
        [aesthetic_scorer(evaluation_image).cpu().numpy()[0] for evaluation_image in evaluation_image_list]
    )
    for metric in METRIC_LIST:
        if metric not in metric_dict:
            metric_dict[metric]=0.0
    return metric_dict