from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union,Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor
from copy import deepcopy

from diffusers import DiffusionPipeline,StableDiffusionControlNetPipeline
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.image_processor import PipelineImageInput

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.activations import get_activation
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    GLIGENTextBoundingboxProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)

from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import (
    get_down_block,
    get_mid_block,
    get_up_block,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel,UNet2DConditionOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from .metadata_unet import MetaDataUnet
from diffusers.utils.outputs import BaseOutput

class HydraMetaDataUnetOutput(BaseOutput):
    sample_list:List[torch.Tensor]=[]

class HydraMetaDataUnet(MetaDataUnet):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
        use_metadata: Optional[bool]=False,
        num_metadata:Optional[int]=5,
        use_metadata_3d:Optional[bool]=False,
        num_metadata_3d:Optional[int]=1,
        metadata_3d_kernel:Optional[int]=4,
        metadata_3d_stride:Optional[int]=2,
        metadata_3d_channel_list:Optional[Tuple[int, ...]] = (4, 8, 16, 32),
        metadata_3d_input_channels:Optional[int]=3,
        metadata_3d_dim:Optional[int]=512,
        n_heads:int=3,
        hydra_junction:str="mid"): #if mid each hydra head has mid and up block, if up just up block
        
        super().__init__(
            sample_size,
            in_channels,
            out_channels,
            center_input_sample,
            flip_sin_to_cos,
            freq_shift,
            down_block_types,
            mid_block_type,
            up_block_types,
            only_cross_attention,
            block_out_channels,
            layers_per_block,
            downsample_padding,
            mid_block_scale_factor,
            dropout,
            act_fn,
            norm_num_groups,
            norm_eps,
            cross_attention_dim,
            transformer_layers_per_block,
            reverse_transformer_layers_per_block,
            encoder_hid_dim,
            encoder_hid_dim_type,
            attention_head_dim,
            num_attention_heads,
            dual_cross_attention,
            use_linear_projection,
            class_embed_type,
            addition_embed_type,
            addition_time_embed_dim,
            num_class_embeds,
            upcast_attention,
            resnet_time_scale_shift,
            resnet_skip_time_act,
            resnet_out_scale_factor,
            time_embedding_type,
            time_embedding_dim,
            time_embedding_act_fn,
            timestep_post_act,
            time_cond_proj_dim,
            conv_in_kernel,
            conv_out_kernel,
            projection_class_embeddings_input_dim,
            attention_type,
            class_embeddings_concat,
            mid_block_only_cross_attention,
            cross_attention_norm,
            addition_embed_type_num_heads,
            use_metadata,
            num_metadata,
            use_metadata_3d,
            num_metadata_3d,
            metadata_3d_kernel,
            metadata_3d_stride,
            metadata_3d_channel_list,
            metadata_3d_input_channels,
            metadata_3d_dim
        )
        self.n_heads=n_heads
        self.hydra_junction=hydra_junction
        if n_heads>1:
            if hydra_junction=="mid" and self.mid_block is not None:
                self.mid_block_list=[deepcopy(self.mid_block) for _ in range(n_heads)]
            else:
                self.mid_block_list=None
            self.up_block_list=[deepcopy(self.up_blocks) for __ in range(n_heads)]
            self.conv_out_list=[deepcopy(self.conv_out) for _ in range(n_heads)]
    
    @classmethod
    def from_unet(cls,old_unet:UNet2DConditionModel,
        use_metadata: Optional[bool]=False,
        num_metadata:Optional[int]=5,
        use_metadata_3d:Optional[bool]=False,
        num_metadata_3d:Optional[int]=1,
        metadata_3d_kernel:Optional[int]=4,
        metadata_3d_stride:Optional[int]=2,
        metadata_3d_channel_list:Optional[Tuple[int, ...]] = (4, 8, 16, 32),
        metadata_3d_input_channels:Optional[int]=3,
        metadata_3d_dim:Optional[int]=512,
        n_heads:Optional[int]=3,
        hydra_junction:str="mid"):
        new_unet=cls(
            old_unet.sample_size,
            old_unet.conv_in.in_channels,
            old_unet.conv_in.out_channels,
            False, #center_input_sample,
            old_unet.time_proj.flip_sin_to_cos, # flip_sin_to_cos,
            old_unet.time_proj.downscale_freq_shift, #freq_shift,
            #[], #down_block_types,
            #[], #mid_block_type,
            #[], #up_block_types,
            use_metadata=use_metadata,
            num_metadata=num_metadata,
            use_metadata_3d=use_metadata_3d,
            num_metadata_3d=num_metadata_3d,
            metadata_3d_kernel=metadata_3d_kernel,
            metadata_3d_stride=metadata_3d_stride,
            metadata_3d_channel_list=metadata_3d_channel_list,
            metadata_3d_input_channels=metadata_3d_input_channels,
            metadata_3d_dim=metadata_3d_dim,
            n_heads=n_heads,
            hydra_junction=hydra_junction
        )
        '''False,only_cross_attention,
            block_out_channels,
            layers_per_block,
            downsample_padding,
            mid_block_scale_factor,
            dropout,
            act_fn,
            norm_num_groups,
            norm_eps,
            cross_attention_dim,
            transformer_layers_per_block,
            reverse_transformer_layers_per_block,
            encoder_hid_dim,
            encoder_hid_dim_type,
            attention_head_dim,
            num_attention_heads,
            dual_cross_attention,
            use_linear_projection,
            class_embed_type,
            addition_embed_type,
            addition_time_embed_dim,
            num_class_embeds,
            upcast_attention,
            resnet_time_scale_shift,
            resnet_skip_time_act,
            resnet_out_scale_factor,
            time_embedding_type,
            time_embedding_dim,
            time_embedding_act_fn,
            timestep_post_act,
            time_cond_proj_dim,
            conv_in_kernel,
            conv_out_kernel,
            projection_class_embeddings_input_dim,
            attention_type,
            class_embeddings_concat,
            mid_block_only_cross_attention,
            cross_attention_norm,
            addition_embed_type_num_heads'''
        try:
            new_unet.sample_size = old_unet.sample_size
        except AttributeError:
            pass
        try:
            new_unet.conv_in = old_unet.conv_in
        except AttributeError:
            pass

        try:
            new_unet.time_proj = old_unet.time_proj
        except AttributeError:
            pass

        try:
            new_unet.time_embedding = old_unet.time_embedding
        except AttributeError:
            pass

        try:
            new_unet.encoder_hid_proj = old_unet.encoder_hid_proj
        except AttributeError:
            pass

        try:
            new_unet.class_embedding = old_unet.class_embedding
        except AttributeError:
            pass

        try:
            new_unet.add_embedding = old_unet.add_embedding
        except AttributeError:
            pass

        try:
            new_unet.time_embed_act = old_unet.time_embed_act
        except AttributeError:
            pass

        try:
            new_unet.up_blocks = old_unet.up_blocks
        except AttributeError:
            pass

        try:
            new_unet.down_blocks = old_unet.down_blocks
        except AttributeError:
            pass

        try:
            new_unet.num_upsamplers = old_unet.num_upsamplers
        except AttributeError:
            pass

        try:
            new_unet.mid_block=old_unet.mid_block
        except AttributeError:
            pass

        try:
            new_unet.conv_norm_out = old_unet.conv_norm_out
        except AttributeError:
            pass

        try:
            new_unet.conv_act = old_unet.conv_act
        except AttributeError:
            pass

        try:
            new_unet.conv_out = old_unet.conv_out
        except AttributeError:
            pass

        try:
            new_unet.position_net = old_unet.position_net
        except AttributeError:
            pass

        try:
            new_unet.config=old_unet.config
        except AttributeError:
            pass

        if n_heads>1:
            if hydra_junction=="mid" and old_unet.mid_block is not None:
                new_unet.mid_block_list=[deepcopy(old_unet.mid_block) for _ in range(n_heads)]
            else:
                new_unet.mid_block_list=None
            new_unet.up_block_list=[deepcopy(old_unet.up_blocks) for _ in range(n_heads)]
            new_unet.conv_out_list=[deepcopy(old_unet.conv_out) for _ in range(n_heads)]
        return new_unet

    def forward(self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        metadata:Optional[Tensor]=None,
        metadata_3d:Optional[Tensor]=None,

    ) -> Union[List[UNet2DConditionOutput], List[Tuple]]:
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # CUSTOM: metadata
        if self.metadata_embedding is not None:
            assert metadata is not None
            assert len(metadata.shape) == 2 and metadata.shape[1] == self.num_metadata, \
                f"Invalid metadata shape: {metadata.shape}. Need batch x num_metadata"

            md_bsz = metadata.shape[0]
            # invalid_metadata_mask = metadata == -1.  # (N, num_md)
            metadata = self.time_proj(metadata.view(-1)).view(md_bsz, self.num_metadata, -1)  # (N, num_md, D)
            # metadata[invalid_metadata_mask] = 0.
            metadata = metadata.to(dtype=self.dtype)
            for i, md_embed in enumerate(self.metadata_embedding):
                md_emb = md_embed(metadata[:, i, :])  # (N, D)
                emb = emb + md_emb  # (N, D)

        #CUSTOM: metadata 3d
        if self.metadata_3d_embedding is not None:
            assert metadata_3d is not None
            md_bsz=metadata_3d.shape[0]
            metadata_3d=metadata_3d.to(dtype=self.dtype)
            for i,md_embed_3d in enumerate(self.metadata_3d_embedding):
                md_emb_3d=md_embed_3d(metadata_3d[:,i,:,:,:,:])
                emb=emb+md_emb_3d

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        #hydra stuff
        # 4. mid
        sample_list=[]
        if self.n_heads>1 and self.hydra_junction=="mid":
            
            for mid_block in self.mid_block_list:
                if hasattr(mid_block, "has_cross_attention") and mid_block.has_cross_attention:
                    new_sample = mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    new_sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0
                    and new_sample.shape == down_intrablock_additional_residuals[0].shape
                ):
                    new_sample += down_intrablock_additional_residuals.pop(0)

            if is_controlnet:
                new_sample = new_sample + mid_block_additional_residual
            sample_list.append(new_sample)


        else:
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0
                    and sample.shape == down_intrablock_additional_residuals[0].shape
                ):
                    sample += down_intrablock_additional_residuals.pop(0)

            if is_controlnet:
                sample = sample + mid_block_additional_residual

        # 5. up
        final_sample_list=[]
        if self.n_heads>1:
            for index,up_blocks in enumerate(self.up_block_list):
                if len(sample_list)>0:
                    new_sample=sample_list[index]
                else:
                    new_sample=sample
                for i, upsample_block in enumerate(up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        new_sample = upsample_block(
                            hidden_states=new_sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                        )
                    else:
                        new_sample = upsample_block(
                            hidden_states=new_sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            upsample_size=upsample_size,
                        )
            if self.conv_norm_out:
                new_sample=self.conv_norm_out(new_sample)
                new_sample=self.conv_act(new_sample)
            new_sample=self.conv_out_list[index](new_sample)
            final_sample_list.append(new_sample)
        else:
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                    )

            # 6. post-process
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            sample = self.conv_out(sample)
            final_sample_list=[sample]


        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (final_sample_list,)

        return HydraMetaDataUnetOutput(sample_list=final_sample_list)