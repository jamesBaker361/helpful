import torch
from torch import nn

import inspect
import math
from typing import Callable, List, Optional, Tuple, Union,Dict
import torch.nn.functional as F



from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import is_torch_version, maybe_allow_in_graph
from diffusers.models.attention_processor import Attention

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from huggingface_hub.utils import validate_hf_hub_args
from safetensors import safe_open

from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_state_dict
from diffusers.utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    is_transformers_available,
    logging,
)

from diffusers.models.modeling_utils import load_model_dict_into_meta, load_state_dict
from diffusers.loaders.unet_loader_utils import _maybe_expand_lora_scales
from diffusers.loaders.ip_adapter import IPAdapterMixin
from diffusers.loaders.unet import UNet2DConditionLoadersMixin
from diffusers.models.embeddings import (
    ImageProjection,
    IPAdapterFaceIDImageProjection,
    IPAdapterFaceIDPlusImageProjection,
    IPAdapterFullImageProjection,
    IPAdapterPlusImageProjection,
    MultiIPAdapterImageProjection,
)

from contextlib import nullcontext


if is_transformers_available():
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    from diffusers.models.attention_processor import (
        AttnProcessor,
        AttnProcessor2_0,
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0,
    )

logger = logging.get_logger(__name__)

class IPAdapterAttnProcessorValue(IPAdapterAttnProcessor2_0): #https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        # for ip-adapter
        for current_ip_hidden_states, scale,  to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        #ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        #ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                        # the output of sdp = (batch, num_heads, seq_len, head_dim)
                        # TODO: add support for attn.scale when we move to Torch 2.1
                        _current_ip_hidden_states = F.scaled_dot_product_attention(
                            query, key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )
                        _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                        hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                else:
                    #ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    #ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1

                    padding_size=(ip_value.size()[0], ip_value.size()[1],77-ip_value.size()[2],ip_value.size()[3])
                    padding_tensor=torch.ones(padding_size).to(ip_value.device)
                    ip_value=torch.cat((ip_value,padding_tensor),dim=2)

                    current_ip_hidden_states = F.scaled_dot_product_attention(
                        query, key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )

                    current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                        batch_size, -1, attn.heads * head_dim
                    )
                    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                    hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

class IPAdapterAttnProcessorKey(IPAdapterAttnProcessor2_0): #https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        # for ip-adapter
        for current_ip_hidden_states, scale,  to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        #ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        #ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                        # the output of sdp = (batch, num_heads, seq_len, head_dim)
                        # TODO: add support for attn.scale when we move to Torch 2.1
                        _current_ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )
                        _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                        hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    
                    #ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    #ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    #print("value size",value.size())
                    #print('key.size()',key.size())
                    #print('ip_value.size()',ip_value.size())
                    #print('ip_key.size()',ip_key.size())
                    padding_size=(ip_key.size()[0], ip_key.size()[1],77-ip_key.size()[2],ip_key.size()[3])
                    padding_tensor=torch.ones(padding_size).to(ip_key.device)
                    ip_key=torch.cat((ip_key,padding_tensor),dim=2)
                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1
                    current_ip_hidden_states = F.scaled_dot_product_attention(
                        query, ip_key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )

                    current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                        batch_size, -1, attn.heads * head_dim
                    )
                    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                    hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

def load_ip_adapter_single( #https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/ip_adapter.py#L55
        self:IPAdapterMixin,
        pretrained_model_name_or_path_or_dict: Union[str, List[str], Dict[str, torch.Tensor]],
        subfolder: Union[str, List[str]],
        weight_name: Union[str, List[str]],
        variant: str,
        image_encoder_folder: Optional[str] = "image_encoder",
        **kwargs,):
    """
    Parameters:
        pretrained_model_name_or_path_or_dict (`str` or `List[str]` or `os.PathLike` or `List[os.PathLike]` or `dict` or `List[dict]`):
            Can be either:

                - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                    the Hub.
                - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                    with [`ModelMixin.save_pretrained`].
                - A [torch state
                    dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
        subfolder (`str` or `List[str]`):
            The subfolder location of a model file within a larger model repository on the Hub or locally. If a
            list is passed, it should have the same length as `weight_name`.
        weight_name (`str` or `List[str]`):
            The name of the weight file to load. If a list is passed, it should have the same length as
            `weight_name`.
        variant (str):
            one of "key" or "value"
        image_encoder_folder (`str`, *optional*, defaults to `image_encoder`):
            The subfolder location of the image encoder within a larger model repository on the Hub or locally.
            Pass `None` to not load the image encoder. If the image encoder is located in a folder inside
            `subfolder`, you only need to pass the name of the folder that contains image encoder weights, e.g.
            `image_encoder_folder="image_encoder"`. If the image encoder is located in a folder other than
            `subfolder`, you should pass the path to the folder that contains image encoder weights, for example,
            `image_encoder_folder="different_subfolder/image_encoder"`.
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.

        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
            Speed up model loading only loading the pretrained weights and not initializing the weights. This also
            tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
            Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
            argument to `True` will raise an error.
    """

    # handle the list inputs for multiple IP Adapters
    if not isinstance(weight_name, list):
        weight_name = [weight_name]

    if not isinstance(pretrained_model_name_or_path_or_dict, list):
        pretrained_model_name_or_path_or_dict = [pretrained_model_name_or_path_or_dict]
    if len(pretrained_model_name_or_path_or_dict) == 1:
        pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict * len(weight_name)

    if not isinstance(subfolder, list):
        subfolder = [subfolder]
    if len(subfolder) == 1:
        subfolder = subfolder * len(weight_name)

    if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
        raise ValueError("`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length.")

    if len(weight_name) != len(subfolder):
        raise ValueError("`weight_name` and `subfolder` must have the same length.")

    # Load the main state dict first.
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

    if low_cpu_mem_usage and not is_accelerate_available():
        low_cpu_mem_usage = False
        logger.warning(
            "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
            " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
            " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
            " install accelerate\n```\n."
        )

    if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
            " `low_cpu_mem_usage=False`."
        )

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }
    state_dicts = []
    for pretrained_model_name_or_path_or_dict, weight_name, subfolder in zip(
        pretrained_model_name_or_path_or_dict, weight_name, subfolder
    ):
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = load_state_dict(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["image_proj", "ip_adapter"]:
            raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

        state_dicts.append(state_dict)

        # load CLIP image encoder here if it has not been registered to the pipeline yet
        if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
            if image_encoder_folder is not None:
                if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                    logger.info(f"loading image_encoder from {pretrained_model_name_or_path_or_dict}")
                    if image_encoder_folder.count("/") == 0:
                        image_encoder_subfolder = Path(subfolder, image_encoder_folder).as_posix()
                    else:
                        image_encoder_subfolder = Path(image_encoder_folder).as_posix()

                    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                        pretrained_model_name_or_path_or_dict,
                        subfolder=image_encoder_subfolder,
                        low_cpu_mem_usage=low_cpu_mem_usage,
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                    ).to(self.device, dtype=self.dtype)
                    self.register_modules(image_encoder=image_encoder)
                else:
                    raise ValueError(
                        "`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict."
                    )
            else:
                logger.warning(
                    "image_encoder is not loaded since `image_encoder_folder=None` passed. You will not be able to use `ip_adapter_image` when calling the pipeline with IP-Adapter."
                    "Use `ip_adapter_image_embeds` to pass pre-generated image embedding instead."
                )

        # create feature extractor if it has not been registered to the pipeline yet
        if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is None:
            # FaceID IP adapters don't need the image encoder so it's not present, in this case we default to 224
            default_clip_size = 224
            clip_image_size = (
                self.image_encoder.config.image_size if self.image_encoder is not None else default_clip_size
            )
            feature_extractor = CLIPImageProcessor(size=clip_image_size, crop_size=clip_image_size)
            self.register_modules(feature_extractor=feature_extractor)

    # load ip-adapter into unet
    unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
    #unet._load_ip_adapter_weights(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage) #this we're changing tho
    _load_ip_adapter_weights_single(unet,state_dicts,variant,low_cpu_mem_usage)

    extra_loras = unet._load_ip_adapter_loras(state_dicts)
    if extra_loras != {}:
        if not USE_PEFT_BACKEND:
            logger.warning("PEFT backend is required to load these weights.")
        else:
            # apply the IP Adapter Face ID LoRA weights
            peft_config = getattr(unet, "peft_config", {})
            for k, lora in extra_loras.items():
                if f"faceid_{k}" not in peft_config:
                    self.load_lora_weights(lora, adapter_name=f"faceid_{k}")
                    self.set_adapters([f"faceid_{k}"], adapter_weights=[1.0])



def _load_ip_adapter_weights_single(self:UNet2DConditionLoadersMixin, state_dicts,variant:str, low_cpu_mem_usage:bool=False): #https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/unet.py#L823
    if not isinstance(state_dicts, list):
        state_dicts = [state_dicts]

    # Kolors Unet already has a `encoder_hid_proj`
    if (
        self.encoder_hid_proj is not None
        and self.config.encoder_hid_dim_type == "text_proj"
        and not hasattr(self, "text_encoder_hid_proj")
    ):
        self.text_encoder_hid_proj = self.encoder_hid_proj

    # Set encoder_hid_proj after loading ip_adapter weights,
    # because `IPAdapterPlusImageProjection` also has `attn_processors`.
    self.encoder_hid_proj = None

    #attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
    attn_procs=_convert_ip_adapter_attn_to_diffusers_single(self,state_dicts,variant,low_cpu_mem_usage)
    self.set_attn_processor(attn_procs)

    # convert IP-Adapter Image Projection layers to diffusers
    image_projection_layers = []
    for state_dict in state_dicts:
        image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(
            state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
        )
        image_projection_layers.append(image_projection_layer)

    self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
    self.config.encoder_hid_dim_type = "ip_image_proj"

    self.to(dtype=self.dtype, device=self.device)


#https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/unet.py#L733

def _convert_ip_adapter_attn_to_diffusers_single(self:UNet2DConditionLoadersMixin, state_dicts, variant:str, low_cpu_mem_usage=False):

    if low_cpu_mem_usage:
        if is_accelerate_available():
            from accelerate import init_empty_weights

        else:
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

    if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
            " `low_cpu_mem_usage=False`."
        )

    # set ip-adapter cross-attention processors & load state_dict
    attn_procs = {}
    key_id = 1
    init_context = init_empty_weights if low_cpu_mem_usage else nullcontext
    for name in self.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = self.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(self.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = self.config.block_out_channels[block_id]

        if cross_attention_dim is None or "motion_modules" in name:
            attn_processor_class = self.attn_processors[name].__class__
            attn_procs[name] = attn_processor_class()
        else:
            if variant=="key":
                attn_processor_class=IPAdapterAttnProcessorKey
            elif variant=="value":
                attn_processor_class=IPAdapterAttnProcessorValue
            num_image_text_embeds = []
            for state_dict in state_dicts:
                if "proj.weight" in state_dict["image_proj"]:
                    # IP-Adapter
                    num_image_text_embeds += [4]
                elif "proj.3.weight" in state_dict["image_proj"]:
                    # IP-Adapter Full Face
                    num_image_text_embeds += [257]  # 256 CLIP tokens + 1 CLS token
                elif "perceiver_resampler.proj_in.weight" in state_dict["image_proj"]:
                    # IP-Adapter Face ID Plus
                    num_image_text_embeds += [4]
                elif "norm.weight" in state_dict["image_proj"]:
                    # IP-Adapter Face ID
                    num_image_text_embeds += [4]
                else:
                    # IP-Adapter Plus
                    num_image_text_embeds += [state_dict["image_proj"]["latents"].shape[1]]

            with init_context():
                attn_procs[name] = attn_processor_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=num_image_text_embeds,
                )

            value_dict = {}
            for i, state_dict in enumerate(state_dicts):
                value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

            if not low_cpu_mem_usage:
                attn_procs[name].load_state_dict(value_dict)
            else:
                device = next(iter(value_dict.values())).device
                dtype = next(iter(value_dict.values())).dtype
                load_model_dict_into_meta(attn_procs[name], value_dict, device=device, dtype=dtype)

            key_id += 2

    return attn_procs