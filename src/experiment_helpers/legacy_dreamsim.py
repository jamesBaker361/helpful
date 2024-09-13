from dreamsim import dreamsim
from peft_legacy import PeftModel, LoraConfig, get_peft_model
from dreamsim.model import download_weights, dreamsim_args, ViTModel,PerceptualModel,ViTConfig
from torchvision import transforms
import os


def dreamsim(pretrained: bool = True, device="cuda", cache_dir="/scratch/jlb638/dreamsim", normalize_embeds: bool = True,
             dreamsim_type: str = "ensemble"):
    """ Initializes the DreamSim model. When first called, downloads/caches model weights for future use.

    :param pretrained: If True, downloads and loads DreamSim weights.
    :param cache_dir: Location for downloaded weights.
    :param device: Device for model.
    :param normalize_embeds: If True, normalizes embeddings (i.e. divides by norm and subtracts mean).
    :param dreamsim_type: The type of dreamsim model to use. The default is "ensemble" (default and best-performing)
                          which concatenates dino_vitb16, clip_vitb16, and open_clip_vitb16 embeddings. Other options
                          are "dino_vitb16", "clip_vitb32", and "open_clip_vitb32" which are finetuned single models.
    :return:
        - PerceptualModel with DreamSim settings and weights.
        - Preprocessing function that converts a PIL image and to a (1, 3, 224, 224) tensor with values [0-1].
    """
    # Get model settings and weights
    download_weights(cache_dir=cache_dir, dreamsim_type=dreamsim_type)

    # initialize PerceptualModel and load weights
    model_list = dreamsim_args['model_config'][dreamsim_type]['model_type'].split(",")
    ours_model = PerceptualModel(**dreamsim_args['model_config'][dreamsim_type], device=device, load_dir=cache_dir,
                                 normalize_embeds=normalize_embeds)
    for extractor in ours_model.extractor_list:
        lora_config = LoraConfig(**dreamsim_args['lora_config'])
        model = get_peft_model(ViTModel(extractor.model, ViTConfig()), lora_config)
        extractor.model = model

    tag = "" if dreamsim_type == "ensemble" else "single_"
    if pretrained:
        for extractor, name in zip(ours_model.extractor_list, model_list):
            load_dir = os.path.join(cache_dir, f"{name}_{tag}lora")
            extractor.model = PeftModel.from_pretrained(extractor.model, load_dir).to(device)
            extractor.model.eval().requires_grad_(False)

    ours_model.eval().requires_grad_(False)

    # Define preprocessing function
    t = transforms.Compose([
        transforms.Resize((dreamsim_args['img_size'], dreamsim_args['img_size']),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    def preprocess(pil_img):
        pil_img = pil_img.convert('RGB')
        return t(pil_img).unsqueeze(0)

    return ours_model, preprocess