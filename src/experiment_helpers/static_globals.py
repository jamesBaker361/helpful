#metrics:
PROMPT_SIMILARITY="prompt_similarity"
IDENTITY_CONSISTENCY="identity_consistency"
TARGET_SIMILARITY="target_similarity"
AESTHETIC_SCORE="aesthetic_score"
IMAGE_REWARD="image_reward"
BLIP_TARGET_CAPTION_SIMILARITY="blip_target_caption_similarity"
BLIP_PROMPT_CAPTION_SIMILARITY="blip_prompt_caption_similarity"
BLIP_TARGET_SIMILARITY="blip_target_similarity"
BLIP_IDENTITY_CONSISTENCY="blip_identity_consistency"
VIT_TARGET_SIMILARITY="vit_target_similarity"
VIT_IDENTITY_CONSISTENCY="vit_identity_consistency"
VIT_STYLE_TARGET_SIMILARITY="style_target_similarity"
VIT_STYLE_CONSISTENCY="style_consistency"
VIT_CONTENT_TARGET_SIMILARITY="content_target_similarity"
VIT_CONTENT_CONSISTENCY="content_consistency"
FACE_TARGET_SIMILARITY="face_target_consistency"
FACE_CONSISTENCY="face_consistency"
FASHION_CONSISTENCY="fashion_consistency"
FASHION_SIMILARITY="fashion_similarity"

METRIC_LIST=[PROMPT_SIMILARITY, IDENTITY_CONSISTENCY, TARGET_SIMILARITY, AESTHETIC_SCORE, IMAGE_REWARD,
    BLIP_TARGET_CAPTION_SIMILARITY,BLIP_PROMPT_CAPTION_SIMILARITY,
    #BLIP_TARGET_SIMILARITY,BLIP_IDENTITY_CONSISTENCY,
    VIT_TARGET_SIMILARITY,VIT_IDENTITY_CONSISTENCY,
    VIT_STYLE_TARGET_SIMILARITY, VIT_STYLE_CONSISTENCY,
    VIT_CONTENT_TARGET_SIMILARITY,VIT_CONTENT_CONSISTENCY,
    FACE_TARGET_SIMILARITY, FACE_CONSISTENCY,FASHION_CONSISTENCY,FASHION_SIMILARITY]

FACE_METRIC_LIST=[FACE_TARGET_SIMILARITY, FACE_CONSISTENCY]