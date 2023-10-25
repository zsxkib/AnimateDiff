#!/bin/bash

# Run with `./download_bashscripts/cog_download_models.sh`
# A bash script that runs some of the `run` commands in the cog.yaml
# done to quickly get a prototype up and running - boot times would include downloading otherwise

# Clone the repo if it doesn't exist
REPO_DIR="models/StableDiffusion/stable-diffusion-v1-5"
if [ ! -d "$REPO_DIR" ]; then
    git clone --branch fp16 https://huggingface.co/runwayml/stable-diffusion-v1-5 $REPO_DIR
else
    echo "Repository already exists at $REPO_DIR, skipping clone."
fi

# Ensure directories exist
mkdir -p models/Motion_Module || true
mkdir -p models/MotionLoRA || true
mkdir -p models/DreamBooth_LoRA || true

# Download Motion_Module models
wget -O models/Motion_Module/mm_sd_v14.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt || true
wget -O models/Motion_Module/mm_sd_v15.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt || true
wget -O models/Motion_Module/mm_sd_v15_v2.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt || true
bash download_bashscripts/0-MotionModule.sh || true

# Download MotionLoRA models
wget -O models/MotionLoRA/v2_lora_ZoomIn.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt || true
wget -O models/MotionLoRA/v2_lora_ZoomOut.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomOut.ckpt || true
wget -O models/MotionLoRA/v2_lora_PanLeft.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.ckpt || true
wget -O models/MotionLoRA/v2_lora_PanRight.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.ckpt || true
wget -O models/MotionLoRA/v2_lora_RollingClockwise.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingClockwise.ckpt || true
wget -O models/MotionLoRA/v2_lora_RollingAnticlockwise.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingAnticlockwise.ckpt || true

# NOTE: The original names for these models were PanUp/PanDown, but the download links have been updated to TiltUp/TiltDown.
# To avoid changing the code in predict.py, we're keeping the old names in the code and just updating the download links.
# This way, when the models are downloaded, they will still be saved under the old names and the code in predict.py won't need to be changed.
wget -O models/MotionLoRA/v2_lora_PanDown.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltDown.ckpt || true
wget -O models/MotionLoRA/v2_lora_PanUp.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltUp.ckpt || true

# Download DreamBooth_LoRA models
pget https://civitai.com/api/download/models/78775 models/DreamBooth_LoRA/toonyou_beta3.safetensors || true
pget https://civitai.com/api/download/models/72396 models/DreamBooth_LoRA/lyriel_v16.safetensors || true
pget https://civitai.com/api/download/models/71009 models/DreamBooth_LoRA/rcnzCartoon3d_v10.safetensors || true
pget https://civitai.com/api/download/models/79068 models/DreamBooth_LoRA/majicmixRealistic_v5Preview.safetensors || true
pget https://civitai.com/api/download/models/29460 models/DreamBooth_LoRA/realisticVisionV40_v20Novae.safetensors || true

# Execute remaining download scripts
bash download_bashscripts/1-ToonYou.sh || true
bash download_bashscripts/2-Lyriel.sh || true
bash download_bashscripts/3-RcnzCartoon.sh || true
bash download_bashscripts/4-MajicMix.sh || true
bash download_bashscripts/5-RealisticVision.sh || true
bash download_bashscripts/6-Tusun.sh || true
bash download_bashscripts/7-FilmVelvia.sh || true
bash download_bashscripts/8-GhibliBackground.sh || true
