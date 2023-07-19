import torch
torch.cuda.empty_cache()
from diffusers import StableDiffusionPipeline
import os

img_directory = "images/generated"
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

#os.environ['TRANSFORMERS_CACHE'] = '~/fs/nexus-scratch/dyang5/cache/'
#os.environ['HF_DATASETS_CACHE'] = '~/fs/nexus-scratch/dyang5/cache/'
#os.environ['HF_HOME'] = '~/fs/nexus-scratch/dyang5/cache/'

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

negs="Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, mutated hands and fingers, out of frame"


all_images = []
num_trials = 4
img_per_prompt = 15
prompts = ["doctor", "group of doctors", "firefighter", "group of firefighters"]
concepts = ["doctor_single", "doctor_group", "firefighter_single", "firefighter_group"]

for i in range(num_trials):
    modifier = ""
    if "single" in concepts[i]:
        modifier = "portrait picture of a "
    elif "group" in concepts[i]:
        modifier = "picture of a "
        
    prompt = "high quality " + modifier + prompts[i]

    generated_images = pipe(prompt, negative_prompt = negs, num_images_per_prompt = img_per_prompt).images
    all_images += generated_images
    
for i in range(len(all_images)):
    image = all_images[i]
    index = int(i/img_per_prompt)
    concept = concepts[int(i/img_per_prompt)]
    image.save(f'{img_directory}{concept}{i}.jpg')
    