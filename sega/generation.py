import torch
import ultralytics
import os
from semdiffusers import SemanticEditPipeline
torch.cuda.empty_cache()

device='cuda'
pipe = SemanticEditPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4").to(device)
gen = torch.Generator(device=device)

save_path = "images/python_generated/"

prompt = "a portrait photo of a doctor"

seed = 1
num_images = 5

idx, num_generated = 0, 0
while num_generated < num_images:
    gen.manual_seed(idx)
    params = {'guidance_scale': 7.5,
              'seed': idx,
              'prompt': prompt,
              'negative_prompt': "deformed, blurry",
              'num_images_per_prompt': 1
             }
    out = pipe(**params, generator=gen)
    image = out.images[0]
    image.save(f"{save_path}/image{idx}.png")

#     # if there is a face
#     model = YOLO('yolov8n-face.pt')
#     results = model(f'{original_image_name}')
#     for result in results:
#         boxes = result.boxes  # Boxes object for bbox outputs
#         masks = result.masks  # Masks object for segmentation masks outputs
#     box_amount = len(boxes)
    
#     for i in range(box_amount):
#         if boxes[i].conf > 0.5: # confirm face confidence > 0.5
#             im_new = im.crop((left[i], top[i], right[i], bottom[i]))
#             current_image_name = os.path.basename(os.path.normpath(original_image_name))
#             if current_image_name.endswith('.jpg'):
#                 current_image_name = current_image_name.removesuffix('.jpg')

#             im_new.save(f'{cropped_image_save_path}{current_image_name}_{i}.jpg')
    
    
    num_generated += 1
    idx += 1
