import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np

# set device
device = 'cuda'  # or cpu
torch.set_default_device(device)

# create model
model = AutoModelForCausalLM.from_pretrained(
    # 'checkpoints/publish/GeoReasoner',
    'NaughtyDog97/FormalEnhencedGPS',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    # 'checkpoints/publish/GeoReasoner',
    'NaughtyDog97/FormalEnhencedGPS',
    trust_remote_code=True)

# text prompt
img_path = 'data/formalgeo7k/formalgeo7k_v2/diagrams/4927.png'
qs = 'As shown in the diagram, AE/AB=1/4, M is the midpoint of segment AC, BE is parallel to CP, EA is parallel to CP. Find the ratio of the length of line BC to the length of line CD.'
prompt = f'Using the provided geometric image and question, first predict the construction_cdl and image_cdl. Then, give a detailed step-by-step solution.\nThe question is:\n{qs}'
text = f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device)

# image, sample images can be found in images folder
image = Image.open(img_path).convert('RGB')

image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)
# generate
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=1,
        max_new_tokens=3500,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=None,
        use_cache=True
    )[0]

print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())
