import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np

def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

# set device
device = 'cuda'  # or cpu
torch.set_default_device(device)

# create model
model = AutoModelForCausalLM.from_pretrained(
    # 'checkpoints/publish/GeoFormalizer',
    'NaughtyDog97/GeoFormalizer',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    # 'checkpoints/publish/GeoFormalizer',
    'NaughtyDog97/GeoFormalizer',
    use_fast=True,
    padding_side="right",
    trust_remote_code=True)

# text prompt
img_path = 'data/formalgeo7k/formalgeo7k_v2/diagrams/4927.png'
prompt = 'Based on the image, first describe what you see in the figure, then predict the construction_cdl and image_cdl and calibrate it.'
text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer_image_token(text, tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()

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


respones = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
print(respones)