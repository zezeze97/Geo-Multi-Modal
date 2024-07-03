import torch
import transformers
from model.builder import load_pretrained_model
from util.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
import warnings

# disable some warnings
# transformers.logging.set_verbosity_error()
# transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
# set device
torch.set_default_device('cuda')  # or 'cuda'
'''
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path='/research/zhangzr/Bunny/checkpoints-phi-2/g-llava-data/bunny-phi-2-epoch-8', 
                                                                       model_base=None, 
                                                                       model_name='bunny-phi-2',
                                                                       model_type='phi-2')
'''
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path='checkpoints/checkpoints-yi1.5/bunny-yi1.5-caption-syth-v3-tune-vision-only', 
                                                                       model_base=None, 
                                                                       model_name='bunny-yi1.5',
                                                                       model_type='yi1.5',
                                                                       vision_encoder_path='checkpoints/checkpoints-yi1.5/bunny-yi1.5-caption-syth-v3-tune-vision-only')


# text prompt
text = "<image>\n"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)
# print(f'input_id is {input_ids}')
# image, sample images can be found in https://huggingface.co/BAAI/Bunny-v1_0-3B/tree/main/images
image = Image.open('data/images/geoqa_plus/8.png').convert('RGB')
if model.config.image_aspect_ratio == 'pad':
    image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

# generate
output_ids = model.generate(
    input_ids,
    images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True),
    max_new_tokens=1024,
    use_cache=False)[0]
# print(f'output_ids is: {output_ids[input_ids.shape[1]:]}')
print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())