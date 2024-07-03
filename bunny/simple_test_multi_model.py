import torch
import transformers
from model.builder import load_pretrained_model
from util.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
import warnings
from conversation import conv_templates, SeparatorStyle
from util.mm_utils import tokenizer_image_token
from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from util.data_aug import crop

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
# disable some warnings
# transformers.logging.set_verbosity_error()
# transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
torch.set_default_device('cuda')  # or 'cuda'
conv_mode = 'qwen-chat'
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path='checkpoints/checkpoints-qwen2-new/bunny-qwen2-formalgeov2-visionInstructTuning-qa-structure_only-sft100/checkpoint-4300', 
                                                                       vision_encoder_path='checkpoints/checkpoints-qwen2-new/bunny-qwen2-formalgeov2-visionInstructTuning-qa-structure_only-sft100/checkpoint-4300', 
                                                                       model_base=None, 
                                                                       model_name='bunny-qwen2',
                                                                       model_type='qwen2',
                                                                       device_map='auto',
                                                                       device='cuda')

'''
conv_mode = 'llama'
model_path = 'checkpoints/checkpoints-llama/bunny-llama-8B-caption-model-tune-all-sft4e'
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=model_path, 
                                                                       model_base=None, 
                                                                       model_name='bunny-llama',
                                                                       model_type='llama',
                                                                       vision_encoder_path=model_path
                                                                       )
'''
# set up conversation templates
conv = conv_templates[conv_mode].copy()
# text prompt
# problem_text_en = 'As shown in the diagram, KL=NL, NM=ML, \u2220JLK=25\u00b0, \u2220KLN=18\u00b0, \u2220NKJ=130\u00b0, \u2220NLM=20\u00b0. Find the measure of \u2220MNL.'
qs = f'<image>\nBased on the image, predict the construction_cdl and image_cdl.'
# qs = '你是谁？'
# text_qs = "As shown in the diagram, \u2220ACB=70\u00b0, the center of \u2299O is O. Find the measure of \u2220BAO."
# qs = f'Based on the image, first predict the construction_cdl. Then, based on the problem_text, predict the text_cdl and image_cdl, and parse out the goal_cdl. Finally, provide the predicted theorem_seqs and the problem answer.\nProblem_text is: {text_qs}'
# qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
print(f'Prompt is\n{prompt}')


input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
#image = Image.open('data/images/geoqa_plus/49.png').convert('RGB')
image = Image.open('data/formalgeo7k/formalgeo7k_v2/diagrams/1.png').convert('RGB')
image = crop(image)
if model.config.image_aspect_ratio == 'pad':
    # print('using pad')
    image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
       
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

with torch.inference_mode():
    # generate
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda:0', non_blocking=True),
        max_new_tokens=1000,
        use_cache=False)[0]
# print(f'output id is{output_ids}')
res = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
print(f'Respones is:\n{res}')