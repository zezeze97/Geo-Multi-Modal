import torch
import transformers
from model.builder import load_pretrained_model
from util.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
import warnings
from conversation import conv_templates, SeparatorStyle
from util.mm_utils import tokenizer_image_token
from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

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
conv_mode = 'yi'
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path='checkpoints/checkpoints-yi1.5/bunny-yi1.5-9B-formalgeo-multiturn-merged', 
                                                                       vision_encoder_path='checkpoints/checkpoints-yi1.5/bunny-yi1.5-9B-formalgeo-multiturn-merged', 
                                                                       model_base=None, 
                                                                       model_name='bunny-yi1.5',
                                                                       model_type='yi1.5')

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

problem_text_en = "As shown in the diagram, triangle RST is congruent to triangle XYZ, TR=x+21, ZX=2*x-14, ∠TRS=4*y-10°, ∠ZXY=3*y+5°. Find the value of y."
qs_lst = ["<image>\nBased on the image, predict the construction_cdl.", 
          f"Then, based on the problem text, predict the text_cdl and image_cdl and parse out the goal_cdl.\nProblem text is: {problem_text_en}",
          "Finally, provide the predicted theorem_seqs and the problem answer."]

# set up conversation templates
conv = conv_templates[conv_mode].copy()


image = Image.open('data/formalgeo7k/formalgeo7k_v1/diagrams/1.png').convert('RGB')
if model.config.image_aspect_ratio == 'pad':
    image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

for qs in qs_lst:
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(f'Prompt is\n{prompt}')
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    # generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True),
            max_new_tokens=1000,
            use_cache=False)[0]
    # print(f'output id is{output_ids}')
    res = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    conv.messages[-1] = [conv.roles[1], res]
    print(f'Respones is:\n{res}')