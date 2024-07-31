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
import re

def parse_cdl(input_string):
    # 使用正则表达式查找各个部分
    patterns = {
        'construction_cdl': r'(?:The )?(?:calibrate )?construction_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl is:|\n(?:The )?(?:calibrate )?\w+_cdl:|\nSolution is:|\Z)',
        'image_cdl': r'(?:The )?(?:calibrate )?image_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl is:|\n(?:The )?(?:calibrate )?\w+_cdl:|\nSolution is:|\Z)',
        'text_cdl': r'(?:The )?text_cdl(?: is)?:\n(.*?)(?=\n(?:The )?\w+_cdl is:|\n(?:The )?\w+_cdl:|\nSolution is:|\Z)',
        'goal_cdl': r'(?:The )?goal_cdl(?: is)?:\n(.*?)(?=\n(?:The )?\w+_cdl is:|\n(?:The )?\w+_cdl:|\nSolution is:|\Z)'
    }
    
    results = {}
    
    # 优先匹配包含"calibrate"的版本
    for key, pattern in patterns.items():
        pattern = pattern.replace("(?:calibrate )?", "(?:calibrate )")
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            results[key] = match.group(1).strip()
        else:
            # 如果未找到包含"calibrate"的版本，尝试匹配不含"calibrate"的版本
            pattern = pattern.replace("(?:calibrate )", "(?:calibrate )?")
            match = re.search(pattern, input_string, re.DOTALL)
            if match:
                results[key] = match.group(1).strip()
    
    return results

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



def cascade_prediction(recong_tokenizer, recong_model, recong_image_processor, recong_conv_mode,
                       tokenizer, model, image_processor, conv_mode, img_path, qs, qa_mode, crop_image, consCDL=None, imgCDL=None, temperature=0, num_beams=1, top_p=None):
    '''
    qs是原始问题, 不包含任何格式
    qa_mode: q2ans, q2cdl_ans, q_cdl2ans, q_predcdl2ans, q_predcdl2cdl_ans
    
    '''
    pred_consCDL = ''
    pred_imgCDL = ''
    image = Image.open(img_path).convert('RGB')
    if crop_image:
        image = crop(image)
    assert recong_model.config.image_aspect_ratio == model.config.image_aspect_ratio
    if model.config.image_aspect_ratio == 'pad':
        image = expand2square(image, (255, 255, 255))
    # 需要模型预测CDL
    if qa_mode in ['q_predcdl2ans', 'q_predcdl2cdl_ans']:
        assert recong_conv_mode == 'qwen-chat'
        # 识别的时候使用先识别，再矫正指令
        recong_qs = DEFAULT_IMAGE_TOKEN + '\n' +'Based on the image, first describe what you see in the figure, then predict the construction_cdl and image_cdl and calibrate it.'
        recong_conv = conv_templates[recong_conv_mode].copy()
        recong_conv.append_message(recong_conv.roles[0], recong_qs)
        recong_conv.append_message(recong_conv.roles[1], None)
        recong_prompt = recong_conv.get_prompt()
        recong_input_ids = tokenizer_image_token(recong_prompt, recong_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        recong_image_tensor = recong_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        recong_stop_str = recong_conv.sep if recong_conv.sep_style != SeparatorStyle.TWO else recong_conv.sep2
        # 进行结构识别
        with torch.inference_mode():
            recong_output_ids = recong_model.generate(
                recong_input_ids,
                images=recong_image_tensor.unsqueeze(0).to(dtype=recong_model.dtype, device='cuda', non_blocking=True),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=3500, # 2048
                eos_token_id=recong_tokenizer.eos_token_id,
                repetition_penalty=None,
                use_cache=True
                )
        recong_input_token_len = recong_input_ids.shape[1]
        n_diff_input_output = (recong_input_ids != recong_output_ids[:, :recong_input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        recong_outputs = recong_tokenizer.batch_decode(recong_output_ids[:, recong_input_token_len:], skip_special_tokens=True)[0]
        print(f'Recognition CDL is:\n{recong_outputs}')
        recong_outputs = recong_outputs.strip()
        if recong_outputs.endswith(recong_stop_str):
            recong_outputs = recong_outputs[:-len(recong_stop_str)]
        recong_outputs = recong_outputs.strip()
        cdl_dict = parse_cdl(recong_outputs)
        if 'construction_cdl' in cdl_dict.keys():
            pred_consCDL = cdl_dict['construction_cdl']
        if 'image_cdl' in cdl_dict.keys():
            pred_imgCDL = cdl_dict['image_cdl']
    
    # 开始模型推理
    # 构建问题
    qs_map = {'q2ans': f'Using the provided geometric image and question, give a detailed step-by-step solution.\nThe question is:\n{qs}',
     'q2cdl_ans': f'Using the provided geometric image and question, first predict the construction_cdl and image_cdl. Then, give a detailed step-by-step solution.\nThe question is:\n{qs}',
     'q_cdl2ans': f'Using the provided geometric image, construction_cdl, image_cdl, and question, give a detailed step-by-step solution.\nThe construction_cdl is:\n{consCDL}\nThe image_cdl is:\n{imgCDL}\nThe question is:\n{qs}',
     'q_predcdl2ans': f'Using the provided geometric image, construction_cdl, image_cdl, and question, give a detailed step-by-step solution. Note that there may be minor errors in the construction_cdl and image_cdl.\nThe construction_cdl is:\n{pred_consCDL}\nThe image_cdl is:\n{pred_imgCDL}\nThe question is:\n{qs}',
     'q_predcdl2cdl_ans': f'Using the provided geometric image and the possibly erroneous construction_cdl and image_cdl, first calibrate the construction_cdl and image_cdl, then give a detailed step-by-step solution to the question.\nThe initial construction_cdl is:\n{pred_consCDL}\nThe initial image_cdl is:\n{pred_imgCDL}\nThe question is:\n{qs}'}
    
    new_qs = qs_map[qa_mode]
    new_qs = DEFAULT_IMAGE_TOKEN + '\n' + new_qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], new_qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True),
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            top_k=None,
            num_beams=num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=3500, # 2048
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=None,
            use_cache=True
            )
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(f"CotAns is:\n{outputs}")
    
            
            
if __name__ == '__main__':
    img_path = 'data/formalgeo7k/formalgeo7k_v2/diagrams/4927.png'
    qs = 'As shown in the diagram, AE/AB=1/4, M is the midpoint of segment AC, BE is parallel to CP, EA is parallel to CP. Find the ratio of the length of line BC to the length of line CD.'
    qa_mode = 'q_predcdl2cdl_ans'
    # 加载识别模型和推理模型
    
    # disable some warnings
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    warnings.filterwarnings('ignore')
    # set device
    torch.set_default_device('cuda')  # or 'cuda'
    recong_conv_mode = 'qwen-chat'
    recong_tokenizer, recong_model, recong_image_processor, recong_context_len = load_pretrained_model(model_path='checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_calibrate_structure_only-sft4-add05/merged', 
                                                                        vision_encoder_path='checkpoints/checkpoints-qwen2/bunny-lora-qwen2-qa-FormalGeoV2Aug10Times_calibrate_structure_only-sft4-add05/merged', 
                                                                        model_base=None, 
                                                                        model_name='bunny-qwen2',
                                                                        model_type='qwen2',
                                                                        device_map='auto',
                                                                        device='cuda')


    conv_mode = 'yi-chat'
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path='checkpoints/checkpoints-yi1.5/bunny-yi1.5-9B-Chat-FormalGeoCoT-VisionPretrained-sft2e-v12', 
                                                                        vision_encoder_path='checkpoints/checkpoints-yi1.5/bunny-yi1.5-9B-Chat-FormalGeoCoT-VisionPretrained-sft2e-v12', 
                                                                        model_base=None, 
                                                                        model_name='bunny-yi1.5',
                                                                        model_type='yi1.5',
                                                                        device_map='auto',
                                                                        device='cuda')
    
    
    cascade_prediction(recong_tokenizer, recong_model, recong_image_processor, recong_conv_mode,
                       tokenizer, model, image_processor, conv_mode, img_path, qs, qa_mode, crop_image=True, consCDL=None, imgCDL=None, temperature=0, num_beams=1, top_p=None)