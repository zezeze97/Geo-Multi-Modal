import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, get_model_name_from_path
from bunny.util.data_aug import crop
from PIL import Image
import math
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
                    
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                           args.model_type, vision_encoder_path=args.vision_encoder_path)
    
    if args.cascade_mode:
        recong_model_path = os.path.expanduser(args.recong_model_path)
        recong_model_name = get_model_name_from_path(recong_model_path)
        recong_tokenizer, recong_model, recong_image_processor, recong_context_len = load_pretrained_model(recong_model_path, args.recong_model_base, recong_model_name,
                                                                                                           args.recong_model_type, vision_encoder_path=args.recong_model_vision_encoder_path)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        if args.cascade_mode:
            # 首先使用识别模型识别出consCDL和imgCDL
            # 目前只使用qwen2作为识别模型
            assert args.recong_conv_mode == 'qwen-chat'
            # 识别的时候使用先识别，再矫正指令
            recong_qs = DEFAULT_IMAGE_TOKEN + '\n' +'Based on the image, first predict the construction_cdl and image_cdl and calibrate it.'
            recong_conv = conv_templates[args.recong_conv_mode].copy()
            recong_conv.append_message(recong_conv.roles[0], recong_qs)
            recong_conv.append_message(recong_conv.roles[1], None)
            recong_prompt = recong_conv.get_prompt()
            
            recong_input_ids = tokenizer_image_token(recong_prompt, recong_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            if args.crop:
                image = crop(image)
            if recong_model.config.image_aspect_ratio == 'pad':
                image = expand2square(image, (255, 255, 255))
            
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
                    num_beams=args.num_beams,
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
            # print(f'recong_outputs is {[recong_outputs]}')
            recong_outputs = recong_outputs.strip()
            if recong_outputs.endswith(recong_stop_str):
                recong_outputs = recong_outputs[:-len(recong_stop_str)]
            recong_outputs = recong_outputs.strip()
            
            
            cdl_dict = parse_cdl(recong_outputs)
            pred_cons_cdl = ''
            pred_image_cdl = ''
            if 'construction_cdl' in cdl_dict.keys():
                pred_cons_cdl = cdl_dict['construction_cdl']
            if 'image_cdl' in cdl_dict.keys():
                pred_image_cdl = cdl_dict['image_cdl']
            
            
            # 完成识别后开始推理
            if args.process_meta_qs_mode == 'Q+PredCDL2Ans':
                # Q + [x consCDL + imageCDL] => A
                qs = f'Using the provided geometric image, construction_cdl, image_cdl, and question, give a detailed step-by-step solution. Note that there may be minor errors in the construction_cdl and image_cdl.\nThe construction_cdl is:\n{pred_cons_cdl}\nThe image_cdl is:\n{pred_image_cdl}\nThe question is:\n{qs}'
            elif args.process_meta_qs_mode == 'Q+PredCDL2CalibrateCDLandAns':
                # Q + [x consCDL + imageCDL] => [consCDL + imageCDL] + A
                qs = f'Using the provided geometric image and the possibly erroneous construction_cdl and image_cdl, first calibrate the construction_cdl and image_cdl, then give a detailed step-by-step solution to the question.\nThe initial construction_cdl is:\n{pred_cons_cdl}\nThe initial image_cdl is:\n{pred_image_cdl}\nThe question is:\n{qs}'
            cur_prompt = qs
            if args.conv_mode != 'caption':
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.lstrip('\n')

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # print(f'Prompt is: {prompt}')  
            else:
                conv = conv_templates[args.conv_mode].copy()
                prompt = DEFAULT_IMAGE_TOKEN + conv.sep
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p,
                    top_k=None,
                    num_beams=args.num_beams,
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

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "recong_consCDL": pred_cons_cdl,
                                    "recong_imgCDL": pred_image_cdl,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
            
        else:
            cur_prompt = qs
            if args.conv_mode != 'caption':
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.lstrip('\n')

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # print(f'Prompt is: {prompt}')  
            else:
                conv = conv_templates[args.conv_mode].copy()
                prompt = DEFAULT_IMAGE_TOKEN + conv.sep
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            if args.crop:
                image = crop(image)
            # debug when image_aspect_ratio == 'pad'
            if model.config.image_aspect_ratio == 'pad':
                # print('using pad')
                # image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                image = expand2square(image, (255, 255, 255))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p,
                    top_k=None,
                    num_beams=args.num_beams,
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

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument('--vision_encoder_path', type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--crop", action='store_true')
    parser.add_argument("--cascade-mode", action='store_true')
    parser.add_argument("--recong-model-path", type=str, default=None)
    parser.add_argument("--recong-model-base", type=str, default=None)
    parser.add_argument("--recong-model-type", type=str, default=None)
    parser.add_argument('--recong-model-vision_encoder_path', type=str, default=None)
    parser.add_argument("--recong-conv-mode", type=str, default=None)
    parser.add_argument("--process-meta-qs-mode", type=str, default=None)
    
    args = parser.parse_args()

    eval_model(args)
