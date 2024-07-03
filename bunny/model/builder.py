import os
import warnings
import torch
import glob
from safetensors import safe_open
import shutil

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, logging

# logging.set_verbosity_error()

from bunny.model import *


def load_pretrained_model(model_path, model_base, model_name, model_type, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", vision_encoder_path=None, **kwargs):
    if model_type not in {'phi-1.5', 'phi-2', 'stablelm-2', 'mistral', 'llama', 'qwen2', 'mixtral', 'yi1.5'}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True 
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )

    else:
        kwargs['torch_dtype'] = torch.float16

    # Load Bunny model
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn(
            'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

        print('Loading Bunny from base model...')
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'mistral':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'llama':
            # debug in 4bit quant
            if hasattr(lora_cfg_pretrained, 'quantization_config'):
                del lora_cfg_pretrained.quantization_config
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'yi1.5':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, 
                                                            config=lora_cfg_pretrained, **kwargs)
            
            
        elif model_type == 'qwen2':
            # tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            # 由于有可能改了tokenizer, 直接从model_path里面读取
            tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
            # lora_cfg由于改了vocab导致和预训练模型不匹配
            origin_cfg = AutoConfig.from_pretrained(model_base)
            real_vocab_size = lora_cfg_pretrained.vocab_size
            lora_cfg_pretrained.vocab_size = origin_cfg.vocab_size
            model = BunnyQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, 
                                                          config=lora_cfg_pretrained, **kwargs) 
            lora_cfg_pretrained.vocab_size = real_vocab_size
            # 替换原有的lm_head为一个新的线性层，匹配新的词汇表大小
            vocab_size = len(tokenizer)
            # print(f'new vocab_size is {vocab_size}')
            model.resize_token_embeddings(vocab_size)
            
        elif model_type == 'mixtral':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            # debug in 4bit quant
            del lora_cfg_pretrained.quantization_config 
            lora_cfg_pretrained.output_router_logits = False
            model = BunnyMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=lora_cfg_pretrained, **kwargs)
        
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Bunny weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')

            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                   non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        # this may be mm projector only
        print('Loading Bunny from base model...')

        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
        elif model_type == 'mistral':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
        elif model_type == 'llama':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
            tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})
        elif model_type == 'yi1.5':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
        elif model_type == 'qwen2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
        elif model_type == 'mixtral':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
            
        
        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'mistral':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
            model = BunnyMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'llama':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
            tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})
            model = BunnyLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'yi1.5':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'qwen2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side="right")
            
            model = BunnyQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'mixtral':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
            model = BunnyMixtralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
 
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
        if vision_encoder_path is not None:
            ckpt_path = vision_encoder_path
            state_dict = {}
            # find safetensors
            paths = glob.glob(os.path.join(ckpt_path, '*.safetensors'))
            paths += glob.glob(os.path.join(ckpt_path, 'non_lora_trainables.bin'))
            print(f'loading vision tower from {paths}')
            for path in paths:
                if 'safetensors' in path:
                    with safe_open(path, framework='pt', device='cpu') as f:
                        for k in f.keys():
                            if 'model.vision_tower' in k:
                                new_k = k.replace('model.vision_tower.', '')
                                state_dict[new_k] = f.get_tensor(k)
                elif 'non_lora_trainables.bin' in path:
                    ckpt = torch.load(path)
                    for k in ckpt.keys():
                        if 'base_model.model.model.vision_tower' in k:
                            weights = ckpt[k]
                            new_k = k.replace('base_model.model.model.vision_tower.', '')
                            state_dict[new_k] = weights 
            vision_tower.load_state_dict(state_dict, strict=True)
        
        
        
            
        
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model, image_processor, context_len
