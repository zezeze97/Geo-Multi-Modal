import os
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Optional

import torch
import torch.distributed as dist
import transformers

import shutil
from bunny.train.bunny_trainer import BunnyTrainer

from bunny import conversation as conversation_lib
from bunny.model import *
from bunny.util.data_utils import make_supervised_data_module, DataArguments
from safetensors import safe_open
import glob
import evaluate
import numpy as np
from eval.evaluation_formalgeo.utils import getConsCdlAcc

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    version: Optional[str] = field(default=None)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained_local_path: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    use_s2: bool = field(default=False)
    add_formal_tokens: bool = field(default=False)
    use_formalgeo_vocab_only: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_vision_tower: bool=field(default=True)
    tune_vision_tower: bool=field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    force_tune_embedding: bool = False
    


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.util.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    global local_rank
    # debug?
    # torch.set_num_threads(1)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # debug
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    assert model_args.vision_tower is not None
    if model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    elif model_args.model_type == 'stablelm-2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )
    elif model_args.model_type == 'mistral':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
    elif model_args.model_type == 'mixtral':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
    elif model_args.model_type == 'llama':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
        tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})
    elif model_args.model_type == 'yi1.5':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
    elif model_args.model_type == 'qwen2':
        if model_args.model_name_or_path is None or model_args.use_formalgeo_vocab_only:
        
            tokenizer = Qwen2Tokenizer(vocab_file='data/formalgeo7k/formalgeo7k_v2/vocab/vocab.json',
                                    merges_file='data/formalgeo7k/formalgeo7k_v2/vocab/merges.txt',
                                    unk_token=None,
                                    model_max_length=training_args.model_max_length)
        
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=True,
                trust_remote_code=True
            )
    
    # support formalgeo
    if model_args.add_formal_tokens or model_args.use_formalgeo_vocab_only:
        special_tokens = ["construction_cdl", "image_cdl", "Shape", "LengthOfLine", "MeasureOfAngle", "Equal", "PerpendicularBetweenLine", "Collinear", "Cocircular", "ParallelBetweenLine"]
        # special_tokens_dict = {'additional_special_tokens': special_tokens}
        # tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.add_tokens(['\n'] + special_tokens)
        rank0_print('adding formal tokens....')
        
    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
  
    if model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2':
        model = BunnyPhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        )
        
    elif model_args.model_type == 'stablelm-2':
        model = BunnyStableLMForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'mistral':
        model = BunnyMistralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        )   
    elif model_args.model_type == 'mixtral':
        model = BunnyMixtralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        ) 
        # enable aux loss for router
        model.config.output_router_logits = True
    elif model_args.model_type == 'llama':
        model = BunnyLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        )   
    elif model_args.model_type == 'yi1.5':
        model = BunnyLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        ) 
    elif model_args.model_type == 'qwen2':
        
        
        if model_args.model_name_or_path is None:
            config = BunnyQwen2Config(attention_dropout=0.0,
                                hidden_act='silu',
                                hidden_size=1024,
                                initializer_range=0.02,
                                intermediate_size=2816,
                                max_position_embeddings=2048,
                                max_window_layers=21,
                                model_type='qwen2',
                                num_attention_heads=8,
                                num_hidden_layers=3, # 24
                                num_key_value_heads=8,
                                rms_norm_eps=1e-6,
                                rope_theta=10000.0,
                                sliding_window=2048,
                                tie_word_embeddings= True,
                                torch_dtype = torch.bfloat16,
                                use_cache = True,
                                use_sliding_window= False,
                                vocab_size=len(tokenizer),
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                )
            model = BunnyQwen2ForCausalLM._from_config(config)
        else:
            
            model = BunnyQwen2ForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                            cache_dir=training_args.cache_dir,
                                                            bos_token_id=tokenizer.bos_token_id,
                                                            eos_token_id=tokenizer.eos_token_id,
                                                            **bnb_model_from_pretrained_args
                                                        )
        
        
        # 替换原有的lm_head为一个新的线性层，匹配新的词汇表大小
        # vocab_size = len(tokenizer)
        # rank0_print(f'new vocab_size is {vocab_size}')
        # model.resize_token_embeddings(vocab_size)
        # force tune word embedding
        # model.lm_head.requires_grad_(True)
        # rank0_print(model)
        
        
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")
    
    # 需要对embedding进行重新初始化
    if model_args.add_formal_tokens or model_args.use_formalgeo_vocab_only:
        rank0_print(f'add formal tokens, resize token embeddings')
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["default"]
        
    # 准备vision encoder的ckpt， 针对zero3 debug， 只能初始化一次
    if model_args.vision_tower_pretrained_local_path is not None and local_rank == 0:
        ckpt_path = model_args.vision_tower_pretrained_local_path
        state_dict = {}
        # find safetensors
        paths = glob.glob(os.path.join(ckpt_path, '*.safetensors'))
        paths += glob.glob(os.path.join(ckpt_path, 'non_lora_trainables.bin'))
        # print(f'vision tower loading paths is {paths}')
        for path in paths:
            if 'safetensors' in path:
                with safe_open(path, framework='pt', device='cpu') as f:
                    for k in f.keys():
                        if 'model.vision_tower' in k:
                            new_k = k.replace('model.vision_tower.vision_tower.', '')
                            state_dict[new_k] = f.get_tensor(k)
            elif 'non_lora_trainables.bin' in path:
                ckpt = torch.load(path)
                for k in ckpt.keys():
                    if 'base_model.model.model.vision_tower' in k:
                        weights = ckpt[k]
                        new_k = k.replace('base_model.model.model.vision_tower.vision_tower.', '')
                        state_dict[new_k] = weights
        # save state_dict to temp dir
        
        temp_dir = os.path.join(ckpt_path, 'vision_encoder_ckpt')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        torch.save(state_dict, os.path.join(temp_dir, 'pytorch_model.bin'))
        config = transformers.AutoConfig.from_pretrained(model_args.vision_tower)
        config.save_pretrained(temp_dir)
    # 确保所有节点都等待主节点完成文件保存
    dist.barrier()
    
    
    if model_args.vision_tower_pretrained_local_path is not None:
        # 更改vision encoder的ckpt path
        ckpt_path = model_args.vision_tower_pretrained_local_path
        model_args.vision_tower_pretrained_local_path = os.path.join(ckpt_path, 'vision_encoder_ckpt')
    # 加载模型
    model.get_model().initialize_vision_modules(model_args=model_args)
    
    # 删除临时vision ckpt
    if model_args.vision_tower_pretrained_local_path is not None and local_rank == 0:
        shutil.rmtree(model_args.vision_tower_pretrained_local_path)
    dist.barrier()
            

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
            
    # add freeze vision tower
    model.config.freeze_vision_tower = training_args.freeze_vision_tower
    if training_args.freeze_vision_tower:
        for p in vision_tower.parameters():
            p.requires_grad = False
        vision_tower.eval()
    else:
        for p in vision_tower.parameters():
            p.requires_grad = True
    # add tune vision tower
    model.config.tune_vision_tower = training_args.tune_vision_tower
    if model.config.tune_vision_tower:
        model.requires_grad_(False)
        for p in vision_tower.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        
    if training_args.force_tune_embedding:
        model.get_model().embed_tokens.requires_grad_(True)
        
    
    
    
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_projector_lr = training_args.mm_projector_lr
    
    model.config.use_s2 = model_args.use_s2

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    # 统计所有可优化参数量
    # 初始化参数总量计数器
    trainable_vision_params = 0
    trainable_llm_params = 0
    trainable_mmadapter_params = 0
    total_params = 0

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        # 只计算可训练的参数
        rank0_print(f'parame name is {name}, grad statu is {param.requires_grad}, param_num is {param.numel()}')
        if param.requires_grad:
            if 'vision' in name:
                trainable_vision_params += param.numel()
            elif 'mm_projector' in name:
                trainable_mmadapter_params += param.numel()
            else:
                trainable_llm_params += param.numel()
            # 使用numel方法获取参数中的元素数量，并累加到总数中
            total_params += param.numel()

    rank0_print(f"Total trainable parameters: {total_params / 1_000_000:.3f}M, Vision: {trainable_vision_params / 1_000_000:.3f}M, Adapter: {trainable_mmadapter_params / 1_000_000:.3f}M, LLM: {trainable_llm_params / 1_000_000:.3f}M")

    
    
    # 定义eval函数, 不用的时候注销掉
    # eval_metric = evaluate.load('cer')
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        # print(f'predictions shape: {predictions.shape}, prediction: {predictions[0]}\nlabels shape: {labels.shape}, label: {labels[0]}')
        def mask_predictions_by_instruction_length(labels, predictions):
            # 假设 labels 和 predictions 都是 torch.Tensor 类型
            mask_value = -100  # 根据需要调整这个值
            batch_size, _ = labels.shape
            
            # 创建一个与 predictions 相同形状的 mask，初始值为 1
            mask = np.ones_like(predictions, dtype=bool)
            
            # 遍历每个样本，计算 instruction 长度并更新 mask
            for i in range(batch_size):
                # 计算当前样本的 instruction 长度, 记得加入729个image token占位
                instruction_length = np.where(labels[i] != -100)[0][0] + 727
                # 将 predictions 中对应 instruction 部分的 mask 设置为 False
                mask[i, :instruction_length] = False
            
            # 应用 mask，将需要 mask 的部分设置为 mask_value
            predictions = np.where(mask, predictions, mask_value)
            return predictions
        def process_predictions(predictions, tokenizer):
            eos_token_id = tokenizer.eos_token_id
            pad_token_id = tokenizer.pad_token_id
            
            # Process predictions to stop at eos token
            processed_predictions = []
            for prediction_list in predictions:
                processed_list = []
                for prediction in prediction_list:
                    if prediction == eos_token_id:
                        break
                    processed_list.append(prediction if prediction != -100 else pad_token_id)
                processed_predictions.append(processed_list)
            
            # Decode the processed predictions
            decoded_preds = tokenizer.batch_decode(processed_predictions, skip_special_tokens=True)
            
            return decoded_preds

        predictions = mask_predictions_by_instruction_length(labels, predictions)
        decoded_preds = process_predictions(predictions, tokenizer)
        # 标签通常是用-100标记padding，因此需要恰当地处理它们
        labels = [[label if label != -100 else tokenizer.pad_token_id for label in label_list] for label_list in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 计算分数
        score = eval_metric.compute(predictions=decoded_preds, references= decoded_labels)
        if local_rank == 0:
            num_samples_to_log = 5  # 定义要记录的样本数量
            for i, (pred, ref) in enumerate(zip(decoded_preds, decoded_labels)):
                if i < num_samples_to_log:
                    print(f"Prediction Sample {i}\n", [pred])
                    print(f"Reference Sample {i}\n", [ref])
        return {"cer": score}


    
    
    
    trainer = BunnyTrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           compute_metrics=compute_metrics if training_args.evaluation_strategy != 'no' else None, 
                           **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            # 保存一个tokenizer
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
