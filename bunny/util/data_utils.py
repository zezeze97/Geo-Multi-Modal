import os
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List, Any, Tuple, Union

import torch

import transformers

from bunny.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from torch.utils.data import Dataset
from bunny.trl.trl.trainer.utils import DPODataCollatorWithPadding


from bunny import conversation as conversation_lib

from bunny.util.mm_utils import tokenizer_image_token

from PIL import Image
from bunny.util.data_aug import enhance_image


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    val_data_path: str = field(default=None, metadata={"help": "Path to the validation data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)
    customized_aug: bool = False


def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

            replace_token = DEFAULT_IMAGE_TOKEN

            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_bunny(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    
    # Tokenize conversations
    # print(f'conversations is:\n{conversations}')
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print(f'conversation:{conversation}')
        rounds = conversation.split(conv.sep2)
        # print(f'rounds:{rounds}')
        cur_len = 0
        end_token_cnt = 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # print(f'Instruction is:\n{parts[0]}')
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 1
            end_token_cnt += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_llama(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    # print(f'sources is: {sources}')
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": " 
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print(f'target id is\n{target}')
        # print(f'target is\n{[tokenizer.decode(target[target!=-200])]}')
        rounds = conversation.split(conv.sep2)
        cur_len = 0
        end_token_cnt = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep) # [' USER: <image>\n question', ans]
            if len(parts) != 2:
                break
            parts[0] += sep 
            if has_image: 
                # temp = torch.stack([tokenizer_image_token(rou, tokenizer, return_tensors='pt')], dim=0)
                # print(f'rou_id:\n{temp}')
                # print(f'rou is:\n{[tokenizer.decode(temp[temp!=-200])]}') 
                round_len = len(tokenizer_image_token(rou, tokenizer))
                # temp = torch.stack([tokenizer_image_token(parts[0], tokenizer, return_tensors='pt')], dim=0)
                # print(f'instruction_id:\n{temp}')
                # print(f'instruction is:\n{[tokenizer.decode(temp[temp!=-200])]}')
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1

            else:
                round_len = len(tokenizer(rou).input_ids)
                # instruction_len = len(tokenizer(parts[0]).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 1 # add <end_of_text>
            end_token_cnt += 1
            
            # temp = target[cur_len: cur_len + instruction_len]
            # print(f'mask id is\n{temp}')
            # print(f'mask part is\n{[tokenizer.decode(temp[temp!=-200])]}')
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            # print(f'target is\n{[tokenizer.decode(target[target!=-100])]}')

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        cur_len -= end_token_cnt 
        

      
        
        
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: 
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
        
        

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_yi_chat(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    # print(f'sources is: {sources}')
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations
    # print(conversations)
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        _rounds = conversation.split("<|im_end|><|im_start|>")
        # 修复
        rounds = []
        for round in _rounds:
            if not round.startswith('<|im_start|>'):
                round = '<|im_start|>' + round
            if round.endswith('<|im_end|>'):
                round = round.rstrip('<|im_end|>')
            rounds.append(round)
        # print(f'rounds is {len(rounds)}')
        cur_len = 0
        end_token_cnt = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep) # ['<|im_start|>user\n<image>\nqa', ans]
            if len(parts) != 2:
                break
            parts[0] += sep 
            if has_image: 
                # temp = torch.stack([tokenizer_image_token(rou, tokenizer, return_tensors='pt')], dim=0)
                # print(f'rou_id:\n{temp}')
                # print(f'rou is:\n{[tokenizer.decode(temp[temp!=-200])]}') 
                round_len = len(tokenizer_image_token(rou, tokenizer))
                # temp = torch.stack([tokenizer_image_token(parts[0], tokenizer, return_tensors='pt')], dim=0)
                # print(f'instruction_id:\n{temp}')
                # print(f'instruction is:\n{[tokenizer.decode(temp[temp!=-200])]}')
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) 

            else:
                round_len = len(tokenizer(rou).input_ids)
                # instruction_len = len(tokenizer(parts[0]).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) 

            round_len += 1 # add <end_of_text>
            end_token_cnt += 1
            
            # temp = target[cur_len: cur_len + instruction_len]
            # print(f'mask id is\n{temp}')
            # print(f'mask part is\n{[tokenizer.decode(temp[temp!=-200])]}')
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        # cur_len -= end_token_cnt 
        
        # print(f'_mask target is\n{[tokenizer.decode(target[target!=-100])]}')
      
        
        
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: 
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print(f'Error: {[conversation]}')
        

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_yi(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    # print(f'sources is: {sources}')
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations
    # print(conversations)
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": " 
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print(f'target id is\n{target}')
        # print(f'target is\n{[tokenizer.decode(target[target!=-200])]}')
        rounds = conversation.split(conv.sep2)
        cur_len = 0
        end_token_cnt = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep) # [' USER: <image>\n question', ans]
            if len(parts) != 2:
                break
            parts[0] += sep 
            if has_image: 
                # temp = torch.stack([tokenizer_image_token(rou, tokenizer, return_tensors='pt')], dim=0)
                # print(f'rou_id:\n{temp}')
                # print(f'rou is:\n{[tokenizer.decode(temp[temp!=-200])]}') 
                round_len = len(tokenizer_image_token(rou, tokenizer))
                # temp = torch.stack([tokenizer_image_token(parts[0], tokenizer, return_tensors='pt')], dim=0)
                # print(f'instruction_id:\n{temp}')
                # print(f'instruction is:\n{[tokenizer.decode(temp[temp!=-200])]}')
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1

            else:
                round_len = len(tokenizer(rou).input_ids)
                # instruction_len = len(tokenizer(parts[0]).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 1 # add <end_of_text>
            end_token_cnt += 1
            
            # temp = target[cur_len: cur_len + instruction_len]
            # print(f'mask id is\n{temp}')
            # print(f'mask part is\n{[tokenizer.decode(temp[temp!=-200])]}')
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        # cur_len -= end_token_cnt 
        
        # print(f'_mask target is\n{[tokenizer.decode(target[target!=-100])]}')
      
        
        
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: 
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
        
        # print(f'mask target is\n{[tokenizer.decode(target[target!=-100])]}')
        

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mistral(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    # print(f'sources is: {sources}')
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(f'conversations is:\n{conversations}') # here conversations is ['[INST] <image>\nqs [/INST]ans</s>']
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] # [/INST]
    for conversation, target in zip(conversations, targets):
        # print(f'target_id is\n{target}')
        # print(f'target is\n{[tokenizer.decode(target[target!=-200])]}') # <s> [INST] <image>\nqs [/INST]ans</s>
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print(f'conversation:\n{conversation}') # Here is: [INST] <image>\nqs [/INST]ans</s>
        rounds = conversation.split(conv.sep2)
        # print(f'rounds:\n{rounds}') # Here is:['[INST] <image>\nqs [/INST]ans', '']
        cur_len = 0
        end_token_cnt = 0

        for i, rou in enumerate(rounds):
            # print(f'rou is:\n{rou}') # row is:[INST] <image>\nqs [/INST]ans
            if rou == "":
                break
            parts = rou.split(sep) # ['[INST] <image>\nqs', ans]
            if len(parts) != 2:
                break
            parts[0] += sep 
            # print(f'Instruction is:\n{parts[0]}') # [INST] <image>\nqs [/INST]
            if has_image: 
                # temp = torch.stack([tokenizer_image_token(rou, tokenizer, return_tensors='pt')], dim=0)
                # print(f'rou_id:\n{temp}')
                # print(f'rou is:\n{tokenizer.decode(temp[temp!=-200])}') 
                round_len = len(tokenizer_image_token(rou, tokenizer))
                # temp = torch.stack([tokenizer_image_token(parts[0], tokenizer, return_tensors='pt')], dim=0)
                # print(f'instruction_id:\n{temp}')
                # print(f'instruction is:\n{tokenizer.decode(temp[temp!=-200])}')
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1

            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids)
                # instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 1 # add </s>
            end_token_cnt += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        # print(f'target is\n{[tokenizer.decode(target[target!=-100])]}')
        # cur_len -= end_token_cnt # debug
        
        
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: 
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
        
        

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_wizard(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    # print(f'sources is: {sources}')
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(f'conversations is:\n{conversations}') # here conversations is ['Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n<image>\nqs\n\n### Response: ans</s>']
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ":" # \n\n### Response: 
    # print(f'sep is {[sep]}')
    for conversation, target in zip(conversations, targets):
        # print(f'target_id is\n{target}')
        # print(f'target is\n{[tokenizer.decode(target[target!=-200])]}') # <s> Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n <image>\nqs.\n\n### Response: ans<\s>
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) + 1 # debug
        # print(f'pad token id is: {tokenizer.pad_token_id}')
        # print(f'conversation:\n{conversation}') # Here is: Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n <image>\nqs.\n\n### Response: ans<\s>
        rounds = conversation.split(conv.sep2)
        # print(f'rounds:\n{rounds}') # Here is:['Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n <image>\nqs.\n\n### Response: ans', '']
        cur_len = 0
        end_token_cnt = 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep) # ['[INST] <image>\nqs', ans]
            if len(parts) != 2:
                break
            parts[0] += sep 
            # print(f'Instruction is:\n{parts[0]}') # [INST] <image>\nqs [/INST]
            if has_image: 
                # temp = torch.stack([tokenizer_image_token(rou, tokenizer, return_tensors='pt')], dim=0)
                # print(f'rou_id:\n{temp}')
                # print(f'rou is:\n{[tokenizer.decode(temp[temp!=-200])]}') 
                round_len = len(tokenizer_image_token(rou, tokenizer))
                # temp = torch.stack([tokenizer_image_token(parts[0], tokenizer, return_tensors='pt')], dim=0)
                # print(f'instruction_id:\n{temp}')
                # print(f'instruction is:\n{[tokenizer.decode(temp[temp!=-200])]}')
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1

            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids)
                # instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 1 # add </s>
            end_token_cnt += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        # cur_len -= end_token_cnt # debug
        
        
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: 
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
        
        

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)



def preprocess_caption(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + conversation_lib.default_conversation.sep + source[1]['value'] + conversation_lib.default_conversation.sep2
        conversations.append(conversation)
    # tokenize conversations
    # print(tokenizer.tokenize(source[1]['value']))
    # print(f'conversations is:{conversations}')
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    # print(f'input id is: {input_ids}')
    # print(f'input_ids is {input_ids}')
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'] + conversation_lib.default_conversation.sep, tokenizer))
        # print(f'tokenized_len is {tokenized_len}')
        target[:tokenized_len] = IGNORE_INDEX
        # print(f'target is {target}')
    # print(f'mask target is\n{[tokenizer.decode(target[target!=-100])]}')
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    # if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
    if conversation_lib.default_conversation.version == "plain":
        return preprocess_plain(sources, tokenizer)
    elif conversation_lib.default_conversation.version == "bunny":
        return preprocess_bunny(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version =='mistral':
        return preprocess_mistral(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version =='llama':
        return preprocess_llama(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version == 'yi':
        return preprocess_yi(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version == 'yi-chat':
        return preprocess_yi_chat(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version == 'qwen-chat':
        return preprocess_yi_chat(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version =='wizard':
        return preprocess_wizard(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version=='caption':
        return preprocess_caption(sources, tokenizer)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.customized_aug:
                image = enhance_image(image)
            if self.data_args.image_aspect_ratio == 'pad':
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

                # image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image =  expand2square(image, (255, 255, 255))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            # print(f'sources is {sources}')
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        labels = labels[:, :self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    eval_dataset = None
    if data_args.val_data_path is not None:
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.val_data_path,
                                          data_args=data_args)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


    
    



class LazyDPODataset(Dataset):
    """Dataset for DPO RL training."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazyDPODataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        '''
        {'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected,
        'image': torch.tensor}
        
        '''
        data_dict = self.list_data_dict[i]
        if 'image' in data_dict:
            image_file = data_dict['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
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

                image = expand2square(image, (255, 255, 255))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([data_dict["conversations"]]), self.data_args)
            # print(f'sources is {sources}')
        else:
            sources = copy.deepcopy([data_dict["conversations"]])
        assert len(sources) == 1 and len(sources[0]) == 3
        for item in sources[0]:
            if item['from'] == 'human':
                prompt = item['value']
            elif item['from'] == 'chosen':
                chosen = item['value']
            elif item['from'] == 'rejected':
                rejected = item['value']
        
        data_dict = dict(prompt=prompt,
                        chosen=chosen,
                        rejected=rejected)

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

def make_conv(prompt, answer):
    return [
        {
            "from": "human",
            "value": prompt,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]

@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    tokenizer: transformers.PreTrainedTokenizer = None
    def collate(self, batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                else:
                    continue
                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        for k in ['chosen_input_ids', 'rejected_input_ids']:
            attn_k = k.replace('input_ids', 'attention_mask')
            padded_batch[attn_k] = padded_batch[k].ne(self.tokenizer.pad_token_id)

        return padded_batch
    
    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        ) -> Dict:
        
        batch = {}
        
        chosen_sources = make_conv(prompt, chosen)
        rejected_sources = make_conv(prompt, rejected)
        
        # print(f'chosen_sources {chosen_sources}')
        
        chosen_data_dict = preprocess([chosen_sources], self.tokenizer, has_image='<image>' in prompt)
        rejected_data_dict = preprocess([rejected_sources], self.tokenizer, has_image='<image>' in prompt)
        
        chosen_data_dict = {k: v[0] for k, v in chosen_data_dict.items()}
        rejected_data_dict = {k: v[0] for k, v in rejected_data_dict.items()}
        
        for k, toks in {
            "chosen": chosen_data_dict,
            "rejected": rejected_data_dict,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []
        image_batch = []
        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]
            image = feature['image']
            image_batch.append(image)
             
            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        padded_batch =  self.collate(tokenized_batch)
        
        if all(x is not None and x.shape == image_batch[0].shape for x in image_batch):
            padded_batch['images'] = torch.stack(image_batch)
        else:
            padded_batch['images'] = image_batch
        return padded_batch
    
    
    

def make_dpo_data_module(tokenizer, data_args) -> Dict:
    """Make dataset and collator for DPO."""
    train_dataset = LazyDPODataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args)
    data_collator = DPODataCollator(pad_token_id=tokenizer.pad_token_id, label_pad_token_id=IGNORE_INDEX, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)








if __name__ == '__main__':
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from bunny.model import Qwen2Tokenizer
    from bunny.model.multimodal_encoder.siglip.siglip_encoder import SigLipImageProcessor
    '''
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct',
                                                padding_side="right",
                                                use_fast=False,
                                                trust_remote_code=True)
    '''
    
    tokenizer = Qwen2Tokenizer(vocab_file='data/formalgeo7k/formalgeo7k_v2/vocab/vocab.json',
                                   merges_file='data/formalgeo7k/formalgeo7k_v2/vocab/merges.txt',
                                   model_max_length=2048)
    special_tokens = ["construction_cdl", "image_cdl", "Shape", "LengthOfLine", "MeasureOfAngle", "Equal", "PerpendicularBetweenLine", "Collinear", "Cocircular", "ParallelBetweenLine"]
    # special_tokens_dict = {'additional_special_tokens': special_tokens}
    # tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_tokens(['\n'] + special_tokens)
    print(len(tokenizer))
    
    
    
    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})
    # tokenizer('<image>\tconstruction_cdl is:\nShape(ED,DA,AF,FE), Shape(DE,AED), Shape(EF,AFE), Shape(DA，AF,AFD), Collinear(DAF), Cocircular(A,EDF)\nimage_cdl is:\nEqual(LengthOfLine(EF),60), Equal(LengthOfLine(FD),65)<|endoftext|>')
    conversation_lib.default_conversation = conversation_lib.conv_templates['caption']
    class temp:
        def __init__(self, image_folder, image_processor, is_multimodal, data_path, val_data_path, image_aspect_ratio, customized_aug=False) -> None:
            self.image_folder = image_folder
            self.image_processor = image_processor
            self.is_multimodal = is_multimodal
            self.data_path = data_path
            self.val_data_path = val_data_path
            self.image_aspect_ratio = image_aspect_ratio
            self.customized_aug = customized_aug
    
    data_args = temp(image_folder='data/formalgeo7k/formalgeo7k_v2',
                    is_multimodal=True,
                    data_path='data/formalgeo7k/formalgeo7k_v2/custom_json/caption_structure_only/caption_structure_only_train_aug.json',
                    val_data_path=None,
                    # data_path='data/qa_tuning.json',
                    image_processor= SigLipImageProcessor(),
                    image_aspect_ratio='pad'
    )
    data_module_dict = make_supervised_data_module(tokenizer, data_args)
    dataset = data_module_dict['train_dataset']
    collector = data_module_dict['data_collator']
    item_lst = []
    for item in tqdm(dataset):
        pass