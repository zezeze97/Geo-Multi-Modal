import torch
import transformers
from bunny.model.builder import load_pretrained_model



# set device
torch.set_default_device('cpu')  # or 'cuda'
# torch.cuda.set_device(0) 
tokenizer, model_1, image_processor, context_len = load_pretrained_model(model_path='checkpoints/checkpoints-qwen2/bunny-qwen2-formalgeov2-qa-structureMix_only-sft10+10', 
                                                                       vision_encoder_path='checkpoints/checkpoints-qwen2/bunny-qwen2-formalgeov2-qa-structureMix_only-sft10+10', 
                                                                       model_base=None, 
                                                                       model_name='bunny-qwen2',
                                                                       model_type='qwen2',
                                                                       device='cpu')

tokenizer, model_2, image_processor, context_len = load_pretrained_model(model_path='checkpoints/checkpoints-qwen2/bunny-qwen2-formalgeov2-qa-structureMix_only-sft10+10/checkpoint-100', 
                                                                       vision_encoder_path='checkpoints/checkpoints-qwen2/bunny-qwen2-formalgeov2-qa-structureMix_only-sft10+10/checkpoint-100', 
                                                                       model_base=None, 
                                                                       model_name='bunny-qwen2',
                                                                       model_type='qwen2',
                                                                       device='cpu')
    
vision_tower_1 = {k: v for k, v in model_1.get_vision_tower().named_parameters()}
vision_model_2 ={k: v for k, v in model_2.get_vision_tower().named_parameters()}
# vision_tower_1 = {k: v for k, v in model_1.named_parameters()}
# vision_model_2 ={k: v for k, v in model_2.named_parameters()}



# 确保两个模型的视觉相关层数量和名称相同
assert vision_tower_1.keys() == vision_model_2.keys()

differences = {}
for name in vision_tower_1.keys():
    differences[name] = (vision_tower_1[name] - vision_model_2[name]).abs().mean().item()
print(differences)

