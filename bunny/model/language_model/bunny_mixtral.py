from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from .mixtral import MixtralModel, MixtralConfig, MixtralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM

    
    

class BunnyMixtralConfig(MixtralConfig):
    model_type = "bunny-mixtral"


class BunnyMixtralModel(BunnyMetaModel, MixtralModel):
    config_class = BunnyMixtralConfig

    def __init__(self, config: MixtralConfig):
        super(BunnyMixtralModel, self).__init__(config)


class BunnyMixtralForCausalLM(MixtralForCausalLM, BunnyMetaForCausalLM):
    config_class = BunnyMixtralConfig

    def __init__(self, config):
        super(MixtralForCausalLM, self).__init__(config)
        self.model = BunnyMixtralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_coef = config.router_aux_loss_coef

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(f"use_cache: {use_cache}")
        if inputs_embeds is None:
            # print(f'input ids shape is:{input_ids.shape}')
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

            # print(f'input_ids is:{input_ids}\n inputs_embeds is {inputs_embeds.shape}\n labels is {labels}\n attention_mask is {attention_mask.shape}, past_key_values is {past_key_values}')
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_router_logits=output_router_logits
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("bunny-mixtral", BunnyMixtralConfig)
AutoModelForCausalLM.register(BunnyMixtralConfig, BunnyMixtralForCausalLM)
