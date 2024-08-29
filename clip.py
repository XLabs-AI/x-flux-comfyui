import json
import os
from transformers import (CLIPImageProcessor, 
                          CLIPVisionModelWithProjection, 
                          CLIPVisionConfig,
                          AutoConfig)



class FluxClipViT:
    def __init__(self, path_model = None):
        if path_model is None:
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            
        else:
            _dir = os.path.dirname(path_model)
            write_config(_dir)
            config = CLIPVisionConfig.from_pretrained(
                os.path.join(_dir, "flux_clip_config.json")
            )
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                path_model,
                config=config,
                use_safetensors = True,
            )
        self.image_processor = CLIPImageProcessor()
        self.load_device = next(self.model.parameters()).device

    def __call__(self, image):
        img = self.image_processor(
            images=image, return_tensors="pt"
            )
        img = img.pixel_values
        return self.model(img).image_embeds


def write_config(path):
    #check if exists
    if os.path.exists(os.path.join(path, "flux_clip_config.json")):
        return
    with open(os.path.join(path, "flux_clip_config.json"), "w") as f:
        json.dump(json_config, f, indent=4)

json_config = {'_name_or_path': 'clip-vit-large-patch14/',
 'architectures': ['CLIPModel'],
 'initializer_factor': 1.0,
 'logit_scale_init_value': 2.6592,
 'model_type': 'clip',
 'projection_dim': 768,
 'text_config': {'_name_or_path': '',
  'add_cross_attention': False,
  'architectures': None,
  'attention_dropout': 0.0,
  'bad_words_ids': None,
  'bos_token_id': 0,
  'chunk_size_feed_forward': 0,
  'cross_attention_hidden_size': None,
  'decoder_start_token_id': None,
  'diversity_penalty': 0.0,
  'do_sample': False,
  'dropout': 0.0,
  'early_stopping': False,
  'encoder_no_repeat_ngram_size': 0,
  'eos_token_id': 2,
  'finetuning_task': None,
  'forced_bos_token_id': None,
  'forced_eos_token_id': None,
  'hidden_act': 'quick_gelu',
  'hidden_size': 768,
  'id2label': {'0': 'LABEL_0', '1': 'LABEL_1'},
  'initializer_factor': 1.0,
  'initializer_range': 0.02,
  'intermediate_size': 3072,
  'is_decoder': False,
  'is_encoder_decoder': False,
  'label2id': {'LABEL_0': 0, 'LABEL_1': 1},
  'layer_norm_eps': 1e-05,
  'length_penalty': 1.0,
  'max_length': 20,
  'max_position_embeddings': 77,
  'min_length': 0,
  'model_type': 'clip_text_model',
  'no_repeat_ngram_size': 0,
  'num_attention_heads': 12,
  'num_beam_groups': 1,
  'num_beams': 1,
  'num_hidden_layers': 12,
  'num_return_sequences': 1,
  'output_attentions': False,
  'output_hidden_states': False,
  'output_scores': False,
  'pad_token_id': 1,
  'prefix': None,
  'problem_type': None,
  'projection_dim': 768,
  'pruned_heads': {},
  'remove_invalid_values': False,
  'repetition_penalty': 1.0,
  'return_dict': True,
  'return_dict_in_generate': False,
  'sep_token_id': None,
  'task_specific_params': None,
  'temperature': 1.0,
  'tie_encoder_decoder': False,
  'tie_word_embeddings': True,
  'tokenizer_class': None,
  'top_k': 50,
  'top_p': 1.0,
  'torch_dtype': None,
  'torchscript': False,
  'transformers_version': '4.16.0.dev0',
  'use_bfloat16': False,
  'vocab_size': 49408},
 'text_config_dict': {'hidden_size': 768,
  'intermediate_size': 3072,
  'num_attention_heads': 12,
  'num_hidden_layers': 12,
  'projection_dim': 768},
 'torch_dtype': 'float32',
 'transformers_version': None,
 'vision_config': {'_name_or_path': '',
  'add_cross_attention': False,
  'architectures': None,
  'attention_dropout': 0.0,
  'bad_words_ids': None,
  'bos_token_id': None,
  'chunk_size_feed_forward': 0,
  'cross_attention_hidden_size': None,
  'decoder_start_token_id': None,
  'diversity_penalty': 0.0,
  'do_sample': False,
  'dropout': 0.0,
  'early_stopping': False,
  'encoder_no_repeat_ngram_size': 0,
  'eos_token_id': None,
  'finetuning_task': None,
  'forced_bos_token_id': None,
  'forced_eos_token_id': None,
  'hidden_act': 'quick_gelu',
  'hidden_size': 1024,
  'id2label': {'0': 'LABEL_0', '1': 'LABEL_1'},
  'image_size': 224,
  'initializer_factor': 1.0,
  'initializer_range': 0.02,
  'intermediate_size': 4096,
  'is_decoder': False,
  'is_encoder_decoder': False,
  'label2id': {'LABEL_0': 0, 'LABEL_1': 1},
  'layer_norm_eps': 1e-05,
  'length_penalty': 1.0,
  'max_length': 20,
  'min_length': 0,
  'model_type': 'clip_vision_model',
  'no_repeat_ngram_size': 0,
  'num_attention_heads': 16,
  'num_beam_groups': 1,
  'num_beams': 1,
  'num_hidden_layers': 24,
  'num_return_sequences': 1,
  'output_attentions': False,
  'output_hidden_states': False,
  'output_scores': False,
  'pad_token_id': None,
  'patch_size': 14,
  'prefix': None,
  'problem_type': None,
  'projection_dim': 768,
  'pruned_heads': {},
  'remove_invalid_values': False,
  'repetition_penalty': 1.0,
  'return_dict': True,
  'return_dict_in_generate': False,
  'sep_token_id': None,
  'task_specific_params': None,
  'temperature': 1.0,
  'tie_encoder_decoder': False,
  'tie_word_embeddings': True,
  'tokenizer_class': None,
  'top_k': 50,
  'top_p': 1.0,
  'torch_dtype': None,
  'torchscript': False,
  'transformers_version': '4.16.0.dev0',
  'use_bfloat16': False},
 'vision_config_dict': {'hidden_size': 1024,
  'intermediate_size': 4096,
  'num_attention_heads': 16,
  'num_hidden_layers': 24,
  'patch_size': 14,
  'projection_dim': 768}}
