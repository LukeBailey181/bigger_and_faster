import argparse
import time
import os
import random
import json
import sys
import json

import torch
import numpy as np

#from transformer.modeling_super_kd import SuperTinyBertForPreTraining, BertConfig
from transformer.tokenization import BertTokenizer
import transformers
from hf_dataset_gen import arch_cpu_time

# Example arch
#{
#    'sample_layer_num': 5, 
#    'sample_num_attention_heads': [12, 12, 12, 12, 12], 
#    'sample_hidden_size': 504, 
#    'sample_intermediate_sizes': [992, 992, 992, 992, 992], 
#    'sample_qkv_sizes': [504, 504, 504, 504, 504]
#}

ARCH_TO_CONFIG_PARAMS = {
    "sample_layer_num" : "num_hidden_layers",
    "sample_num_attention_heads" : "num_attention_layers",
    "sample_hidden_size" : "hidden_size",
    "sample_intermediate_sizes" : "intermediate_size"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)

    # architecture
    parser.add_argument("--arch_path", type=str)
    parser.add_argument("--infer_cnt", type=int, default=5)
    parser.add_argument(
        "--data_type", type=str, default="fp", help="One of 'fp' or 'ptq'"
    )
    args = parser.parse_args()

    if args.data_type not in ["fp", "ptq"]:
        raise ValueError(
            f"data_type argument {args.data_type} not one of ['fp', 'ptq']"
        )

    config_dict = {
      "attention_probs_dropout_prob": 0.1,
      "gradient_checkpointing": False,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "model_type": "bert",
      "num_attention_heads": 4,
      "num_hidden_layers": 4,
      "pad_token_id": 0,
      "position_embedding_type": "absolute",
      "transformers_version": "4.3.1",
      "type_vocab_size": 2,
      "use_cache": True,
      "vocab_size": 30522
    }

    # Transfer input config to 
    with open(args.arch_path, 'r') as f:
        arch = json.load(f)

    for arch_param, config_param in ARCH_TO_CONFIG_PARAMS.items():

        if isinstance(arch[arch_param], list):
            config_dict[config_param] = arch[arch_param][0]
        else:
            config_dict[config_param] = arch[arch_param]

    bert_config = transformers.BertConfig.from_dict(config_dict)
    model = transformers.BertModel(config=bert_config)

    if args.data_type == "ptq":
        arch_cpu_time(model, arch, args, quant=True)
    elif args.data_type == "fp":
        arch_cpu_time(model, arch, args, quant=False)

