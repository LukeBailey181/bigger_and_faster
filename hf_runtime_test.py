
import argparse
import time
import os
import random
import json
import sys

import torch
import numpy as np

#from transformer.modeling_super_kd import SuperTinyBertForPreTraining, BertConfig
from transformer.tokenization import BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoConfig
import transformers


def text_padding(max_seq_length, device, batch_size):
    input_ids = [9333] * max_seq_length
    input_masks = max_seq_length * [1]
    input_segments = max_seq_length * [0]

    input_ids = torch.tensor([input_ids]*batch_size, dtype=torch.long).to(device)
    input_masks = torch.tensor([input_masks]*batch_size, dtype=torch.long).to(device)
    input_segments = torch.tensor([input_segments]*batch_size, dtype=torch.long).to(device)
    return input_ids, input_masks, input_segments

def arch_cpu_time(model, arch, args, save_dir, quant=False):

    master_start = time.time()

    aver_time = 0.
    infer_cnt = args.infer_cnt

    for i in range(infer_cnt):
        input_ids, input_masks, input_segments = text_padding(args.max_seq_length,
                                                              device,
                                                              args.batch_size)

        if quant:

            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )


            start = time.time()
            with torch.no_grad():
                quantized_model(input_ids, input_masks)
            end = time.time()

        else:
            start = time.time()
            with torch.no_grad():
                model(input_ids, input_masks)
            end = time.time()

        sep = 1000 * (end - start)

        if i == 0:
            continue
        else:
            aver_time += sep / (args.infer_cnt - 1)

    print('{}\t{}'.format(arch, aver_time))
    with open(save_dir + 'lat.tmp', 'a') as f:
        f.write(f'{arch}\t{aver_time}\n')

    master_end = time.time()

    print(master_end - master_start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)

    # Search space for sub_bart architecture
    parser.add_argument('--mlm', action='store_true')
    parser.add_argument('--save_dir', nargs='+', type=str, default='./latency_dataset')

    parser.add_argument('--infer_cnt', type=int, default=10)

    args = parser.parse_args()

    model_names = ["google/bert_uncased_L-4_H-512_A-8"]
    
    """
    config = BertConfig.from_pretrained(os.path.join(args.bert_model, 'config.json'))
    model = SuperTinyBertForPreTraining.from_scratch(args.bert_model, config)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    device = 'cpu'
    model.to(device)
    model.eval()
    """
    #get save dir
    save_dir = args.save_dir[0] 
    if save_dir[-1] != "/":
        save_dir += "/"

    # Init write file
    print(save_dir)
    with open(save_dir + 'lat.tmp', 'w') as f:
        pass

        # Test BERT-base time

    config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 512,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522
    }

    torch.set_num_threads(1)

    for model_name in model_names:
        #config = AutoConfig.from_pretrained(model_name)
        #config.num_labels = 3
        #model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        bert_config = transformers.BertConfig.from_dict(config_dict)
        model = transformers.BertModel(config=bert_config)

        # Instantiate model
        device = 'cpu'
        model.to(device)
        model.eval()

        arch_cpu_time(model, model_name, args, save_dir, quant=False)

