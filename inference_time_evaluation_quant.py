
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
import transformers


def text_padding(max_seq_length, device, batch_size):
    input_ids = [9333] * max_seq_length
    input_masks = max_seq_length * [1]
    input_segments = max_seq_length * [0]

    input_ids = torch.tensor([input_ids]*batch_size, dtype=torch.long).to(device)
    input_masks = torch.tensor([input_masks]*batch_size, dtype=torch.long).to(device)
    input_segments = torch.tensor([input_segments]*batch_size, dtype=torch.long).to(device)
    return input_ids, input_masks, input_segments

def arch_cpu_time(model, arch, args, save_dir, quant=False, partition_name=''):

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

    #print('{}\t{}'.format(arch, aver_time))
    output_file_prefix = 'lat_'
    if quant:
        output_file_prefix += 'quant_'
    with open(save_dir + output_file_prefix + partition_name + '.tmp', 'a') as f:
        f.write(f'{arch}\t{aver_time}\n')

    master_end = time.time()

    #print(master_end - master_start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='tinybert_model/4l/', type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)

    # Search space for sub_bart architecture
    parser.add_argument('--layer_num_space', nargs='+', type=int, default=[1, 5]) 
    parser.add_argument('--hidden_size_space', nargs='+', type=int, default=[120, 564])
    parser.add_argument('--qkv_size_space', nargs='+', type=int, default=[180, 528])
    parser.add_argument('--head_num_space', nargs='+', type=int, default=[1, 12])
    parser.add_argument('--intermediate_size_space', nargs='+', type=int, default=[128, 1024])
    parser.add_argument('--mlm', action='store_true')
    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--save_dir', nargs='+', type=str, default='./latency_dataset')

    parser.add_argument('--infer_cnt', type=int, default=5)

    args = parser.parse_args()
    
    """
    config = BertConfig.from_pretrained(os.path.join(args.bert_model, 'config.json'))
    model = SuperTinyBertForPreTraining.from_scratch(args.bert_model, config)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    device = 'cpu'
    model.to(device)
    model.eval()
    """

    torch.set_num_threads(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build arch space
    min_hidden_size, max_hidden_size = args.hidden_size_space
    min_ffn_size, max_ffn_size = args.intermediate_size_space
    min_qkv_size, max_qkv_size = args.qkv_size_space
    min_head_size, max_head_size = args.head_num_space

    #get save dir
    save_dir = args.save_dir[0] 
    if save_dir[-1] != "/":
        save_dir += "/"

    """
    hidden_step = 12
    ffn_step = 12
    qkv_step = 12
    head_step = 1
    """

    hidden_step = 12
    ffn_step = 12
    qkv_step = 12
    head_step = 1

    number_hidden_step = int((max_hidden_size - min_hidden_size) / hidden_step)
    number_ffn_step = int((max_ffn_size - min_ffn_size) / ffn_step)
    number_qkv_step = int((max_qkv_size - min_qkv_size) / qkv_step)
    number_head_step = int((max_head_size - min_head_size) / head_step)

    layer_numbers = list(range(args.layer_num_space[0], args.layer_num_space[1] + 1))
    hidden_sizes = [i * hidden_step + min_hidden_size for i in range(number_hidden_step + 1)]
    ffn_sizes = [i * ffn_step + min_ffn_size for i in range(number_ffn_step + 1)]
    qkv_sizes = [i * qkv_step + min_qkv_size for i in range(number_qkv_step + 1)]
    head_sizes = [i * head_step + min_head_size for i in range(number_head_step + 1)]

    '''
    layer_numbers.reverse()
    hidden_sizes.reverse()
    ffn_sizes.reverse()
    qkv_sizes.reverse()
    head_sizes.reverse()
    '''

    # Test BERT-base time

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
    # Init write file
    print(save_dir)
    if args.quant:
        print("Measuring fixed point models")
    else:
        print("Measuring float point models")

    # Instantiate model

    #config = BertConfig.from_dict(config)
    bert_config = transformers.BertConfig.from_dict(config_dict)
    model = transformers.BertModel(bert_config)
    #tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    device = 'cpu'
    model.to(device)
    model.eval()
    

    #arch_cpu_time(model, config, args)
    print(layer_numbers)
    print(hidden_sizes[-1:-5:-1])
    print(ffn_sizes[-1:-5:-1])
    print(qkv_sizes[-1:-5:-1])
    print('Total Length:', len(layer_numbers)*len(hidden_sizes)*len(ffn_sizes))

    for layer_num in layer_numbers:
        arch = dict()
        arch['sample_layer_num'] = layer_num
        config_dict['num_hidden_layers'] = layer_num

        if not args.mlm:
            arch['sample_num_attention_heads'] = [12] * layer_num
            config_dict['num_attention_heads'] = 12  

            for hidden_size in hidden_sizes:
                arch['sample_hidden_size'] = hidden_size
                config_dict['hidden_size'] = hidden_size

                for ffn_size in ffn_sizes:
                    arch['sample_intermediate_sizes'] = [ffn_size] * layer_num
                    arch['sample_qkv_sizes'] = [hidden_size] * layer_num
                    config_dict['intermediate_size'] = ffn_size  
                    
                    bert_config = transformers.BertConfig.from_dict(config_dict)
                    model = transformers.BertModel(config=bert_config)
                    arch_cpu_time(model, arch, args, save_dir, quant=args.quant)

        print(f"CURRENT PROP DONE: {(layer_num/len(layer_numbers)) * 100}%")
