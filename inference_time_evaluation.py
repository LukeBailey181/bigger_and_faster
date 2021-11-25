import argparse
import time
import os
import random
import json
import sys

import torch
import numpy as np

from transformer.modeling_super_kd import SuperTinyBertForPreTraining, BertConfig
from transformer.tokenization import BertTokenizer

import onnxruntime
from onnxruntime.quantization import quantize_dynamic

def text_padding(max_seq_length, device, batch_size):
    input_ids = [9333] * max_seq_length
    input_masks = max_seq_length * [1]
    input_segments = max_seq_length * [0]

    input_ids = torch.tensor([input_ids]*batch_size, dtype=torch.long).to(device)
    input_masks = torch.tensor([input_masks]*batch_size, dtype=torch.long).to(device)
    input_segments = torch.tensor([input_segments]*batch_size, dtype=torch.long).to(device)
    return input_ids, input_masks, input_segments

def convert_torch_to_onnx_inpput(input_ids, input_masks, config_keys, arch):

    list_arch = []

    for key in config_keys:
        if isinstance(arch[key], list):
            list_arch.append(arch[key][0])
        else:
            list_arch.append(arch[key])

    onnx_input = (input_ids, torch.Tensor(list_arch), input_masks)

    return onnx_input

def arch_cpu_time(model, arch, args, save_dir, quant=False, config_keys=None):

    master_start = time.time()

    aver_time = 0.
    infer_cnt = args.infer_cnt

    if quant:

        input_ids, input_masks, input_segments = text_padding(args.max_seq_length,
                                               device,
                                               args.batch_size)

        dummy_input = convert_torch_to_onnx_inpput(input_ids, input_masks, config_keys, arch) 

        input_names = [f"input_{i}" for i in range(len(dummy_input))]
        start = time.time()
        torch.onnx.export(model,
                  dummy_input,
                  "./temp.onnx",
                  do_constant_folding=True,
                  input_names = input_names,
                  output_names = ["output"],
                  verbose=False,
                  opset_version=12)

        quantize_dynamic("./temp.onnx", "./temp_q.onnx")

    for i in range(infer_cnt):
        input_ids, input_masks, input_segments = text_padding(args.max_seq_length,
                                                              device,
                                                              args.batch_size)

        if quant:

            session  = onnxruntime.InferenceSession(
                "./temp_q.onnx", providers=["CPUExecutionProvider"]
            )

            inp = convert_torch_to_onnx_inpput(input_ids, input_masks, config_keys, arch)

            inp = [np.array(x) for x in inp]

            ort_inputs = {
                'input_0': inp[0],
                'input_2' : inp[2]
            }
            start = time.time()
            preds = session.run(None, ort_inputs)
            end = time.time()

        else:
            start = time.time()
            with torch.no_grad():
                model(input_ids, arch, input_masks, kd=not args.mlm)
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
    parser.add_argument("--bert_model", default='tinybert_model/4l/', type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)

    # Search space for sub_bart architecture
    parser.add_argument('--layer_num_space', nargs='+', type=int, default=[5, 5])
    parser.add_argument('--hidden_size_space', nargs='+', type=int, default=[128, 564])
    parser.add_argument('--qkv_size_space', nargs='+', type=int, default=[180, 528])
    parser.add_argument('--head_num_space', nargs='+', type=int, default=[1, 12])
    parser.add_argument('--intermediate_size_space', nargs='+', type=int, default=[128, 1024])
    parser.add_argument('--mlm', action='store_true')
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

    hidden_step = 16
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

    # Test BERT-base time
    config = dict()
    config['sample_layer_num'] = 5
    config['sample_hidden_size'] = 564
    config['sample_intermediate_sizes'] = [1024] * config['sample_layer_num']
    config['sample_num_attention_heads'] = [12] * config['sample_layer_num']
    config['sample_qkv_sizes'] = [528] * config['sample_layer_num']
    config['vocab_size'] =  30522
    # Init write file
    print(save_dir)

    #config = BertConfig.from_dict(config)
    bert_config = BertConfig.from_pretrained(os.path.join(args.bert_model, 'config.json'))
    model = SuperTinyBertForPreTraining.from_scratch(args.bert_model, bert_config)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    device = 'cpu'
    model.to(device)
    model.eval()

    print("Doing first test")
    arch_cpu_time(model, config, args, save_dir, quant=True, config_keys=config_keys)
    print("First test done")

    #arch_cpu_time(model, config, args)
    print(layer_numbers)
    print(hidden_sizes[-1:-5:-1])
    print(ffn_sizes[-1:-5:-1])
    print(qkv_sizes[-1:-5:-1])
    print(hidden_sizes)
    print(ffn_sizes)
    print(qkv_sizes)

    for layer_num in layer_numbers:
        config = dict()
        config['vocab_size'] = 30522
        config['sample_layer_num'] = layer_num

        if not args.mlm:
            config['sample_num_attention_heads'] = [12] * layer_num

            for hidden_size in hidden_sizes:
                config['sample_hidden_size'] = hidden_size

                for ffn_size in ffn_sizes:
                    config['sample_intermediate_sizes'] = [ffn_size] * layer_num

                    for qkv_size in qkv_sizes:
                        config['sample_qkv_sizes'] = [qkv_size] * layer_num

                        arch_cpu_time(model, config, args, save_dir, quant=True, config_keys=config_keys)
        else:
            for head_size in head_sizes:
                config['sample_num_attention_heads'] = [head_size] * layer_num
                config['sample_qkv_sizes'] = [head_size * 64] * layer_num

                for hidden_size in hidden_sizes:
                    config['sample_hidden_size'] = hidden_size

                    for ffn_size in ffn_sizes:
                        config['sample_intermediate_sizes'] = [ffn_size] * layer_num

                        arch_cpu_time(model, config, args, save_dir, quant=True, config_keys=config_keys)

        print(f"CURRENT PROP DONE: {(layer_num/len(layer_numbers)) * 100}%")
