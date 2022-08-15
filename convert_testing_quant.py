import torch
import collections
import transformers
import transformer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# autotiny bert config for v5
# note this is NOT the same as what is in model.config as that
# is the config for the entire supermodel 
atb_config = {"sample_layer_num": 5, 
    "sample_num_attention_heads": [12, 12, 12, 12, 12], 
    "sample_hidden_size": 528, 
    "sample_intermediate_sizes": [832, 832, 832, 832, 832], 
    "sample_qkv_sizes": [528, 528, 528, 528, 528]}


atb_cofig_class = transformer.modeling_super_kd.BertConfig(
    30522,
    hidden_size=528,
    num_hidden_layers=5,
    num_attention_heads=12,
    intermediate_size=832,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12)

atb_config_dict = {
    "attention_probs_dropout_prob": 0.1,
    "cell": {},
    "emb_size": 528, #?
    "finetuning_task": "mnli",
    "fix_config": {
        "sample_hidden_size": 528,
        "sample_intermediate_sizes": [
        832,
        832,
        832,
        832,
        832
        ],
        "sample_layer_num": 5,
        "sample_num_attention_heads": [
        12,
        12,
        12,
        12,
        12
        ],
        "sample_qkv_sizes": [
        528,
        528,
        528,
        528,
        528
        ]
    },
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 528,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 5,
    "num_labels": 3,
    "pre_trained": "",
    "qkv_size": 528,
    "structure": [],
    "training": "",
    "type_vocab_size": 2,
    "vocab_size": 30522
}


# function to convert autotinybert config to huggingface config
def atb_config_to_hf_config(atb_config, num_labels):
        
    hf_config = {
        "attention_probs_dropout_prob": 0.1,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.3.1",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 30522,
    }
    
    hf_config["hidden_size"] = atb_config["sample_hidden_size"]
    hf_config["intermediate_size"] = atb_config["sample_intermediate_sizes"][0]
    hf_config["num_hidden_layers"] = atb_config["sample_layer_num"]
    #hf_config["num_labels"] = super_bert.state_dict()["classifier.weight"].shape[0]
    hf_config["num_labels"] = num_labels


    return hf_config


# function to convert autotinybert model to huggingface model
def convert_atb_to_hf_model(super_bert, super_bert_config):

    num_labels = super_bert.state_dict()["classifier.weight"].shape[0]
    hf_config_dict = atb_config_to_hf_config(super_bert_config, num_labels)

    hf_config = transformers.BertConfig.from_dict(hf_config_dict)
    hf_bert = transformers.BertForSequenceClassification(hf_config)

    atb_to_hf_mapping = {}
    pos_embeddings = torch.Tensor([[i for i in range(512)]]).type(torch.int64)

    super_bert_state = super_bert.state_dict()
    hf_bert_state = hf_bert.state_dict()

    del super_bert_state['bert.dense_fit.weight']
    del super_bert_state['bert.dense_fit.bias']

    super_bert_state.update({'bert.embeddings.position_ids':pos_embeddings})
    super_bert_state.move_to_end('bert.embeddings.position_ids', last = False)

    for (atbert_name, atbert_tens), (hf_bert_name,hf_bert_tens) in zip(super_bert_state.items(),
                                                                     hf_bert_state.items()):

        atb_to_hf_mapping[atbert_name] = hf_bert_name

    new_state_dict = collections.OrderedDict()

    for hf_tens, (atbert_name, atbert_tens) in zip(hf_bert.state_dict().values(), 
                                                   super_bert_state.items()):
        
        hf_name = atb_to_hf_mapping[atbert_name]
        
        if len(hf_tens.shape) == 2:
            (a, b) = hf_tens.shape
            convert_tens = atbert_tens[:a, :b]
        else:
            (a,) = hf_tens.shape
            convert_tens = atbert_tens[:a]
            
        new_state_dict[hf_name] = convert_tens

    hf_bert.load_state_dict(new_state_dict)

    return hf_bert


def load_testing_model(model_path="./output/test_fp/model-fp-20ms-v5.pt"):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    converted_model = convert_atb_to_hf_model(model, atb_config)

    return converted_model