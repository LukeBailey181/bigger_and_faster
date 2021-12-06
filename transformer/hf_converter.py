import transformers
import torch
import collections

def convert_atb_to_hf_model(super_bert):

    hf_config_dict = convert_atb_to_hf_config(super_bert)

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

    for atbert_name, atbert_tens in super_bert_state.items():
        
        hf_name = atb_to_hf_mapping[atbert_name]
        new_state_dict[hf_name] = atbert_tens

    hf_bert.load_state_dict(new_state_dict)

    return hf_bert


def convert_atb_to_hf_config(super_bert):

    atb_config = super_bert.config.to_dict()

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
    hf_config["num_labels"] = super_bert.state_dict()["classifier.weight"].shape[0]

    return hf_config
