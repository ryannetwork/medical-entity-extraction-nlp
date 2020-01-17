##Converting the MIMIC BERT weights

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert

import logging
logging.basicConfig(level=logging.INFO)

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)



if __name__ == '__main__':
    MIMIC_BERT_PRETRAINED_PATH = '/content/drive/Shared drives/BioNLP/project/medical_data/embeddings/MIMIC_BERT/'
    MIMIC_BERT_TF_PATH = MIMIC_BERT_PRETRAINED_PATH + "bert_model.ckpt"
    MIMIC_BERT_CONFIG_PATH = MIMIC_BERT_PRETRAINED_PATH + "config.json"
    MIMIC_BERT_PYTORCH_MODEL_PATH = MIMIC_BERT_PRETRAINED_PATH + "pytorch_model.bin"
    convert_tf_checkpoint_to_pytorch(MIMIC_BERT_TF_PATH,MIMIC_BERT_CONFIG_PATH,MIMIC_BERT_PYTORCH_MODEL_PATH)