import torch
import os
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer

from config import preEncDec as config

src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tgt_tokenizer.bos_token = '<s>'
tgt_tokenizer.eos_token = '</s>'

encoder_config = BertConfig(vocab_size=src_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12)
encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)

encoder = BertModel(encoder_config)
encoder.set_input_embeddings(encoder_embeddings.cpu())

input_dirs = config.model_output_dirs
suffix = "pytorch_model.bin"
encoderPath = os.path.join(input_dirs['encoder'], suffix)
encoder_state_dict = torch.load(encoderPath)
encoder.load_state_dict(encoder_state_dict)

encoder.eval()

while(True):
    data = input("> ")
    print(encoder(data))