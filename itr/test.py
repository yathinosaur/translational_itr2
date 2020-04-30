import torch
import os
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer

from config import preEncDec as config
from model import TranslationModel

src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tgt_tokenizer.bos_token = '<s>'
tgt_tokenizer.eos_token = '</s>'

#hidden_size and intermediate_size are both wrt all the attention heads. 
#Should be divisible by num_attention_heads
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

decoder_config = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
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
                            layer_norm_eps=1e-12,
                            is_decoder=True)

#Create encoder and decoder embedding layers.
encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)
decoder_embeddings = torch.nn.Embedding(tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=tgt_tokenizer.pad_token_id)

encoder = BertModel(encoder_config)
encoder.set_input_embeddings(encoder_embeddings.cpu())

decoder = BertForMaskedLM(decoder_config)
decoder.set_input_embeddings(decoder_embeddings.cpu())

input_dirs = config.model_output_dirs

suffix = "pytorch_model.bin"
decoderPath = os.path.join(input_dirs['decoder'], suffix)
encoderPath = os.path.join(input_dirs['encoder'], suffix)

decoder_state_dict = torch.load(decoderPath)
encoder_state_dict = torch.load(encoderPath)
decoder.load_state_dict(decoder_state_dict)
encoder.load_state_dict(encoder_state_dict)
model = TranslationModel(encoder, decoder, None, None, config)
model.cpu()


model.eval()

def get_ids(tokens):
    token_ids = src_tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [1] * len(token_ids)
    return token_ids, segment_ids


while(True):
    data = str(input("> "))
    data_tokens = src_tokenizer.tokenize(data)
    input_ids, segment_ids = get_ids(data_tokens)
    tokens_tensor = torch.LongTensor([input_ids])
    segment_tensor = torch.LongTensor([segment_ids])

    resp = model(tokens_tensor, segment_tensor)[1]
    print(resp.size())
    output = ''
    for i in range(len(input_ids)):
        resp_index = torch.argmax(resp[0][i]).item()
        resp_token = tgt_tokenizer.convert_ids_to_tokens([resp_index])[0]
        output += str(resp_index) + " "
    print(output)
    OutFile = open("output.txt", "a")
    OutFile.write(output)
    OutFile.close()