import torch
import torch.nn.functional as F
import os
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer

from config import preEncDec as config
from model import TranslationModel


class Model():
        
    src_tokenizer = None
    tgt_tokenizer = None
    model = None

    def __init__(self):
        self.src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.tgt_tokenizer.bos_token = '<s>'
        self.tgt_tokenizer.eos_token = '</s>'

        #hidden_size and intermediate_size are both wrt all the attention heads. 
        #Should be divisible by num_attention_heads
        encoder_config = BertConfig(vocab_size=self.src_tokenizer.vocab_size,
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

        decoder_config = BertConfig(vocab_size=self.tgt_tokenizer.vocab_size,
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
        encoder_embeddings = torch.nn.Embedding(self.src_tokenizer.vocab_size, config.hidden_size, padding_idx=self.src_tokenizer.pad_token_id)
        decoder_embeddings = torch.nn.Embedding(self.tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=self.tgt_tokenizer.pad_token_id)

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
        self.model = TranslationModel(encoder, decoder, None, None, self.tgt_tokenizer, config)
        self.model.cpu()


        #model.eval()
        self.model.encoder.eval()
        self.model.decoder.eval()

def get_ids(tokens):
    token_ids = src_tokenizer.convert_tokens_to_ids(tokens)
    #segment_ids = [1] * len(token_ids)
    segment_ids = [src_tokenizer.mask_token_id]
    
    return token_ids, segment_ids

def test(data, test_model):
    """
    data_encode = torch.LongTensor([test_model.src_tokenizer.encode(data)])
    masked_decoder_input_ids = torch.LongTensor([test_model.tgt_tokenizer.encode(['MASK'])])
    encoder_output = test_model.model.encoder(data_encode)[0]
    #decoder_output = model.model.decoder(masked_decoder_input_ids, encoder_hidden_states=encoder_output)[0]
    #decoder_output = model.model.decoder.generate(encoder_output)
    output = test_model.model.decoder(data_encode, encoder_hidden_states=encoder_output)
    print(output.size())
    #print(decoder_output.size())
    resp = torch.topk(F.softmax(output[0]), k=1).indices
    print(test_model.tgt_tokenizer.decode(resp))
    """
    #data_tokens = src_tokenizer.tokenize(data)
    input_ids = test_model.src_tokenizer.encode(data)
    #segment_ids = [test_model.tgt_tokenizer.bos_token_id] + [test_model.tgt_tokenizer.mask_token_id] * 10 + [test_model.tgt_tokenizer.eos_token_id]
    segment_ids = [test_model.tgt_tokenizer.bos_token_id] + test_model.tgt_tokenizer.encode('Hello') + [test_model.tgt_tokenizer.eos_token_id]
    #input_ids, segment_ids = get_ids(data_tokens)
    tokens_tensor = torch.LongTensor([input_ids])
    segment_tensor = torch.LongTensor([segment_ids])
    print(tokens_tensor.size())
    print(segment_tensor.size())

    resp = test_model.model(tokens_tensor, segment_tensor, isEval=True)[0]
    print(resp.size())
    output = ''
    for i in range(resp.size()[1]):
        #print(resp[0][i])
        resp_indices = torch.topk(F.softmax(resp[0][i], dim=0), k=5).indices
        for resp_index in resp_indices:
            resp_token = test_model.tgt_tokenizer.decode([resp_index])[0]
            output += str(resp_token) + " "
        output += "\n"
    print(output)

def main():
    model = Model()
    while(True):
        data = str(input("> "))
        test(data, model)

if __name__ == '__main__':
    main()