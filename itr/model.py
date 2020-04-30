import torch
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer

from easydict import EasyDict as ED

import pytorch_lightning as pl

import os.path

#class TranslationModel(torch.nn.Module):
class TranslationModel(pl.LightningModule):
    #def __init__(self, encoder, decoder):
    def __init__(self, encoder, decoder, train_loader, eval_loader, config):

        super().__init__() 
        
        self.config = config

        #Creating encoder and decoder with their respective embeddings.
        self.encoder = encoder
        self.decoder = decoder
        self.tran_loader = train_loader
        self.eval_loader = eval_loader
        self.device = torch.device("cpu")

    def forward(self, encoder_input_ids, decoder_input_ids):

        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    masked_lm_labels=decoder_input_ids)

        return loss, logits

    def save(self, tokenizers, output_dirs):
        from train_util import save_model

        save_model(self.encoder, output_dirs.encoder)
        save_model(self.decoder, output_dirs.decoder)
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        loss, logits = self.forward(data, target)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        loss, logits = self.forward(data, target)
        return {'val_loss': loss}

    def train_dataloader(self):
        return self.tran_loader
        
    def val_dataloader(self):
        return self.eval_loader
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

def get_tokenizer():
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tgt_tokenizer.bos_token = '<s>'
    tgt_tokenizer.eos_token = '</s>'

    
    tokenizers = ED({'src': src_tokenizer, 'tgt': tgt_tokenizer})
    return tokenizers

def build_model(config, train_loader, eval_loader):
    
    

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

    if(os.listdir(input_dirs['decoder']) and os.listdir(input_dirs['encoder'])):
        suffix = "pytorch_model.bin"
        decoderPath = os.path.join(input_dirs['decoder'], suffix)
        encoderPath = os.path.join(input_dirs['encoder'], suffix)
        
        decoder_state_dict = torch.load(decoderPath)
        encoder_state_dict = torch.load(encoderPath)
        decoder.load_state_dict(decoder_state_dict)
        encoder.load_state_dict(encoder_state_dict)
        model = TranslationModel(encoder, decoder, train_loader, eval_loader, config)
        model.cpu()
        return model

    #model = TranslationModel(encoder, decoder)
    model = TranslationModel(encoder, decoder, train_loader, eval_loader, config)
    model.cpu()


    tokenizers = ED({'src': src_tokenizer, 'tgt': tgt_tokenizer})
    #return model, tokenizers
    return model









