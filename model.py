from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model
import torch.nn as nn


class RLIU(Wav2Vec2PreTrainedModel):
  def __init__(self,config, out_vocab=70):
    super().__init__(config)
    
    self.wav2vec2module = Wav2Vec2Model(config)
    self.linear = nn.Linear(768,768)
    self.classifier_vocab = nn.Linear(768,out_vocab)
    self.error_classifier = nn.Linear(768, 2)
    self.multihead_attention = nn.MultiheadAttention(embed_dim=768, num_heads=16, dropout=0.2, batch_first=True)
    self.compare_attention = nn.MultiheadAttention(embed_dim=768, num_heads=16, dropout=0.2, batch_first=True)
    
    self.post_init()
    self.embedding = nn.Embedding(out_vocab,768,padding_idx=out_vocab-1)
    self.error_classifier = nn.Linear(768, 2)

  def freeze_feature_extractor(self):
    self.wav2vec2module.feature_extractor._freeze_parameters()

  def forward(self, audio_input, canonical_phone_seq):
    #x1 = phoneme classifier branch
    #x2 = error classifier branch
    x1 = self.wav2vec2module(audio_input, attention_mask=None).last_hidden_state

    x2_1 = self.embedding(canonical_phone_seq) #out[B,T,756]
    x2,_ = self.compare_attention(x2_1,x1,x1) #out[B,T,756]
    x2 = x2_1 - x2

    x1,_ = self.multihead_attention(x1, x2, x2)
    x1 = self.classifier_vocab(x1) #out vocab

    x2 = self.error_classifier(x2) #out error bool

    return x1,x2




