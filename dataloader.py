import json
from torch.utils.data import Dataset
import torch, librosa, gc
import ast
from transformers import Wav2Vec2FeatureExtractor
import numpy as np

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)


with open('vocab.json') as f:
    dict_vocab = json.load(f)

def text_to_tensor(string_text):
    text = string_text
    text = text.split(" ")
    text_list = []
    for idex in text:
        text_list.append(dict_vocab[idex])
    return text_list


class L2_Dataset(Dataset):
  def __init__(self,data,feature_extractor=feature_extractor):
    self.len = len(data)
    self.path = data['Label']
    self.canonical = list(data['Canonical'])
    self.transcript = list(data['Transcript'])
    self.error = list(data['Error'])
    self.feature_extractor = feature_extractor
  
  def __getitem__(self,idx):
    waveform, _ = librosa.load("/content/L2/train/audio/" + self.path[idx] + ".wav", sr=16000)
    waveform = self.feature_extractor(waveform, sampling_rate = 16000, return_tensors="pt")
    waveform = waveform.input_values

    canonical = text_to_tensor(self.canonical[idx])
    transcript = text_to_tensor(self.transcript[idx])

    error = ast.literal_eval(self.error[idx])

    return waveform.squeeze(0),canonical,transcript,error



  def __len__(self):
      return self.len
  
def collate_fn(batch):
  with torch.no_grad():

      #Get max len
      max = [-1]*4
      for row in batch:
        print(row[0])
        if row[0].shape[0] > max[0]:
          max[0]=row[0].shape[0]
        if len(row[1]) > max[1]:
          max[1] = len(row[1])
        if len(row[2]) > max[2]:
          max[2] = len(row[2])
        if len(row[3]) > max[3]:
          max[3] = len(row[3])
      print(max)


      cols = {'waveform':[], 'canonical':[], 'transcript':[], 'error':[], 'outputlengths':[]}
      for row in batch:
        padded_wav = np.concatenate([row[0], np.zeros(max[0] - row[0].shape[0])])
        cols['waveform'].append(padded_wav)

        row[1].extend([69] * (max[1] - len(row[1])))
        cols['canonical'].append(row[1])

        cols['outputlengths'].append(len(row[2]))
        row[2].extend([69] * (max[2] - len(row[2])))
        cols['transcript'].append(row[2])


        row[3].extend([2] * (max[3] - len(row[3])))
        cols['error'].append(row[3])

      cols['waveform']        = torch.tensor(cols['waveform'])
      cols['canonical']      = torch.tensor(cols['canonical'],  dtype=torch.long)
      cols['transcript']      = torch.tensor(cols['transcript'],  dtype=torch.long)
      cols['error']           = torch.tensor(cols['error'],       dtype=torch.long)
      cols['outputlengths']   = torch.tensor(cols['outputlengths'], dtype=torch.long)
      
  return cols['waveform'], cols['canonical'], cols['transcript'], cols['error'], cols['outputlengths']
      



    

