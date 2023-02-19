import os
import argparse
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import celpnet
from dataset import CELPNetDataset

nb_features = 36
nb_used_features = 20

parser = argparse.ArgumentParser()

parser.add_argument('model', type=str, help='CELPNet model')
parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('output', type=str, help='path to output file (16-bit PCM)')

parser.add_argument('--cuda-visible-devices', type=str, help="comma separates list of cuda visible device indices, default: CUDA_VISIBLE_DEVICES", default=None)


model_group = parser.add_argument_group(title="model parameters")
model_group.add_argument('--cond-size', type=int, help="first conditioning size, default: 256", default=256)

args = parser.parse_args()

if args.cuda_visible_devices != None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices


features_file = args.features
signal_file = args.output



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoint = torch.load(args.model, map_location='cpu')

model = celpnet.CELPNet(*checkpoint['model_args'], **checkpoint['model_kwargs'])


model.load_state_dict(checkpoint['state_dict'], strict=False)

features = np.reshape(np.memmap(features_file, dtype='float32', mode='r'), (1, -1, nb_features))
lpc = features[:,4-1:-1,nb_used_features:]
features = features[:, :, :nb_used_features]
periods = np.round(50*features[:,:,nb_used_features-2]+100).astype('int')

nb_frames = features.shape[1]
#nb_frames = 1000

if __name__ == '__main__':
    model.to(device)
    features = torch.tensor(features).to(device)
    lpc = torch.tensor(lpc).to(device)
    periods = torch.tensor(periods).to(device)
    
    sig, _ = model(features, periods, nb_frames - 4, lpc=lpc)
    
    sig = sig.detach().numpy().flatten()
    mem = 0
    for i in range(len(sig)):
        sig[i] += 0.85*mem
        mem = sig[i]

    pcm = np.round(32768*np.clip(sig, a_max=.99, a_min=-.99)).astype('int16')
    pcm.tofile(signal_file)
