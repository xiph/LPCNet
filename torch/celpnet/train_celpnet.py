import os
import argparse
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import celpnet
from dataset import CELPNetDataset

parser = argparse.ArgumentParser()

parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('signal', type=str, help='path to signal file in .s16 format')
parser.add_argument('output', type=str, help='path to output folder')

parser.add_argument('--suffix', type=str, help="model name suffix", default="")
parser.add_argument('--cuda-visible-devices', type=str, help="comma separates list of cuda visible device indices, default: CUDA_VISIBLE_DEVICES", default=None)


model_group = parser.add_argument_group(title="model parameters")
model_group.add_argument('--cond-size', type=int, help="first conditioning size, default: 256", default=256)
model_group.add_argument('--has-gain', action='store_true', help="use gain-shape network")
model_group.add_argument('--has-lpc', action='store_true', help="use LPC")
model_group.add_argument('--passthrough-size', type=int, help="state passing through in addition to audio, default: 0", default=0)
model_group.add_argument('--gamma', type=float, help="Use A(z/gamma), default: 1", default=None)

training_group = parser.add_argument_group(title="training parameters")
training_group.add_argument('--batch-size', type=int, help="batch size, default: 512", default=512)
training_group.add_argument('--lr', type=float, help='learning rate, default: 1e-3', default=1e-3)
training_group.add_argument('--epochs', type=int, help='number of training epochs, default: 20', default=20)
training_group.add_argument('--sequence-length', type=int, help='sequence length, default: 15', default=15)
training_group.add_argument('--lr-decay', type=float, help='learning rate decay factor, default: 1e-4', default=1e-4)
training_group.add_argument('--initial-checkpoint', type=str, help='initial checkpoint to start training from, default: None', default=None)

args = parser.parse_args()

if args.cuda_visible_devices != None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

# checkpoints
checkpoint_dir = os.path.join(args.output, 'checkpoints')
checkpoint = dict()
os.makedirs(checkpoint_dir, exist_ok=True)


# training parameters
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
sequence_length = args.sequence_length
lr_decay = args.lr_decay

adam_betas = [0.9, 0.99]
adam_eps = 1e-8
features_file = args.features
signal_file = args.signal

# model parameters
cond_size  = args.cond_size


checkpoint['batch_size'] = batch_size
checkpoint['lr'] = lr
checkpoint['lr_decay'] = lr_decay
checkpoint['epochs'] = epochs
checkpoint['sequence_length'] = sequence_length
checkpoint['adam_betas'] = adam_betas


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoint['model_args']    = ()
checkpoint['model_kwargs']  = {'cond_size': cond_size, 'has_gain': args.has_gain, 'has_lpc': args.has_lpc, 'passthrough_size': args.passthrough_size, 'gamma': args.gamma}
print('has_lpc', args.has_lpc)
print(checkpoint['model_kwargs'])
model = celpnet.CELPNet(*checkpoint['model_args'], **checkpoint['model_kwargs'])

#model = celpnet.CELPNet()
#model = nn.DataParallel(model)

if type(args.initial_checkpoint) != type(None):
    checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

checkpoint['state_dict']    = model.state_dict()


dataset = CELPNetDataset(features_file, signal_file, sequence_length=sequence_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)


optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps)


# learning rate scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda x : 1 / (1 + lr_decay * x))

spec320 = celpnet.new_specgram(320, device)
spec640 = celpnet.new_specgram(640, device)

states = None

fweight320 = celpnet.gen_weight(320//2, device)
fweight640 = celpnet.gen_weight(640//2, device)

if __name__ == '__main__':
    model.to(device)

    for epoch in range(1, epochs + 1):

        running_loss320 = 0
        running_loss640 = 0
        running_lsd320 = 0
        running_lsd640 = 0
        running_cont_loss = 0
        running_loss = 0

        print(f"training epoch {epoch}...")
        with tqdm.tqdm(dataloader, unit='batch') as tepoch:
            for i, (features, periods, target, lpc) in enumerate(tepoch):
                optimizer.zero_grad()
                features = features.to(device)
                lpc = lpc.to(device)
                periods = periods.to(device)
                target = target.to(device)
                #nb_pre = random.randrange(1, 6)
                nb_pre = int(np.minimum(8, 1-2*np.log(np.random.rand())))
                pre = target[:, :nb_pre*160]
                sig, states = model(features, periods, target.size(1)//160 - nb_pre, pre=pre, states=states, lpc=lpc)
                sig = torch.cat([pre, sig], -1)

                T320, T320m = spec320(target)
                S320, S320m = spec320(sig)
                T640, T640m = spec640(target)
                S640, S640m = spec640(sig)

                loss320 = celpnet.spec_loss(T320, S320, T320m, 0.3, fweight320)
                loss640 = celpnet.spec_loss(T640, S640, T640m, 0.3, fweight640)
                lsd320 = celpnet.lsd_loss(T320m, S320m, fweight320)
                lsd640 = celpnet.lsd_loss(T640m, S640m, fweight640)
                cont_loss = celpnet.sig_l1(target[:, nb_pre*160:nb_pre*160+40], sig[:, nb_pre*160:nb_pre*160+40])
                loss = loss320 + loss640 + .05*lsd320 + .05*lsd640 + 10*cont_loss
                #loss = lsd320

                loss.backward()
                optimizer.step()
                
                #model.clip_weights()
                
                scheduler.step()

                running_loss320 += loss320.detach().cpu().item()
                running_loss640 += loss640.detach().cpu().item()
                running_lsd320 += lsd320.detach().cpu().item()
                running_lsd640 += lsd640.detach().cpu().item()
                running_cont_loss += cont_loss.detach().cpu().item()

                running_loss += loss.detach().cpu().item()
                tepoch.set_postfix(loss=f"{running_loss/(i+1):8.5f}",
                                   loss320=f"{running_loss320/(i+1):8.5f}",
                                   loss640=f"{running_loss640/(i+1):8.5f}",
                                   lsd320=f"{running_lsd320/(i+1):8.5f}",
                                   lsd640=f"{running_lsd640/(i+1):8.5f}",
                                   cont_loss=f"{running_cont_loss/(i+1):8.5f}",
                                   )

        # save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'celpnet{args.suffix}_{epoch}.pth')
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['loss'] = running_loss / len(dataloader)
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, checkpoint_path)
