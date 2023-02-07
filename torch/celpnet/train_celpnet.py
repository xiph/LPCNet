import os
import argparse

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
checkpoint['model_kwargs']  = {'cond_size': cond_size}

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

spec1 = celpnet.new_specgram(320, 0.5, device)
spec2 = celpnet.new_specgram(320, 0, device)
spec1b = celpnet.new_specgram(640, 0.5, device)
spec2b = celpnet.new_specgram(640, 0, device)

if __name__ == '__main__':
    model.to(device)

    for epoch in range(1, epochs + 1):

        running_loss1 = 0
        running_loss2 = 0
        running_loss1b = 0
        running_loss2b = 0
        running_loss = 0

        print(f"training epoch {epoch}...")
        with tqdm.tqdm(dataloader, unit='batch') as tepoch:
            for i, (features, periods, signal) in enumerate(tepoch):
                optimizer.zero_grad()
                features = features.to(device)
                periods = periods.to(device)
                signal = signal.to(device)
                pre = signal[:, :3*160]
                sig = model(features, periods, pre, signal.size(1)//160 - 3)
                sig = torch.cat([pre, sig], -1)

                #loss = celpnet.sig_l1(signal, sig)
                loss1 = celpnet.spec_l1(spec1, signal, sig)
                loss2 = celpnet.spec_l1(spec2, signal, sig)
                loss1b = celpnet.spec_l1(spec1b, signal, sig)
                loss2b = celpnet.spec_l1(spec2b, signal, sig)
                loss = 5*loss1 + loss2 + 5*loss1b + loss2b

                loss.backward()
                optimizer.step()
                
                #model.clip_weights()
                
                scheduler.step()

                running_loss1 += loss1.detach().cpu().item()
                running_loss2 += loss2.detach().cpu().item()
                running_loss1b += loss1b.detach().cpu().item()
                running_loss2b += loss2b.detach().cpu().item()

                running_loss += loss.detach().cpu().item()
                tepoch.set_postfix(loss=f"{running_loss/(i+1):8.5f}",
                                   loss1=f"{running_loss1/(i+1):8.5f}",
                                   loss2=f"{running_loss2/(i+1):8.5f}",
                                   loss1b=f"{running_loss1b/(i+1):8.5f}",
                                   loss2b=f"{running_loss2b/(i+1):8.5f}",
                                   )

        # save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'celpnet{args.suffix}_{epoch}.pth')
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['loss'] = running_loss / len(dataloader)
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, checkpoint_path)
