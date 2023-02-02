import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import celpnet
from dataset import CELPNetDataset

lr = 0.001
adam_betas = [0.9, 0.99]
adam_eps = 1e-8
lr_decay_factor = 2e-5
epochs = 20
batch_size = 512
features_file = 'features56.f32'
signal_file = 'data56.s16'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = celpnet.CELPNet()
#model = nn.DataParallel(model)

dataset = CELPNetDataset(features_file, signal_file)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)


optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps)


# learning rate scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda x : 1 / (1 + lr_decay_factor * x))

if __name__ == '__main__':
    model.to(device)

    for epoch in range(1, epochs + 1):

        running_l1_loss = 0

        print(f"training epoch {epoch}...")
        with tqdm.tqdm(dataloader, unit='batch') as tepoch:
            for i, (features, periods, signal) in enumerate(tepoch):
                optimizer.zero_grad()
                features = features.to(device)
                periods = periods.to(device)
                signal = signal.to(device)
                pre = signal[:, :3*160]
                sig = model(features, pre, signal.size(1)//160 - 3)
                sig = torch.cat([pre, sig], -1)
                loss = celpnet.sig_l1(signal, sig)
                loss.backward()
                optimizer.step()
                
                #model.clip_weights()
                
                scheduler.step()

                running_l1_loss += loss
                tepoch.set_postfix(running_l1_loss=running_l1_loss/(i+1))
