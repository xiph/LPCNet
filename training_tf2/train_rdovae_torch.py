import os

import torch
import tqdm

from rdovae_torch import RDOVAE, distortion_loss, hard_rate_estimate, soft_rate_estimate
from rdovae_dataset_torch import RDOVAEDataset

# checkpoints
checkpoint_dir = './checkpoints'
checkpoint = dict()
os.makedirs(checkpoint_dir, exist_ok=True)

# training parameters
batch_size = 32
lr = 1e-4
epochs = 20
sequence_length = 1024
log_interval = 10

checkpoint['batch_size'] = batch_size
checkpoint['lr'] = lr
checkpoint['epochs'] = epochs
checkpoint['sequence_length'] = sequence_length

# device
device = torch.device("cpu")

# model parameters
cond_size  = 128
cond_size2 = 256
num_features = 20
latent_dim = 80
quant_levels = 40


# training data
feature_file = 'input/features_small.f32'

# model
model = RDOVAE(num_features, latent_dim, quant_levels, cond_size, cond_size2)

checkpoint['model_args']    = (num_features, latent_dim, quant_levels, cond_size, cond_size2)
checkpoint['model_kwargs']  = dict()
checkpoint['state_dict']    = model.state_dict()


# dataloader
dataset = RDOVAEDataset(feature_file, 1024, 20, 36, 0.0007, 0.002, 2)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

checkpoint['dataset_args'] = (feature_file, 1024, 20, 36, 0.0007, 0.002, 2)
checkpoint['dataset_kwargs'] = dict()

# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=lr)


if __name__ == '__main__':

    # push model to device
    model.to(device)

    # training loop

    for epoch in range(1, epochs + 1):

        print(f"training epoch {epoch}...")

        # running stats
        running_rate_loss       = 0
        running_soft_dist_loss  = 0
        running_hard_dist_loss  = 0
        running_total_loss      = 0
        running_rate_metric     = 0
        previous_total_loss     = 0

        with tqdm.tqdm(dataloader, unit='batch') as tepoch:
            for i, (features, rate_lambda, q_ids) in enumerate(tepoch):
                optimizer.zero_grad()

                # run model
                model_output = model(features.to(device), q_ids.to(device), rate_lambda.to(device))

                # collect outputs
                z                   = model_output['z']
                outputs_hard_quant  = model_output['outputs_hard_quant']
                outputs_soft_quant  = model_output['outputs_soft_quant']
                statistical_model   = model_output['statistical_model']

                # rate loss
                hard_rate = hard_rate_estimate(z, statistical_model['r_hard'], statistical_model['theta_hard'], reduce=False)
                soft_rate = soft_rate_estimate(z, statistical_model['r_soft'], reduce=False)
                rate_loss = torch.mean(rate_lambda.squeeze(-1) * (hard_rate + soft_rate))
                hard_rate_metric = torch.mean(hard_rate)

                ## distortion losses

                # hard quantized decoder input
                distortion_loss_hard_quant = torch.zeros_like(rate_loss)
                for dec_features, start, stop in outputs_hard_quant:
                    distortion_loss_hard_quant += distortion_loss(features[..., start : stop, :], dec_features) / len(outputs_hard_quant)

                # soft quantized decoder input
                distortion_loss_soft_quant = torch.zeros_like(rate_loss)
                for dec_features, start, stop in outputs_soft_quant:
                    distortion_loss_soft_quant += distortion_loss(features[..., start : stop, :], dec_features) / len(outputs_soft_quant)

                # total loss
                total_loss = rate_loss + (distortion_loss_hard_quant + distortion_loss_soft_quant) / 2

                total_loss.backward()

                optimizer.step()

                # collect running stats
                running_hard_dist_loss  += float(distortion_loss_hard_quant.detach().cpu())
                running_soft_dist_loss  += float(distortion_loss_soft_quant.detach().cpu())
                running_rate_loss       += float(rate_loss.detach().cpu())
                running_rate_metric     += float(hard_rate_metric.detach().cpu())
                running_total_loss      += float(total_loss.detach().cpu())

                if (i + 1) % log_interval == 0:
                    current_loss = (running_total_loss - previous_total_loss) / log_interval
                    tepoch.set_postfix(
                        current_loss=current_loss,
                        total_loss=running_total_loss / (i + 1),
                        dist_hq=running_hard_dist_loss / (i + 1),
                        dist_sq=running_soft_dist_loss / (i + 1),
                        rate_loss=running_rate_loss / (i + 1),
                        rate=running_rate_metric / (i + 1)
                    )
                    previous_total_loss = running_total_loss

        # save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch _{epoch}.pth')
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['loss'] = running_total_loss / len(dataloader)
        torch.save(checkpoint, checkpoint_path)
