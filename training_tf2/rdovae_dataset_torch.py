
import torch
import numpy as np

class RDOVAEDataset(torch.utils.data.Dataset):
    def __init__(self,
                feature_file,
                sequence_length,
                num_used_features=20,
                num_features=36,
                lambda_min=0.0007,
                lambda_max=0.002,
                enc_stride=2):
        
        self.sequence_length = sequence_length
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.enc_stride = enc_stride

        if sequence_length % enc_stride:
            raise ValueError(f"RDOVAEDataset.__init__: enc_stride {enc_stride} does not divide sequence length {sequence_length}")
        
        self.features = np.reshape(np.fromfile(feature_file, dtype=np.float32), (-1, num_features))
        self.features = self.features[:, :num_used_features]
        self.num_sequences = self.features.shape[0] // sequence_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        features = self.features[index * self.sequence_length: (index + 1) * self.sequence_length, :]
        rate_lambda = np.random.uniform(self.lambda_min, self.lambda_max, (1, 1)).astype(np.float32)
        rate_lambda = np.repeat(rate_lambda, self.sequence_length // self.enc_stride, axis=0)
        q_ids = np.reshape(np.round(10 * np.log(rate_lambda / self.lambda_min)).astype(np.int64), (-1))

        return features, rate_lambda, q_ids

