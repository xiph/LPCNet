""" Pytorch implementations of rate distortion optimized variational autoencoder """

import torch
from torch import nn
import torch.nn.functional as F

# Quantization and rate related utily functions

def soft_pvq(x, k):
    """ soft pyramid vector quantizer """

    # L2 normalization
    x_norm2 = x / (1e-15 + torch.norm(x, dim=-1, keepdim=True))
    

    with torch.no_grad():
        # quantization loop, no need to track gradients here
        x_norm1 = x / torch.sum(torch.abs(x), dim=-1, keepdim=True)

        # set initial scaling factor to k
        scale_factor = k
        x_scaled = scale_factor * x_norm1
        x_quant = torch.round(x_scaled)

        # we aim for ||x_quant||_L1 = k
        for _ in range(10):
            # remove signs and calculate L1 norm
            abs_x_quant = torch.abs(x_quant)
            abs_x_scaled = torch.abs(x_scaled)
            l1_x_quant = torch.sum(abs_x_quant, axis=-1)

            # increase, where target is to small and decrease, where target is too large
            plus  = 1.001 * torch.min((abs_x_quant + 0.5) / (abs_x_scaled + 1e-15), dim=-1).values
            minus = 0.999 * torch.max((abs_x_quant - 0.5) / (abs_x_scaled + 1e-15), dim=-1).values
            factor = torch.where(l1_x_quant > k, minus, plus)
            factor = torch.where(l1_x_quant == k, torch.ones_like(factor), factor)
            scale_factor = scale_factor * factor.unsqueeze(-1)

            # update x
            x_scaled = scale_factor * x_norm1
            x_quant = torch.round(x_quant)

    # L2 normalization of quantized x
    x_quant_norm2 = x_quant / (1e-15 + torch.norm(x_quant, dim=-1, keepdim=True))
    quantization_error = x_quant_norm2 - x_norm2

    # @JM: tensor.detach() is the pytorch equivalent of tf.stop_gradient
    return x_norm2 + quantization_error.detach()



def soft_rate_estimate(z, r, reduce=True):
    """ rate approximation with dependent theta Eq. (7)"""

    rate = torch.sum(
        - torch.log2((1 - r)/(1 + r) * r ** torch.abs(z) + 1e-6),
        dim=-1
    )

    if reduce:
        rate = torch.mean(rate)

    return rate

    

def hard_rate_estimate(z, r, theta, reduce=True):
    """ hard rate approximation """

    z_q = torch.round(z)
    p0 = 1 - r ** (0.5 + 0.5 * theta)
    alpha = torch.relu(1 - torch.abs(z_q)) ** 2
    rate = - torch.sum(
        (alpha * torch.log2(p0 * r ** torch.abs(z_q) + 1e-6) 
        + (1 - alpha) * torch.log2(0.5 * (1 - p0) * (1 - r) * r ** torch.abs(z_q) + 1e-6)),
        dim=-1
    )

    if reduce:
        rate = torch.mean(rate)

    return rate


def soft_dead_zone(x, dead_zone):
    """ approximates application of a dead zond to x """
    d = dead_zone * 0.05
    return x - d * torch.tanh(x / (0.1 + d))


def hard_quantize(x):
    """ round with copy gradient trick """
    return x + (torch.round(x) - x).detach()


def noise_quantize(x):
    """ simulates quantization with addition of random uniform noise """
    return x + (torch.rand_like(x) - 0.5)


# loss functions


def distortion_loss(y_true, y_pred):
    """ custom distortion loss for LPCNet features """
    
    if y_true.size(-1) != 20:
        raise ValueError('distortion loss is designed to work with 20 features')

    ceps_error   = y_pred[..., :18] - y_true[..., :18]
    pitch_error  = 2 * (y_pred[..., 18:19] - y_true[..., 18:19]) / (2 + y_true[..., 18:19])
    corr_error   = y_pred[..., 19:] - y_true[..., 19:]
    pitch_weight = torch.relu(y_true[..., 19:] + 0.5) ** 2

    loss = torch.mean(ceps_error ** 2 + (10/18) * torch.abs(pitch_error) * pitch_weight + (1/18) * corr_error ** 2)

    return loss



# RDOVAE module and submodules


class CoreEncoder(nn.Module):
    STATE_HIDDEN = 128
    FRAMES_PER_STEP = 2
    CONV_KERNEL_SIZE = 4
    
    def __init__(self, feature_dim, output_dim, statistical_model, cond_size, cond_size2, state_size=24):
        """ core encoder for RDOVAE
        
            Computes latents, initial states, and rate estimates from features and lambda parameter
        
        """

        super(CoreEncoder, self).__init__()

        # hyper parameters
        self.feature_dim        = feature_dim
        self.output_dim         = output_dim
        self.cond_size          = cond_size
        self.cond_size2         = cond_size2
        self.state_size         = state_size

        # shared statistical model
        self.statistical_model = statistical_model

        # derived parameters
        self.input_dim = self.FRAMES_PER_STEP * self.feature_dim + self.statistical_model.embedding_dim + 1
        self.conv_input_channels =  5 * cond_size + 3 * cond_size2

        # layers
        self.dense_1 = nn.Linear(self.input_dim, self.cond_size2)
        self.gru_1   = nn.GRU(self.cond_size2, self.cond_size, batch_first=True)
        self.dense_2 = nn.Linear(self.cond_size, self.cond_size2)
        self.gru_2   = nn.GRU(self.cond_size2, self.cond_size, batch_first=True)
        self.dense_3 = nn.Linear(self.cond_size, self.cond_size2)
        self.gru_3   = nn.GRU(self.cond_size2, self.cond_size, batch_first=True)
        self.dense_4 = nn.Linear(self.cond_size, self.cond_size)
        self.dense_5 = nn.Linear(self.cond_size, self.cond_size)
        self.conv1   = nn.Conv1d(self.conv_input_channels, self.output_dim, kernel_size=self.CONV_KERNEL_SIZE, padding='valid')

        self.state_dense_1 = nn.Linear(self.cond_size, self.STATE_HIDDEN)
        self.state_dense_2 = nn.Linear(self.STATE_HIDDEN, self.state_size)

        # state buffers for inference
        self.register_buffer('gru_1_state', torch.zeros((1, 1, cond_size)))
        self.register_buffer('gru_2_state', torch.zeros((1, 1, cond_size)))
        self.register_buffer('gru_3_state', torch.zeros((1, 1, cond_size)))
        self.register_buffer('conv_state_buffer', torch.zeros((1, self.conv_input_channels, self.CONV_KERNEL_SIZE-1))) # fixme

    def forward(self, features, q_id, distortion_lambda):
        
        # get statistical model
        statistical_model = self.statistical_model(q_id)

        # reshape features
        x = torch.reshape(features, (features.size(0), features.size(1)//2, 2 * features.size(2)))

        # prepare input
        x = torch.cat((x, statistical_model['quant_embedding'].detach(), distortion_lambda), dim=-1)

        batch = x.size(0)
        device = x.device

        # run encoding layer stack
        x1      = torch.tanh(self.dense_1(x))
        x2, _   = self.gru_1(x1, torch.zeros((1, batch, self.cond_size)).to(device))
        x3      = torch.tanh(self.dense_2(x2))
        x4, _   = self.gru_2(x3, torch.zeros((1, batch, self.cond_size)).to(device))
        x5      = torch.tanh(self.dense_3(x4))
        x6, _   = self.gru_3(x5, torch.zeros((1, batch, self.cond_size)).to(device))
        x7      = torch.tanh(self.dense_4(x6))
        x8      = torch.tanh(self.dense_5(x7))

        # concatenation of all hidden layer outputs
        x9 = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=-1)

        # pad for causal use @JM: pytorch uses channels first in convolutions
        x9 = F.pad(x9.permute(0, 2, 1), [3, 0])
        z = self.conv1(x9).permute(0, 2, 1)
        z = z * statistical_model['quant_scale']

        # initial states for decoding
        states = x6
        states = torch.tanh(self.state_dense_1(states))
        states = torch.tanh(self.state_dense_2(states))

        return z, states




class CoreDecoder(nn.Module):

    FRAMES_PER_STEP = 4

    def __init__(self, input_dim, output_dim, statistical_model, cond_size, cond_size2, state_size=24):
        """ core decoder for RDOVAE
        
            Computes features from latents, initial state, and quantization index
        
        """

        super(CoreDecoder, self).__init__()

        # hyper parameters
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.cond_size  = cond_size
        self.cond_size2 = cond_size2
        self.state_size = state_size

        # shared statistical model
        self.statistical_model = statistical_model

        # derived parameters
        self.input_size = self.input_dim + statistical_model.embedding_dim + self.state_size
        self.concat_size = 4 * self.cond_size + 4 * self.cond_size2

        # layers
        self.dense_1    = nn.Linear(self.input_size, cond_size2)
        self.dense_2    = nn.Linear(cond_size2, cond_size)
        self.dense_3    = nn.Linear(cond_size, cond_size2)
        self.gru_1      = nn.GRU(cond_size2, cond_size, batch_first=True)
        self.gru_2      = nn.GRU(cond_size, cond_size, batch_first=True)
        self.gru_3      = nn.GRU(cond_size, cond_size, batch_first=True)
        self.dense_4    = nn.Linear(cond_size, cond_size2)
        self.dense_5    = nn.Linear(cond_size2, cond_size2)

        self.output  = nn.Linear(self.concat_size, self.FRAMES_PER_STEP * self.output_dim)

    def forward(self, z, q_id, initial_state):

        batch_size = z.size(0)
        device = z.device

        # get statistical model
        statistical_model = self.statistical_model(q_id)

        # reverse scaling
        x = z / statistical_model['quant_scale']

        initial_state = torch.repeat_interleave(initial_state, x.size(1), 1)

        x = torch.cat((x, statistical_model['quant_embedding'].detach(), initial_state), dim=-1)

        # run decoding layer stack
        x1  = torch.tanh(self.dense_1(x))
        x2  = torch.tanh(self.dense_2(x1))
        x3  = torch.tanh(self.dense_3(x2))

        x4, _ = self.gru_1(x3, torch.zeros((1, batch_size, self.cond_size)).to(device))
        x5, _ = self.gru_2(x4, torch.zeros((1, batch_size, self.cond_size)).to(device))
        x6, _ = self.gru_3(x5, torch.zeros((1, batch_size, self.cond_size)).to(device))
        
        x7  = torch.tanh(self.dense_4(x6))
        x8  = torch.tanh(self.dense_5(x7))
        x9 = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=-1)


        # output layer and reshaping
        x10 = torch.tanh(self.output(x9))
        features = torch.reshape(x10, (x10.size(0), x10.size(1) * self.FRAMES_PER_STEP, x10.size(2) // self.FRAMES_PER_STEP))

        return features


class StatisticalModel(nn.Module):
    def __init__(self, quant_levels, latent_dim):
        """ Statistical model for latent space
        
            Computes scaling, deadzone, and r and theta 
        
        """

        super(StatisticalModel, self).__init__()

        # copy parameters
        self.latent_dim     = latent_dim
        self.quant_levels   = quant_levels
        self.embedding_dim  = 6 * latent_dim

        # quantization embedding
        self.quant_embedding    = nn.Embedding(quant_levels, self.embedding_dim)


    def forward(self, quant_ids):
        """ takes quant_ids and returns statistical model parameters"""

        x = self.quant_embedding(quant_ids)

        # @JM: theta_soft is not used anymore. Kick it out?
        quant_scale = F.softplus(x[..., 0 * self.latent_dim : 1 * self.latent_dim])
        dead_zone   = F.softplus(x[..., 1 * self.latent_dim : 2 * self.latent_dim])
        r_hard      = torch.sigmoid(x[..., 2 * self.latent_dim : 3 * self.latent_dim])
        theta_hard  = torch.sigmoid(x[..., 3 * self.latent_dim : 4 * self.latent_dim])
        r_soft      = torch.sigmoid(x[..., 4 * self.latent_dim : 5 * self.latent_dim])
        theta_soft  = torch.sigmoid(x[..., 5 * self.latent_dim : 6 * self.latent_dim])

        return {
            'quant_embedding'   : x,
            'quant_scale'       : quant_scale,
            'dead_zone'         : dead_zone,
            'r_hard'            : r_hard,
            'theta_hard'        : theta_hard,
            'r_soft'            : r_soft,
            'theta_soft'        : theta_soft
        }


class RDOVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, quant_levels, cond_size, cond_size2):

        super(RDOVAE, self).__init__()

        self.feature_dim  = feature_dim
        self.latent_dim   = latent_dim
        self.quant_levels = quant_levels
        self.cond_size    = cond_size
        self.cond_size2   = cond_size2
        
        # submodules encoder and decoder share the statistical model
        self.statistical_model = StatisticalModel(quant_levels, latent_dim)
        self.core_encoder = CoreEncoder(feature_dim, latent_dim, self.statistical_model, cond_size, cond_size2)
        self.core_decoder = CoreDecoder(latent_dim, feature_dim, self.statistical_model, cond_size, cond_size2)
    

    def get_decoder_chunks(self, z, mode='split', chunks_per_offset=4):
        """ calculates a set of matching encoder frame and feature frame ranges (CRITICAL!!!) """

        # we don't care for other configurations for the time being
        assert self.core_decoder.FRAMES_PER_STEP == 4 and self.core_encoder.FRAMES_PER_STEP == 2

        stride = self.core_decoder.FRAMES_PER_STEP // self.core_encoder.FRAMES_PER_STEP
        num_enc_frames = z.size(1)

        chunks = []
        
        if mode == 'split':
            for offset in range(stride):
                # calculate split points
                start = 2 - offset
                stop = num_enc_frames - offset
                length = (stop - start)
                split_points = [start + 2 * int(i * length / chunks_per_offset / 2) for i in range(chunks_per_offset)] + [stop]

                for i in range(chunks_per_offset):
                    # (enc_frame_start, enc_frame_stop, enc_frame_stride, stride, feature_frame_start, feature_frame_stop)
                    # encoder range(i, j, 2) maps to feature range(2 * i + 1 - 3, 2 * (j - 1) + 1 + 1) = range(2 * i - 2, 2 * j)
                    # provided that i - j = 0 mod 2
                    chunks.append({
                        'enc_start'         : split_points[i],
                        'enc_stop'          : split_points[i + 1],
                        'enc_stride'        : 2,
                        'features_start'    : 2 * split_points[i] - 2,
                        'features_stop'     : 2 * split_points[i + 1]
                    })

        else:
            raise ValueError(f"get_decoder_chunks: unkown mode {mode}")
            # ToDo: implement random splits
        
        return chunks

    def forward(self, features, q_id, distortion_lambda):

        # calculate statistical model from quantization ID
        statistical_model = self.statistical_model(q_id)

        # run encoder
        z, states = self.core_encoder(features, q_id, distortion_lambda)

        # quantization
        z_q = hard_quantize(z)
        z_n = noise_quantize(z)
        states_q = soft_pvq(states, 30)

        # decoder
        chunks = self.get_decoder_chunks(z)

        outputs = []
        for chunk in chunks:
            # decoder with hard quantized input
            z_dec_reverse       = torch.flip(z_q[..., chunk['enc_start'] : chunk['enc_stop'] : chunk['enc_stride'], :], [1])
            q_id_dec_reverse    = torch.flip(q_id[..., chunk['enc_start'] : chunk['enc_stop'] : chunk['enc_stride']], [1])
            dec_initial_state   = states_q[..., chunk['enc_stop'] - 1 : chunk['enc_stop'], :]
            features_reverse = self.core_decoder(z_dec_reverse, q_id_dec_reverse, dec_initial_state)
            outputs.append((torch.flip(features_reverse, [1]), chunk['features_start'], chunk['features_stop']))


            # decoder with soft quantized input
            z_dec_reverse       = torch.flip(z_n[..., chunk['enc_start'] : chunk['enc_stop'] : chunk['enc_stride'], :],  [1])
            features_reverse    = self.core_decoder(z_dec_reverse, q_id_dec_reverse, dec_initial_state)
            outputs.append((torch.flip(features_reverse, [1]), chunk['features_start'], chunk['features_stop']))          

        return {
            'outputs'           : outputs,
            'z'                 : z,
            'statistical_model' : statistical_model
        }

    def encode(self):
        pass

    def decode(self):
        pass

# rate estimates and distortion loss
# hard_rate = hard_rate_estimate(z, statistical_model['r_hard'], statistical_model['theta_hard'], reduce=False)
# soft_rate = soft_rate_estimate(z, statistical_model['r_soft'], reduce=False)
# distortion_loss = torch.mean(distortion_lambda * (hard_rate + soft_rate))

debug = False
if __name__ == "__main__" and debug:
    sm = StatisticalModel(40, 80)
    cenc = CoreEncoder(20, 80, sm, 128, 256, 24)
    cdec = CoreDecoder(80, 20, sm, 128, 256, 24)
    batch_size = 32
    num_frames = 512

    features = torch.randn((batch_size, num_frames, 20))
    distortion_lambda = torch.ones((batch_size, num_frames // 2, 1))
    q_ids = torch.ones((batch_size, num_frames // 2), dtype=torch.int64)

    z, states = cenc(features, q_ids, distortion_lambda)

    states_q = soft_pvq(states, 30)

    z_reverse = torch.flip(z, [1])
    q_ids_reverse = torch.flip(q_ids, [1])
    states_reverse = torch.flip(states, [1])

    dec_features = cdec(z_reverse[:, ::2, :], q_ids_reverse[:, ::2], states_reverse[:, 0:1, :])

    rdovae = RDOVAE(20, 80, 40, 128, 256)

    model_output = rdovae(features, q_ids, distortion_lambda)