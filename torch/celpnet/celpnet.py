import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def new_specgram(N, alpha, device):
    x = np.arange(N, dtype='float32')
    w = np.sin(.5*np.pi*np.sin((x+.5)/N*np.pi)**2)
    w = torch.tensor(w).to(device)
    def compute_specgram(x):
        X = torch.stft(x, N, hop_length=N//4, return_complex=True, center=False, window=w)
        if alpha == 0:
            return 20*torch.log10(1e-5+torch.abs(X))
        else:
            return (1./alpha)*(1e-15+torch.abs(X))**alpha

    return compute_specgram

def sig_l1(y_true, y_pred):
    return torch.mean(abs(y_true-y_pred))

def spec_l1(spec, y_true, y_pred):
    return torch.mean(abs(spec(y_true)-spec(y_pred)))

# weight initialization and clipping
def init_weights(module):
    if isinstance(module, nn.GRU):
        for p in module.named_parameters():
            if p[0].startswith('weight_hh_'):
                nn.init.orthogonal_(p[1])

def gen_phase_embedding(periods, frame_size):
    device = periods.device
    batch_size = periods.size(0)
    nb_frames = periods.size(1)
    w0 = 2*torch.pi/periods
    w0_shift = torch.cat([2*torch.pi*torch.rand((batch_size, 1)).to(device)/frame_size, w0[:,:-1]], 1)
    cum_phase = frame_size*torch.cumsum(w0_shift, 1)
    fine_phase = w0[:,:,None]*torch.broadcast_to(torch.arange(frame_size).to(device), (batch_size, nb_frames, frame_size))
    embed = torch.unsqueeze(cum_phase, 2) + fine_phase
    embed = torch.reshape(embed, (batch_size, -1))
    return torch.cos(embed), torch.sin(embed)


class CELPNetCond(nn.Module):
    def __init__(self, feature_dim=20, cond_size=256, pembed_dims=64):
        super(CELPNetCond, self).__init__()

        self.feature_dim = feature_dim
        self.cond_size = cond_size

        self.pembed = nn.Embedding(256, pembed_dims)
        self.fdense1 = nn.Linear(self.feature_dim + pembed_dims, self.cond_size)
        self.fconv1 = nn.Conv1d(self.cond_size, self.cond_size, kernel_size=3, padding='valid')
        self.fconv2 = nn.Conv1d(self.cond_size, self.cond_size, kernel_size=3, padding='valid')
        self.fdense2 = nn.Linear(self.cond_size, self.cond_size)

        self.apply(init_weights)

    def forward(self, features, period):
        p = self.pembed(period)
        features = torch.cat((features, p), -1)
        tmp = torch.tanh(self.fdense1(features))
        tmp = tmp.permute(0, 2, 1)
        tmp = torch.tanh(self.fconv1(tmp))
        tmp = torch.tanh(self.fconv2(tmp))
        tmp = tmp.permute(0, 2, 1)
        tmp = torch.tanh(self.fdense2(tmp))
        return tmp

class CELPNetSub(nn.Module):
    def __init__(self, subframe_size=40, nb_subframes=4, cond_size=256, has_gain=False):
        super(CELPNetSub, self).__init__()

        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.cond_size = cond_size
        self.has_gain = has_gain
        
        print("has_gain:", self.has_gain)
        
        gain_param = 1 if self.has_gain else 0

        self.sig_dense1 = nn.Linear(3*self.subframe_size+self.cond_size+gain_param, self.cond_size)
        self.sig_dense2 = nn.Linear(self.cond_size, self.cond_size)
        self.gru1 = nn.GRUCell(self.cond_size, self.cond_size)
        self.gru2 = nn.GRUCell(self.cond_size, self.cond_size)
        self.gru3 = nn.GRUCell(self.cond_size, self.cond_size)
        self.sig_dense_out = nn.Linear(self.cond_size, self.subframe_size)
        if self.has_gain:
            self.gain_dense_out = nn.Linear(self.cond_size, 1)


        self.apply(init_weights)

    def forward(self, cond, prev, phase, states):
        #print(cond.shape, prev.shape)
        if self.has_gain:
            gain = torch.norm(prev, dim=1, p=2, keepdim=True)
            prev = prev/(1e-5+gain)
            prev = torch.cat([prev, torch.log(1e-5+gain)], 1)
        
        tmp = torch.cat((cond, prev, phase), 1)
        tmp = torch.tanh(self.sig_dense1(tmp))
        tmp = torch.tanh(self.sig_dense2(tmp))
        gru1_state = self.gru1(tmp, states[0])
        gru2_state = self.gru2(gru1_state, states[1])
        gru3_state = self.gru3(gru2_state, states[2])
        sig_out = torch.tanh(self.sig_dense_out(gru3_state))
        if self.has_gain:
            out_gain = torch.exp(self.gain_dense_out(gru3_state))
            sig_out = sig_out * out_gain
        return sig_out, (gru1_state, gru2_state, gru3_state)

class CELPNet(nn.Module):
    def __init__(self, subframe_size=40, nb_subframes=4, feature_dim=20, cond_size=256, stateful=False, has_gain=False):
        super(CELPNet, self).__init__()

        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.frame_size = self.subframe_size*self.nb_subframes
        self.feature_dim = feature_dim
        self.cond_size = cond_size
        self.stateful = stateful
        self.has_gain = has_gain

        self.cond_net = CELPNetCond(feature_dim=feature_dim, cond_size=cond_size)
        self.sig_net = CELPNetSub(subframe_size=subframe_size, nb_subframes=nb_subframes, cond_size=cond_size, has_gain=has_gain)

    def forward(self, features, period, nb_frames, pre=None, states=None):
        device = features.device
        batch_size = features.size(0)

        phase_real, phase_imag = gen_phase_embedding(period[:, 3:-1], self.frame_size)
        #np.round(32000*phase.detach().numpy()).astype('int16').tofile('phase.sw')

        if states is None:
            states = (
                torch.zeros(batch_size, self.cond_size).to(device),
                torch.zeros(batch_size, self.cond_size).to(device),
                torch.zeros(batch_size, self.cond_size).to(device)
            )

        sig = torch.zeros((batch_size, 0)).to(device)
        cond = self.cond_net(features, period)
        if pre is None:
            nb_pre_frames = 0
            prev = torch.zeros(batch_size, self.subframe_size).to(device)
        else:
            nb_pre_frames = pre.size(1)//self.frame_size
            for n in range(nb_pre_frames):
                for k in range(self.nb_subframes):
                    pos = n*self.frame_size + k*self.subframe_size
                    if pos > 0:
                        preal = phase_real[:, pos:pos+self.subframe_size]
                        pimag = phase_imag[:, pos:pos+self.subframe_size]
                        prev = pre[:, pos-self.subframe_size:pos]
                        phase = torch.cat([preal, pimag], 1)
                        _, states = self.sig_net(cond[:, n, :], prev, phase, states)
            prev = pre[:, -self.subframe_size:]
        for n in range(nb_frames):
            for k in range(self.nb_subframes):
                pos = n*self.frame_size + k*self.subframe_size
                preal = phase_real[:, pos:pos+self.subframe_size]
                pimag = phase_imag[:, pos:pos+self.subframe_size]
                phase = torch.cat([preal, pimag], 1)
                #print("now: ", preal.shape, prev.shape, sig_in.shape)
                out, states = self.sig_net(cond[:, nb_pre_frames+n, :], prev, phase, states)
                sig = torch.cat([sig, out], 1)
                prev = out
        states = [s.detach() for s in states]
        return sig, states

