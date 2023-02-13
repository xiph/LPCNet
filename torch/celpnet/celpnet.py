import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import filters

Fs = 16000

fid_dict = {}
def dump_signal(x, filename):
    return
    if filename in fid_dict:
        fid = fid_dict[filename]
    else:
        fid = open(filename, "w")
        fid_dict[filename] = fid
    x = x.detach().numpy().astype('float32')
    x.tofile(fid)

def gen_filterbank(N, device):
    in_freq = (np.arange(N+1, dtype='float32')/N*Fs/2)[None,:]
    out_freq = (np.arange(N, dtype='float32')/N*Fs/2)[:,None]
    #ERB from B.C.J Moore, An Introduction to the Psychology of Hearing, 5th Ed., page 73.
    ERB_N = 24.7 + .108*in_freq
    delta = np.abs(in_freq-out_freq)/ERB_N
    center = (delta<.5).astype('float32')
    R = -12*center*delta**2 + (1-center)*(3-12*delta)
    RE = 10.**(R/10.)
    norm = np.sum(RE, axis=0)
    #print(norm.shape)
    RE = RE/norm
    return torch.tensor(RE).to(device)

def gen_weight(N, device):
    freq = np.arange(N, dtype='float32')/N*Fs/2
    #ERB from B.C.J Moore, An Introduction to the Psychology of Hearing, 5th Ed., page 73.
    ERB_N = 24.7 + .108*freq
    W = 240./ERB_N
    W = W[None,:,None]
    return torch.tensor(np.sqrt(W)).to(device)

def new_specgram(N, device):
    x = np.arange(N, dtype='float32')
    w = np.sin(.5*np.pi*np.sin((x+.5)/N*np.pi)**2)
    w = torch.tensor(w).to(device)
    mask = gen_filterbank(N//2, device)
    def compute_specgram(x):
        X = torch.stft(x, N, hop_length=N//4, return_complex=True, center=False, window=w)
        X = torch.abs(X)**2
        M = torch.matmul(mask, X)
        return X, M

    return compute_specgram

def sig_l1(y_true, y_pred):
    return torch.mean(abs(y_true-y_pred))

def lsd_loss(y_true, y_pred, fweight):
    T = 10*torch.log10(1e-8+y_true)
    P = 10*torch.log10(1e-8+y_pred)
    return torch.mean(fweight*torch.square(T-P))

def spec_loss(y_true, y_pred, mask, alpha, fweight):
    diff = (y_true+1e-6)**alpha - (y_pred+1e-6)**alpha
    den = (1e-6+mask)**alpha
    diff = diff[:,:-1,:] / den
    return torch.mean(fweight*torch.square(diff))/(alpha**2)

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
    def __init__(self, subframe_size=40, nb_subframes=4, cond_size=256, passthrough_size=0, has_gain=False, has_lpc=False):
        super(CELPNetSub, self).__init__()

        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.cond_size = cond_size
        self.has_gain = has_gain
        self.has_lpc = has_lpc
        self.passthrough_size = passthrough_size
        
        print("has_gain:", self.has_gain)
        print("has_lpc:", self.has_lpc)
        print("passthrough_size:", self.passthrough_size)
        
        gain_param = 1 if self.has_gain else 0

        self.sig_dense1 = nn.Linear(3*self.subframe_size+self.passthrough_size+self.cond_size+gain_param, self.cond_size)
        self.sig_dense2 = nn.Linear(self.cond_size, self.cond_size)
        self.gru1 = nn.GRUCell(self.cond_size, self.cond_size)
        self.gru2 = nn.GRUCell(self.cond_size, self.cond_size)
        self.gru3 = nn.GRUCell(self.cond_size, self.cond_size)
        self.sig_dense_out = nn.Linear(self.cond_size, self.subframe_size+self.passthrough_size)
        if self.has_gain:
            self.gain_dense_out = nn.Linear(self.cond_size, 1)


        self.apply(init_weights)

    def forward(self, cond, prev, exc_mem, phase, period, states, mat=None, groundtruth=None):
        device = exc_mem.device
        #print(cond.shape, prev.shape)
        
        dump_signal(prev, 'prev_in.f32')
        if self.has_lpc:
            mem = prev[:,-16:].reshape(-1, 16, 1)
            fir_mat = mat[0]
            exc = torch.bmm(fir_mat, mem)

        idx = 256-torch.maximum(torch.tensor(self.subframe_size).to(device), period[:,None])
        rng = torch.arange(self.subframe_size).to(device)
        idx = idx + rng[None,:]
        prev = torch.gather(exc_mem, 1, idx)
        #prev = prev*0
        dump_signal(prev, 'pitch_exc.f32')
        dump_signal(exc_mem, 'exc_mem.f32')
        if self.has_gain:
            gain = torch.norm(prev, dim=1, p=2, keepdim=True)
            prev = prev/(1e-5+gain)
            prev = torch.cat([prev, torch.log(1e-5+gain)], 1)

        passthrough = states[3]
        tmp = torch.cat((cond, prev, passthrough, phase), 1)

        tmp = torch.tanh(self.sig_dense1(tmp))
        tmp = torch.tanh(self.sig_dense2(tmp))
        gru1_state = self.gru1(tmp, states[0])
        gru2_state = self.gru2(gru1_state, states[1])
        gru3_state = self.gru3(gru2_state, states[2])
        sig_out = torch.tanh(self.sig_dense_out(gru3_state))
        if self.passthrough_size != 0:
            passthrough = sig_out[:,self.subframe_size:]
            sig_out = sig_out[:,:self.subframe_size]
        if self.has_gain:
            out_gain = torch.exp(self.gain_dense_out(gru3_state))
            sig_out = sig_out * out_gain
        dump_signal(sig_out, 'exc_out.f32')
        exc_mem = torch.cat([exc_mem[:,self.subframe_size:], sig_out], 1)
        if self.has_lpc:
            iir_mat = mat[1]
            exc2 = torch.cat([exc, sig_out.reshape(-1, self.subframe_size, 1)], 1)
            #print("bmm iir:", iir_mat.shape, exc2.shape)
            syn = torch.bmm(iir_mat, exc2)
            sig_out = syn[:,16:,0]
        return sig_out, exc_mem, (gru1_state, gru2_state, gru3_state, passthrough)

class CELPNet(nn.Module):
    def __init__(self, subframe_size=40, nb_subframes=4, feature_dim=20, cond_size=256, passthrough_size=0, stateful=False, has_gain=False, has_lpc=False, gamma=None):
        super(CELPNet, self).__init__()

        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.frame_size = self.subframe_size*self.nb_subframes
        self.feature_dim = feature_dim
        self.cond_size = cond_size
        self.stateful = stateful
        self.has_gain = has_gain
        self.has_lpc = has_lpc
        self.passthrough_size = passthrough_size
        self.gamma = gamma

        self.cond_net = CELPNetCond(feature_dim=feature_dim, cond_size=cond_size)
        self.sig_net = CELPNetSub(subframe_size=subframe_size, nb_subframes=nb_subframes, cond_size=cond_size, has_gain=has_gain, has_lpc=has_lpc, passthrough_size=passthrough_size)

    def forward(self, features, period, nb_frames, pre=None, states=None, lpc=None):
        device = features.device
        batch_size = features.size(0)

        if self.has_lpc:
            if self.gamma is not None:
                bw = 0.9**(torch.arange(1, 17).to(device))
                lpc = lpc*bw[None,None,:]
            ones = torch.ones((*(lpc.shape[:-1]), 1)).to(device)
            zeros = torch.zeros((*(lpc.shape[:-1]), self.subframe_size-1)).to(device)
            a = torch.cat([ones, lpc], -1)
            a_big = torch.cat([a, zeros], -1)
            fir_mat = filters.toeplitz_from_filter(a)[:,:,:-1,:-1]
            fir_mat_big = filters.toeplitz_from_filter(a_big)
            iir_mat = filters.toeplitz_from_filter(filters.filter_iir_response(a, self.subframe_size+16))
        else:
            fir_mat=None
            iir_mat=None

        phase_real, phase_imag = gen_phase_embedding(period[:, 3:-1], self.frame_size)
        #np.round(32000*phase.detach().numpy()).astype('int16').tofile('phase.sw')

        prev = torch.zeros(batch_size, self.subframe_size).to(device)
        exc_mem = torch.zeros(batch_size, 256).to(device)
        groundtruth = torch.zeros(batch_size, self.subframe_size+16).to(device)
        nb_pre_frames = pre.size(1)//self.frame_size if pre is not None else 0

        if states is None:
            states = (
                torch.zeros(batch_size, self.cond_size).to(device),
                torch.zeros(batch_size, self.cond_size).to(device),
                torch.zeros(batch_size, self.cond_size).to(device),
                torch.zeros(batch_size, self.passthrough_size).to(device)
            )

        sig = torch.zeros((batch_size, 0)).to(device)
        cond = self.cond_net(features, period)
        passthrough = torch.zeros(batch_size, self.passthrough_size).to(device)
        for n in range(nb_frames+nb_pre_frames):
            for k in range(self.nb_subframes):
                pos = n*self.frame_size + k*self.subframe_size
                preal = phase_real[:, pos:pos+self.subframe_size]
                pimag = phase_imag[:, pos:pos+self.subframe_size]
                phase = torch.cat([preal, pimag], 1)
                #print("now: ", preal.shape, prev.shape, sig_in.shape)
                pitch = period[:, 3+n]
                mat = (fir_mat[:,n,:,:], iir_mat[:,n,:,:]) if self.has_lpc else None
                out, exc_mem, states = self.sig_net(cond[:, n, :], prev, exc_mem, phase, pitch, states, mat=mat)

                if n < nb_pre_frames:
                    out = pre[:, pos:pos+self.subframe_size]
                    groundtruth = torch.cat([groundtruth[:,:-self.subframe_size], out], 1)
                    if self.has_lpc:
                        groundtruth_exc = torch.bmm(fir_mat_big[:,n,:,:], groundtruth[:,:,None])
                        exc_mem[:,-self.subframe_size:] = groundtruth_exc[:,-self.subframe_size:,0]
                else:
                    sig = torch.cat([sig, out], 1)

                prev = out
        states = [s.detach() for s in states]
        return sig, states

