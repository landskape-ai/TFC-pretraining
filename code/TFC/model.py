import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""Two contrastive encoders"""


class Time_Encoder(nn.Module):
    def __init__(self, configs, mask_ratio):
        super(Time_Encoder, self).__init__()
        self.configs = configs

        self.hidden_dim = 256

        encoder_layers_t = TransformerEncoderLayer(
            # 0.4 * 10 * 18
            d_model=int(mask_ratio * configs.num_tokens * configs.mini_token_length),
            dim_feedforward=int(2 * mask_ratio * configs.num_tokens * configs.mini_token_length),
            nhead=2,
        )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(int(mask_ratio * configs.num_tokens * configs.mini_token_length), self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, int(mask_ratio * configs.num_tokens * configs.mini_token_length)),
        )

    def forward(self, x_in_t):
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time).reshape(-1, 1, self.configs.mini_token_length)

        return h_time, z_time


class Freq_Encoder(nn.Module):
    def __init__(self, configs):
        super(Freq_Encoder, self).__init__()

        encoder_layers_f = TransformerEncoderLayer(
            int(configs.TSlength_aligned * configs.f_mask_ratio),
            dim_feedforward=2 * configs.TSlength_aligned * configs.f_mask_ratio,
            nhead=2,
        )
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.f_mask_ratio, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x_in_f):
        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_freq, z_freq


class Time_Decoder(nn.Module):
    def __init__(self, configs):
        super(Time_Decoder, self).__init__()

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256), # 178, 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, configs.TSlength_aligned),
        )

    def forward(self, h_time):
        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        return z_time


class Freq_Decoder(nn.Module):
    def __init__(self, configs):
        super(Freq_Decoder, self).__init__()

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, configs.TSlength_aligned),
        )

    def forward(self, h_freq):
        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return z_freq

class SSL_Timeseries(nn.Module):
    def __init__(self, configs):
        self.configs = configs
        self.time_encoder_x = Time_Encoder(configs, 0.4)
        self.time_encoder_f = Time_Encoder(configs, 0.6)
        self.freq_encoder = Freq_Encoder(configs)
        self.time_decoder = Time_Decoder(configs)
        self.freq_decoder = Freq_Decoder(configs)

    # def forward(self, x):



# class TFC(nn.Module):
#     def __init__(self, configs):
#         super(TFC, self).__init__()

#         encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=2, )
#         self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

#         self.projector_t = nn.Sequential(
#             nn.Linear(configs.TSlength_aligned, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )

#         encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,nhead=2,)
#         self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

#         self.projector_f = nn.Sequential(
#             nn.Linear(configs.TSlength_aligned, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )


#     def forward(self, x_in_t, x_in_f):
#         """Use Transformer"""
#         x = self.transformer_encoder_t(x_in_t)
#         h_time = x.reshape(x.shape[0], -1)

#         """Cross-space projector"""
#         z_time = self.projector_t(h_time)

#         """Frequency-based contrastive encoder"""
#         f = self.transformer_encoder_f(x_in_f)
#         h_freq = f.reshape(f.shape[0], -1)

#         """Cross-space projector"""
#         z_freq = self.projector_f(h_freq)

#         return h_time, z_time, h_freq, z_freq


# """Downstream classifier only used in finetuning"""


class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2 * 128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
