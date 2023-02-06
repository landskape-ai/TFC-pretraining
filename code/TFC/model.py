import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""Two contrastive encoders"""


def random_masking(x, mask_ratio):
    """x: (batch_size, num_tokens, hidden_dim)"""

    (batch_size, num_tokens, hidden_dim) = x.shape
    len_keep = int(num_tokens * (1 - mask_ratio))

    noise = torch.rand(batch_size, num_tokens, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, hidden_dim)
    )

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([batch_size, num_tokens], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


class Time_Encoder(nn.Module):
    def __init__(self, configs):
        super(Time_Encoder, self).__init__()
        self.mask_ratio = configs.t_mask_ratio

        self.embedder_t = nn.Conv1d(
            configs.input_channels,
            configs.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pos_embedder_t = nn.Embedding(configs.TSlength_aligned, configs.hidden_dim)

        encoder_layers_t = TransformerEncoderLayer(
            configs.hidden_dim,
            dim_feedforward=4 * configs.hidden_dim,
            nhead=configs.num_heads,
        )
        self.transformer_encoder_t = TransformerEncoder(
            encoder_layers_t, configs.num_layers
        )

        # projection module
        self.ff1_t = nn.Linear(configs.hidden_dim, 256)
        self.bn_t = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.ff2_t = nn.Linear(256, 128, bias=False)

        # final batch norm 
        self.bn_final = nn.BatchNorm1d(128)

    def forward(self, x_in_t):
        """Embedding"""
        # x_in_t: (batch_size, in_channels, seq_len)
        x_in_t = self.embedder_t(x_in_t)  # (batch_size, hidden_dim, seq_len)
        x_in_t = x_in_t.transpose(-2, -1)  # (batch_size, seq_len, hidden_dim)

        pos_ids = torch.arange(
            x_in_t.size(1), dtype=torch.long, device=x_in_t.device
        ).unsqueeze(0)
        pos_embeds = self.pos_embedder_t(pos_ids)

        x_in_t = x_in_t + pos_embeds

        """ Masking """
        x_in_t_masked, mask_t, ids_restore_t = random_masking(x_in_t, self.mask_ratio)
        x_in_t_masked_2 ,mask_t2, ids_restore_t2 = random_masking(x_in_t, self.mask_ratio)
        x_in_t_massed = torch.cat((x_in_t_masked, x_in_t_masked_2), dim=0)

        """Use Transformer"""
        h_time = self.transformer_encoder_t(x_in_t_masked)

        """Cross-space projector"""
        z_time = self.ff1_t(h_time)  # (batch_size, seq_len, 256)
        z_time = self.bn_t(z_time.transpose(-2, -1))  # (batch_size, 256, seq_len)
        z_time = self.relu(z_time.transpose(-2, -1))  # (batch_size, seq_len, 256)
        z_time = self.ff2_t(z_time)  # (batch_size, seq_len, 128)

        # avg pool
        z_time = z_time.mean(dim=1)  # (batch_size, 128)

        z_time = self.bn_final(z_time)


        h_time, h_time2 = torch.split(h_time, h_time.size(0)//2, dim=0)
        z_time, z_time2 = torch.split(z_time, z_time.size(0)//2, dim=0)
        
        tuple1 = h_time, z_time, mask_t, ids_restore_t
        tuple2 = h_time2, z_time2, mask_t2, ids_restore_t2

        return tuple1, z_time2


class Freq_Encoder(nn.Module):
    def __init__(self, configs):
        super(Freq_Encoder, self).__init__()
        self.mask_ratio = configs.f_mask_ratio

        self.embedder_f = nn.Conv1d(
            configs.input_channels,
            configs.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pos_embedder_f = nn.Embedding(configs.TSlength_aligned, configs.hidden_dim)

        encoder_layers_f = TransformerEncoderLayer(
            configs.hidden_dim,
            dim_feedforward=4 * configs.hidden_dim,
            nhead=configs.num_heads,
        )
        self.transformer_encoder_f = TransformerEncoder(
            encoder_layers_f, configs.num_layers
        )

        # projection module
        self.ff1_f = nn.Linear(configs.hidden_dim, 256)
        self.bn_f = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.ff2_f = nn.Linear(256, 128, bias=False)

        # final batch norm 
        self.bn_final = nn.BatchNorm1d(128)

    def forward(self, x_in_f):
        """Embedding"""
        # x_in_f: (batch_size, in_channels, seq_len)
        x_in_f = self.embedder_f(x_in_f)  # (batch_size, hidden_dim, seq_len)
        x_in_f = x_in_f.transpose(-2, -1)  # (batch_size, seq_len, hidden_dim)

        pos_ids = torch.arange(
            x_in_f.size(1), dtype=torch.long, device=x_in_f.device
        ).unsqueeze(0)
        pos_embeds = self.pos_embedder_f(pos_ids)

        x_in_f = x_in_f + pos_embeds

        """ Masking """
        x_in_f_masked, mask_f, ids_restore_f = random_masking(x_in_f, self.mask_ratio)
        x_in_f_masked_2 ,mask_f2, ids_restore_f2 = random_masking(x_in_f, self.mask_ratio)
        # TODO: hamming distance 
        x_in_f_masked = torch.cat((x_in_f_masked, x_in_f_masked_2), dim=0)

        """Frequency-based contrastive encoder"""
        h_freq = self.transformer_encoder_f(x_in_f_masked)

        """Cross-space projector"""
        z_freq = self.ff1_f(h_freq)  # (batch_size, seq_len, 256)
        z_freq = self.bn_f(z_freq.transpose(-2, -1))  # (batch_size, 256, seq_len)
        z_freq = self.relu(z_freq.transpose(-2, -1))  # (batch_size, seq_len, 256)
        z_freq = self.ff2_f(z_freq)  # (batch_size, seq_len, 128)

        # avg pool
        z_freq = z_freq.mean(dim=1)  # (batch_size, 128)

        z_freq = self.bn_final(z_freq)

        h_freq, h_freq2 = torch.split(h_freq, h_freq.size(0)//2, dim=0)
        z_freq, z_freq2 = torch.split(z_freq, z_freq.size(0)//2, dim=0)
        
        tuple1 = h_freq, z_freq, mask_f, ids_restore_f
        tuple2 = h_freq2, z_freq2, mask_f2, ids_restore_f2

        return tuple1, z_freq2


class Time_Decoder(nn.Module):
    def __init__(self, configs):
        super(Time_Decoder, self).__init__()

        self.pos_embedder_t = nn.Embedding(configs.TSlength_aligned, configs.hidden_dim)

        decoder_layers_t = TransformerEncoderLayer(
            configs.hidden_dim,
            dim_feedforward=4 * configs.hidden_dim,
            nhead=configs.num_heads,
        )
        self.transformer_decoder_t = TransformerEncoder(
            decoder_layers_t, configs.num_layers
        )
        self.head_t = nn.Linear(configs.hidden_dim, configs.input_channels)

    def forward(self, h_time):
        """Add positional embedding"""
        pos_ids = torch.arange(
            h_time.size(1), dtype=torch.long, device=h_time.device
        ).unsqueeze(0)
        pos_embeds = self.pos_embedder_t(pos_ids)
        h_time = h_time + pos_embeds

        """Cross-space projector"""
        z_time = self.transformer_decoder_t(h_time)
        z_time = self.head_t(z_time)

        return z_time


class Freq_Decoder(nn.Module):
    def __init__(self, configs):
        super(Freq_Decoder, self).__init__()

        self.pos_embedder_f = nn.Embedding(configs.TSlength_aligned, configs.hidden_dim)

        decoder_layers_f = TransformerEncoderLayer(
            configs.hidden_dim,
            dim_feedforward=4 * configs.hidden_dim,
            nhead=configs.num_heads,
        )
        self.transformer_decoder_f = TransformerEncoder(
            decoder_layers_f, configs.num_layers
        )
        self.head_f = nn.Linear(configs.hidden_dim, configs.input_channels)

    def forward(self, h_freq):
        """Add positional embedding"""
        pos_ids = torch.arange(
            h_freq.size(1), dtype=torch.long, device=h_freq.device
        ).unsqueeze(0)
        pos_embeds = self.pos_embedder_f(pos_ids)
        h_freq = h_freq + pos_embeds

        """Cross-space projector"""
        z_freq = self.transformer_decoder_f(h_freq)
        z_freq = self.head_f(z_freq)

        return z_freq


class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()
        self.configs = configs

        self.time_encoder = Time_Encoder(configs)
        self.freq_encoder = Freq_Encoder(configs)

        self.time_decoder = Time_Decoder(configs)
        self.freq_decoder = Freq_Decoder(configs)

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        (h_time, z_time, mask_t, ids_restore_t), z_time2  = self.time_encoder(x_in_t)
        (h_freq, z_freq, mask_f, ids_restore_f), z_freq2 = self.freq_encoder(x_in_f)

        # h_time, h_freq: (batch_size, unmasked_seq_len, hidden_dim) [directly from encoder]
        # h_time, h_freq: (batch_size, seq_len, hidden_dim) [input expected by decoder]

        """ Append mask tokens to the sequence """
        mask_tokens = torch.zeros(1, 1, self.configs.hidden_dim).repeat(
            h_time.shape[0], ids_restore_t.shape[1] + 1 - h_time.shape[1], 1
        )
        h_time_ = torch.cat([h_time, mask_tokens], dim=1)
        h_time = torch.gather(
            h_time_,
            dim=1,
            index=ids_restore_t.unsqueeze(-1).repeat(1, 1, h_time.shape[2]),
        )

        mask_tokens = torch.zeros(1, 1, self.configs.hidden_dim).repeat(
            h_freq.shape[0], ids_restore_f.shape[1] + 1 - h_freq.shape[1], 1
        )
        h_freq_ = torch.cat([h_freq, mask_tokens], dim=1)
        h_freq = torch.gather(
            h_freq_,
            dim=1,
            index=ids_restore_f.unsqueeze(-1).repeat(1, 1, h_freq.shape[2]),
        )

        """ Reconstruct the masked input sequence """
        h_time = self.time_decoder(h_time)
        h_freq = self.freq_decoder(h_freq)

        return h_time, h_freq, z_time, z_freq, mask_t, mask_f, z_time2, z_freq2


"""Downstream classifier only used in finetuning"""


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
