import typing as tp

import numpy as np

import torch
import torch.nn as nn


class FilterBasedTokenizer(nn.Module):
    def __init__(self, feature_map_cs: int, visual_tokens_cs: int = 1024, n_visual_tokens: int = 8) -> None:
        super().__init__()
        self.feature_map_cs, self.visual_tokens_cs, self.n_visual_tokens = feature_map_cs, visual_tokens_cs, n_visual_tokens
        self.W_a = nn.Conv1d(feature_map_cs, n_visual_tokens, kernel_size=1, bias=False)
        self.W_v = nn.Conv1d(feature_map_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X: torch.Tensor, T_in: tp.Optional[torch.Tensor]=None):
        # X.shape = bs, feature_map_cs, HW
        A = self.W_a(X) / np.sqrt(self.feature_map_cs)  # bs, n_visual_tokens, HW
        V = self.W_v(X)  # bs, visual_tokens_cs, HW
        attention_weights = self.softmax(A).permute(0, 2, 1) # bs, HW, n_visual_tokens

        return torch.bmm(V, attention_weights)  # bs, visual_tokens_cs, n_visual_tokens [32,1024,8]


class RecurrentTokenizer(nn.Module):
    def __init__(self, feature_map_cs: int, visual_tokens_cs: int = 1024):
        super().__init__()
        self.feature_map_cs, self.visual_tokens_cs = feature_map_cs, visual_tokens_cs
        self.W_tr = nn.Conv1d(visual_tokens_cs, feature_map_cs, kernel_size=1, bias=False)
        self.W_v = nn.Conv1d(feature_map_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X: torch.Tensor, T_in: torch.Tensor):
        # c2-[32,64,3136]  c3-[32,128,784]  c4-[32,256,196]
        # visual_tokens2: torch.Size([32, 1024, 8])
        # X.shape = bs, feature_map_cs, HW
        # T_in.shape = bs, visual_tokens_cs, n_visual_tokens
        W_r = self.W_tr(T_in).permute(0, 2, 1)  # bs, n_visual_tokens, feature_map_cs
        proj = torch.bmm(W_r, X) / np.sqrt(self.feature_map_cs)  # bs, n_visual_tokens, HW
        attention_weights = self.softmax(proj).permute(0, 2, 1)  # bs, HW, n_visual_tokens
        values = self.W_v(X)  # bs, visual_tokens_cs, HW
        return torch.bmm(values, attention_weights)  # bs, visual_tokens_cs, n_visual_tokens


class Transformer(nn.Module):
    def __init__(self, visual_tokens_cs: int):
        super().__init__()
        # self.key_proj = nn.Conv1d(visual_tokens_cs, visual_tokens_cs // 2, kernel_size=1, bias=False)
        # self.query_proj = nn.Conv1d(visual_tokens_cs, visual_tokens_cs // 2, kernel_size=1, bias=False)
        self.key_proj = nn.Conv1d(visual_tokens_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.query_proj = nn.Conv1d(visual_tokens_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.value_proj = nn.Conv1d(visual_tokens_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.f1 = nn.Conv1d(visual_tokens_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.f2 = nn.Conv1d(visual_tokens_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm([visual_tokens_cs, 128])

    def forward(self, T_in: torch.Tensor):
        # T_in.shape = bs, visual_tokens_cs, n_visual_tokens  [32,1024,24]
        visual_tokens_cs = T_in.shape[1]
        key_query_product = torch.bmm(self.key_proj(T_in).permute(0, 2, 1), self.query_proj(T_in))  # bs, n_visual_tokens, n_visual_tokens
        # attention_weights = self.softmax(key_query_product / np.sqrt(visual_tokens_cs // 2)) # / torch.sqrt(n_visual_tokens)
        attention_weights = self.softmax(key_query_product / np.sqrt(visual_tokens_cs))
        values = self.value_proj(T_in)  # bs, visual_tokens_cs, n_visual_tokens
        T_out_dash = self.norm(T_in + torch.bmm(values, attention_weights))  # bs, visual_tokens_cs, n_visual_tokens
        return self.norm(T_out_dash + self.f2(self.activation(self.f1(T_out_dash))))

class Transformer_Decoder(nn.Module):
    def __init__(self, feature_map_cs: int, visual_tokens_cs: int):
        super().__init__()
        self.W_q = nn.Conv1d(feature_map_cs, feature_map_cs, kernel_size=1, bias=False)
        self.W_k = nn.Conv1d(visual_tokens_cs, feature_map_cs, kernel_size=1, bias=False)
        self.W_v = nn.Conv1d(visual_tokens_cs, feature_map_cs, kernel_size=1, bias=False)
        self.f1 = nn.Conv1d(visual_tokens_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.f2 = nn.Conv1d(visual_tokens_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.norm = nn.LayerNorm([visual_tokens_cs, 128])

    def forward(self, X_in: torch.Tensor, T: torch.Tensor):
        # X_in=[32,1024,8]  T=[32,1024,8]
        # X_in: 上一个Decoder的输出； T：encoder
        feature_map_cs = X_in.weight.shape[1]
        pixel_token_similarity = torch.bmm(self.W_q(X_in).permute(0, 2, 1), self.W_k(T))  # [32,8,8] bs, HW, n_visual_tokens
        attention_weights = self.softmax(pixel_token_similarity / np.sqrt(feature_map_cs)).permute(0, 2, 1)  # bs, n_visual_tokens, HW
        values = self.W_v(T)  #[32,1024,8] bs, feature_map_cs, n_visual_tokens
        X_out = self.norm(X_in + torch.bmm(values, attention_weights)) # [32,1024,8]
        decoder_out = self.norm(X_out + self.f2(self.activation(self.f1(X_out))))
        return decoder_out  # [32,1024,8]

class Projector(nn.Module):
    def __init__(self, feature_map_cs: int, visual_tokens_cs: int):
        super().__init__()
        self.W_q = nn.Conv1d(feature_map_cs, feature_map_cs, kernel_size=1, bias=False)
        self.W_k = nn.Conv1d(visual_tokens_cs, feature_map_cs, kernel_size=1, bias=False)
        self.W_v = nn.Conv1d(visual_tokens_cs, feature_map_cs, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X_in: torch.Tensor, T: torch.Tensor):
        # 在Decoder中，Key和Value来自encoder，Quary来自上一个decoder的输出（此处就是输入X_in）
        # X_in.shape = bs, feature_map_cs, HW
        # T.shape = bs, visual_tokens_cs, n_visual_tokens
        feature_map_cs = X_in.shape[1]
        pixel_token_similarity = torch.bmm(self.W_q(X_in).permute(0, 2, 1), self.W_k(T))  # bs, HW, n_visual_tokens
        attention_weights = self.softmax(pixel_token_similarity / np.sqrt(feature_map_cs)).permute(0, 2, 1)  # bs, n_visual_tokens, HW
        values = self.W_v(T)  # bs, feature_map_cs, n_visual_tokens
        return X_in + torch.bmm(values, attention_weights)  # bs, feature_map_cs, HW


class VisualTransformer(nn.Module):
    def __init__(self, tokenizer: tp.Union[FilterBasedTokenizer, RecurrentTokenizer], use_projector):
        super().__init__()
        self.use_projector = use_projector
        self.tokenizer = tokenizer
        self.transformer = Transformer(tokenizer.visual_tokens_cs)
        if use_projector:
            self.projector = Projector(tokenizer.feature_map_cs, tokenizer.visual_tokens_cs)

    def forward(self, feature_map: torch.Tensor, T_in: tp.Optional[torch.Tensor]):
        visual_tokens = self.tokenizer(feature_map, T_in)
        visual_tokens = self.transformer(visual_tokens)
        if not self.use_projector:
            return visual_tokens
        else:
            return self.projector(feature_map, visual_tokens), visual_tokens
