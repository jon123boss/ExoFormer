# model2.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from functools import partial
import math

@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 57601 
    n_layer: int = 12 
    n_head: int = 12
    n_embd: int = 768
    mlp_hidden_dim: int = None
    mlp_ratio: float = 4.0
    weight_tying: bool = False
    act_type: str = "gelu"
    rope_theta: float = 500000.0
    rmsnorm_eps: float = 1e-6
    rmsnorm_use_weight: bool = True
    rmsnorm_use_bias: bool = False
    embedding_dropout: float = 0.0
    residual_dropout: float = 0.0 
    attention_dropout: float = 0.0
    norm_pos: str = "after"
    qk_norm: bool = True
    clip_qkv: float = None
    flash_attention: bool = False
    init_std: float = 0.02
    init_cutoff_factor: float = None
    yarn_enabled: bool = False
    yarn_max_seq_len: int = 16384
    yarn_alpha: float = 1.0 
    yarn_beta: float = 32.0
    logit_soft_cap: float = None 
    smear_gate_enabled: bool = True
    smear_gate_dim: int = 12
    value_res_enabled: bool = True
    value_res_lambda_init: float = 0.5
    query_res_enabled: bool = True
    query_res_lambda_init: float = 0.5
    key_res_enabled: bool = True
    key_res_lambda_init: float = 0.5
    per_layer_backout: bool = True
    residual_mode: str = "elementwise"
    gated_attention_enabled: bool = True
    gate_res_enabled: bool = True
    gate_res_lambda_init: float = 0.5
    decouple_anchor: bool = True
    q_residual_norm_enabled: bool = True
    k_residual_norm_enabled: bool = True
    v_residual_norm_enabled: bool = True
    g_residual_norm_enabled: bool = True
    embedding_layer0_mix_enabled: bool = False
    embedding_layer0_alpha_init: float = 0.5
    dynamic_mixing_enabled: bool = True 
    dynamic_mixing_hidden_dim: int = 16
    anchor_delta_enabled: bool = False
    anchor_delta_init: float = 0.0


class ZeroInitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=False)
    
    def reset_parameters(self):
        with torch.no_grad():
            self.weight.zero_()


class Dropout(nn.Dropout):
    def forward(self, input):
        if self.p == 0.0:
            return input
        return F.dropout(input, self.p, self.training, self.inplace)


class ActivationFunction(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        self.act_type = act_type.lower()
        if self.act_type == "relu":
            self.activation = nn.ReLU()
        elif self.act_type == "gelu":
            self.activation = nn.GELU()
        elif self.act_type == "silu":
            self.activation = nn.SiLU()
        elif self.act_type == "swiglu":
            self.activation = SwiGLU()
        elif self.act_type == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {act_type}")
    
    def forward(self, x):
        return self.activation(x)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class RMSNorm(nn.Module):
    def __init__(self, config, dim=None):
        super().__init__()
        self.eps = config.rmsnorm_eps
        dim = dim if dim is not None else config.n_embd

        if config.rmsnorm_use_weight:
            self.weight = nn.Parameter(torch.ones(dim))
            if config.rmsnorm_use_bias:
                self.bias = nn.Parameter(torch.zeros(dim))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        x_norm = x_norm.to(orig_dtype)

        if self.weight is not None:
            x_norm = x_norm * self.weight.to(x_norm.dtype)
        if self.bias is not None:
            x_norm = x_norm + self.bias.to(x_norm.dtype)

        return x_norm


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"

        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("base_inv_freq", inv_freq.clone(), persistent=False)

        self.cos_cached = None
        self.sin_cached = None
        
        self.yarn_enabled = getattr(config, "yarn_enabled", False)
        self.yarn_alpha = getattr(config, "yarn_alpha", 1.0)
        self.yarn_beta = getattr(config, "yarn_beta", 32.0)
        self.yarn_block_size = getattr(config, "block_size", 1024)
        self.yarn_max_seq_len = getattr(config, "yarn_max_seq_len", self.yarn_block_size)
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

    def _build_cache(self, seq_len, device, dtype):
        if (
            self.cos_cached is not None
            and self.cos_cached.size(-2) >= seq_len
            and self.cos_cached.device == device
            and self.cos_cached.dtype == dtype
        ):
            return

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self.cos_cached = cos.to(dtype=dtype)
        self.sin_cached = sin.to(dtype=dtype)

    def _rotate_half(self, x):
        x = x.view(*x.shape[:-1], 2, x.shape[-1] // 2)
        x1, x2 = x.unbind(-2)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary(self, x, cos, sin):
        return (x * cos) + (self._rotate_half(x) * sin)

    def reset_yarn(self):
        if not self.yarn_enabled:
            return
        self.inv_freq = self.base_inv_freq.clone()
        self.cos_cached = None
        self.sin_cached = None
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

    def apply_yarn(self, old_window: int, new_window: int):
        if not self.yarn_enabled or new_window <= old_window:
            return

        inv_freq = self.inv_freq
        rotations = self.yarn_block_size * float(old_window) * inv_freq / (2.0 * math.pi)

        alpha = self.yarn_alpha
        beta = self.yarn_beta
        denom = max(beta - alpha, 1e-6)
        interpolation_weight = torch.clamp((rotations - alpha) / denom, 0.0, 1.0)

        scaling_factor = float(old_window) / float(new_window)
        new_inv_freq = inv_freq * (scaling_factor + interpolation_weight * (1.0 - scaling_factor))
        self.inv_freq = new_inv_freq
        self.cos_cached = None
        self.sin_cached = None
        self.attn_scale *= 0.2 * math.log(float(new_window) / float(old_window)) + 1.0

    def forward(self, q, k, pos_offset=0):
        device = q.device
        dtype = q.dtype
        T = q.size(-2)
        total_len = pos_offset + T

        self._build_cache(total_len, device, dtype)
        cos = self.cos_cached[..., pos_offset:pos_offset + T, :]
        sin = self.sin_cached[..., pos_offset:pos_offset + T, :]

        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        return q, k


class MultiHeadAttention(nn.Module):
    flash_attn_func = None
    flash_attn_varlen_func = None
    flash_tried = False
    
    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.rope = RotaryEmbedding(config)
        self.attention_dropout = config.attention_dropout
        self.layer_idx = layer_idx
        self.config = config
        self.residual_mode = config.residual_mode
        
        self.gated_attention_enabled = config.gated_attention_enabled
        self.gate_res_enabled = config.gate_res_enabled
        self.value_res_enabled = config.value_res_enabled
        self.query_res_enabled = config.query_res_enabled
        self.key_res_enabled = config.key_res_enabled
        
        num_projections = 3 + (1 if self.gated_attention_enabled else 0)
        self.c_attn = nn.Linear(config.n_embd, num_projections * config.n_embd, bias=False)
        self.c_proj = ZeroInitLinear(config.n_embd, config.n_embd)
        
        self.q_norm = RMSNorm(config, dim=self.head_dim) if config.qk_norm else None
        self.k_norm = RMSNorm(config, dim=self.head_dim) if config.qk_norm else None
        self.v_norm = RMSNorm(config, dim=self.head_dim) if config.qk_norm else None
        self.g_norm = RMSNorm(config, dim=self.head_dim) if config.qk_norm and self.gated_attention_enabled else None
        self.clip_qkv = config.clip_qkv
        
        if layer_idx > 0:
            if self.residual_mode == "elementwise":
                if self.value_res_enabled:
                    self.lambda_v1 = nn.Parameter(torch.full((self.n_head, self.head_dim), config.value_res_lambda_init))
                    self.lambda_v2 = nn.Parameter(torch.full((self.n_head, self.head_dim), config.value_res_lambda_init))
                if self.query_res_enabled:
                    self.lambda_q1 = nn.Parameter(torch.full((self.n_head, self.head_dim), config.query_res_lambda_init))
                    self.lambda_q2 = nn.Parameter(torch.full((self.n_head, self.head_dim), config.query_res_lambda_init))
                if self.key_res_enabled:
                    self.lambda_k1 = nn.Parameter(torch.full((self.n_head, self.head_dim), config.key_res_lambda_init))
                    self.lambda_k2 = nn.Parameter(torch.full((self.n_head, self.head_dim), config.key_res_lambda_init))
                if self.gate_res_enabled and self.gated_attention_enabled:
                    self.lambda_g1 = nn.Parameter(torch.full((self.n_head, self.head_dim), config.gate_res_lambda_init))
                    self.lambda_g2 = nn.Parameter(torch.full((self.n_head, self.head_dim), config.gate_res_lambda_init))
            elif self.residual_mode == "headwise":
                if self.value_res_enabled:
                    self.lambda_v1 = nn.Parameter(torch.full((self.n_head,), config.value_res_lambda_init))
                    self.lambda_v2 = nn.Parameter(torch.full((self.n_head,), config.value_res_lambda_init))
                if self.query_res_enabled:
                    self.lambda_q1 = nn.Parameter(torch.full((self.n_head,), config.query_res_lambda_init))
                    self.lambda_q2 = nn.Parameter(torch.full((self.n_head,), config.query_res_lambda_init))
                if self.key_res_enabled:
                    self.lambda_k1 = nn.Parameter(torch.full((self.n_head,), config.key_res_lambda_init))
                    self.lambda_k2 = nn.Parameter(torch.full((self.n_head,), config.key_res_lambda_init))
                if self.gate_res_enabled and self.gated_attention_enabled:
                    self.lambda_g1 = nn.Parameter(torch.full((self.n_head,), config.gate_res_lambda_init))
                    self.lambda_g2 = nn.Parameter(torch.full((self.n_head,), config.gate_res_lambda_init))
            else: 
                if self.value_res_enabled:
                    self.lambda_v1 = nn.Parameter(torch.tensor(config.value_res_lambda_init))
                    self.lambda_v2 = nn.Parameter(torch.tensor(config.value_res_lambda_init))
                if self.query_res_enabled:
                    self.lambda_q1 = nn.Parameter(torch.tensor(config.query_res_lambda_init))
                    self.lambda_q2 = nn.Parameter(torch.tensor(config.query_res_lambda_init))
                if self.key_res_enabled:
                    self.lambda_k1 = nn.Parameter(torch.tensor(config.key_res_lambda_init))
                    self.lambda_k2 = nn.Parameter(torch.tensor(config.key_res_lambda_init))
                if self.gate_res_enabled and self.gated_attention_enabled:
                    self.lambda_g1 = nn.Parameter(torch.tensor(config.gate_res_lambda_init))
                    self.lambda_g2 = nn.Parameter(torch.tensor(config.gate_res_lambda_init))
        
        if config.flash_attention and not MultiHeadAttention.flash_tried:
            try:
                from flash_attn import flash_attn_func, flash_attn_varlen_func
                MultiHeadAttention.flash_attn_func = flash_attn_func
                MultiHeadAttention.flash_attn_varlen_func = flash_attn_varlen_func
                MultiHeadAttention.flash_tried = True
            except Exception as e:
                print(f"Error with flash-attn {e}.")
                MultiHeadAttention.flash_tried = True

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, 
                                       cu_doc_len=None, max_doc_len=None, window_size=None, softmax_scale=None):
        B, H, T, D = q.size()
        
        if cu_doc_len is not None and max_doc_len is not None and MultiHeadAttention.flash_attn_varlen_func is not None:
            q_flat = q.transpose(1, 2).reshape(B * T, H, D)
            k_flat = k.transpose(1, 2).reshape(B * T, H, D)
            v_flat = v.transpose(1, 2).reshape(B * T, H, D)

            cu_doc_len = cu_doc_len.to(device=q.device, dtype=torch.int32)
            x = MultiHeadAttention.flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q=cu_doc_len,
                cu_seqlens_k=cu_doc_len,
                max_seqlen_q=max_doc_len,
                max_seqlen_k=max_doc_len,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=is_causal,
                window_size=(window_size if window_size is not None else -1, -1),
            )
            return x.view(B, T, H, D).contiguous().view(B, T, self.n_embd)
        
        elif MultiHeadAttention.flash_attn_func is not None and attn_mask is None and window_size is None:
            x = MultiHeadAttention.flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=is_causal,
            )
            return x.contiguous().view(B, T, self.n_embd)
        
        else:
            merged_mask = attn_mask

            if window_size is not None and window_size < T:
                idx = torch.arange(T, device=q.device)
                block_start = (idx // window_size) * window_size

                i = idx[:, None]
                j = idx[None, :]

                allowed = (j <= i) & (j >= block_start[:, None])
                local_mask = ~allowed

                if merged_mask is None:
                    merged_mask = local_mask
                else:
                    if merged_mask.dtype != torch.bool:
                        merged_mask = merged_mask.to(torch.bool)
                    merged_mask = merged_mask | local_mask

                causal_flag = False
            else:
                causal_flag = is_causal

            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=merged_mask,
                dropout_p=dropout_p,
                is_causal=causal_flag,
            )
            return x.transpose(1, 2).contiguous().view(B, T, self.n_embd)

    def _apply_gating(self, attention_output, g, g_1):
        B, T, C = attention_output.shape
        g_pre_act = g
        
        if self.gate_res_enabled and self.layer_idx > 0 and g_1 is not None:
            if g_1.size(2) != T:
                g_1 = g_1[:, :, -T:, :]
            
            if self.residual_mode == "elementwise":
                lambda_g1 = self.lambda_g1.view(1, self.n_head, 1, self.head_dim).to(g_pre_act.dtype)
                lambda_g2 = self.lambda_g2.view(1, self.n_head, 1, self.head_dim).to(g_pre_act.dtype)
            elif self.residual_mode == "headwise":
                lambda_g1 = self.lambda_g1.view(1, self.n_head, 1, 1).to(g_pre_act.dtype)
                lambda_g2 = self.lambda_g2.view(1, self.n_head, 1, 1).to(g_pre_act.dtype)
            else: 
                lambda_g1 = self.lambda_g1.to(g_pre_act.dtype)
                lambda_g2 = self.lambda_g2.to(g_pre_act.dtype)
            
            g_pre_act = lambda_g1 * g_1 + lambda_g2 * g_pre_act
        
        gate_scores = torch.sigmoid(g_pre_act)
        attention_reshaped = attention_output.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        gated_output = attention_reshaped * gate_scores
        gated_output = gated_output.transpose(1, 2).contiguous().view(B, T, C)
        
        return gated_output, gate_scores

    def forward(self, x, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None, 
                window_size=None, v_1=None, q_1=None, k_1=None, g_1=None, 
                output_attentions=False, output_gating_scores=False):
        B, T, C = x.size()
        
        if self.gated_attention_enabled:
            q, k, v, g = self.c_attn(x).split(self.n_embd, dim=2)
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            g = None
        
        if self.clip_qkv is not None:
            q.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            k.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            v.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if g is not None:
            g = g.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if self.layer_idx > 0:
            if self.query_res_enabled and q_1 is not None:
                if q_1.size(2) != T:
                    q_1 = q_1[:, :, -T:, :]
                if self.residual_mode == "elementwise":
                    lambda_q1 = self.lambda_q1.view(1, self.n_head, 1, self.head_dim).to(q.dtype)
                    lambda_q2 = self.lambda_q2.view(1, self.n_head, 1, self.head_dim).to(q.dtype)
                elif self.residual_mode == "headwise":
                    lambda_q1 = self.lambda_q1.view(1, self.n_head, 1, 1).to(q.dtype)
                    lambda_q2 = self.lambda_q2.view(1, self.n_head, 1, 1).to(q.dtype)
                else:
                    lambda_q1 = self.lambda_q1.to(q.dtype)
                    lambda_q2 = self.lambda_q2.to(q.dtype)
                q = lambda_q1 * q_1 + lambda_q2 * q
            
            if self.key_res_enabled and k_1 is not None:
                if k_1.size(2) != T:
                    k_1 = k_1[:, :, -T:, :]
                if self.residual_mode == "elementwise":
                    lambda_k1 = self.lambda_k1.view(1, self.n_head, 1, self.head_dim).to(k.dtype)
                    lambda_k2 = self.lambda_k2.view(1, self.n_head, 1, self.head_dim).to(k.dtype)
                elif self.residual_mode == "headwise":
                    lambda_k1 = self.lambda_k1.view(1, self.n_head, 1, 1).to(k.dtype)
                    lambda_k2 = self.lambda_k2.view(1, self.n_head, 1, 1).to(k.dtype)
                else:
                    lambda_k1 = self.lambda_k1.to(k.dtype)
                    lambda_k2 = self.lambda_k2.to(k.dtype)
                k = lambda_k1 * k_1 + lambda_k2 * k
            
            if self.value_res_enabled and v_1 is not None:
                if v_1.size(2) != T:
                    v_1 = v_1[:, :, -T:, :]
                if self.residual_mode == "elementwise":
                    lambda_v1 = self.lambda_v1.view(1, self.n_head, 1, self.head_dim).to(v.dtype)
                    lambda_v2 = self.lambda_v2.view(1, self.n_head, 1, self.head_dim).to(v.dtype)
                elif self.residual_mode == "headwise":
                    lambda_v1 = self.lambda_v1.view(1, self.n_head, 1, 1).to(v.dtype)
                    lambda_v2 = self.lambda_v2.view(1, self.n_head, 1, 1).to(v.dtype)
                else:
                    lambda_v1 = self.lambda_v1.to(v.dtype)
                    lambda_v2 = self.lambda_v2.to(v.dtype)
                v = lambda_v1 * v_1 + lambda_v2 * v
        
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        if self.layer_idx == 0:
            q_residual = q
            k_residual = k
            v_residual = self.v_norm(v) if self.v_norm is not None else v
            g_residual = self.g_norm(g) if (g is not None and self.g_norm is not None) else g
        
        if past_kv is not None:
            past_k, past_v = past_kv
            pos_offset = past_k.size(-2)
        else:
            pos_offset = 0
        
        q, k = self.rope(q, k, pos_offset=pos_offset)
        
        if past_kv is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        dropout_p = self.attention_dropout if self.training else 0.0
        is_causal = past_kv is None

        softmax_scale = None
        if (self.flash_attn_func is not None or self.flash_attn_varlen_func is not None) and getattr(self.rope, "yarn_enabled", False):
            softmax_scale = self.rope.attn_scale

        attn_weights = None
        gate_scores = None
        if output_attentions:
            scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(self.head_dim))
            
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            T_q = q.size(-2)
            T_k = k.size(-2)
            causal_mask = torch.tril(torch.ones(T_q, T_k, device=q.device)).view(1, 1, T_q, T_k)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            if self.attention_dropout and self.training:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
                
            attention_output = torch.matmul(attn_weights, v)
            attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        else:
            attention_output = self._scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=is_causal,
                cu_doc_len=cu_doc_len,
                max_doc_len=max_doc_len,
                window_size=window_size,
                softmax_scale=softmax_scale,
            )
        
        if self.gated_attention_enabled:
            attention_output, gate_scores = self._apply_gating(attention_output, g, g_1)
        
        x = self.c_proj(attention_output)
        
        def pack_return(ret_tuple):
            if output_attentions:
                ret_tuple = ret_tuple + (attn_weights,)
            if output_gating_scores:
                ret_tuple = ret_tuple + (gate_scores,)
            return ret_tuple

        if self.layer_idx == 0:
            if use_cache:
                return pack_return((x, (k, v), q_residual, k_residual, v_residual, g_residual))
            else:
                return pack_return((x, q_residual, k_residual, v_residual, g_residual))
        else:
            if use_cache:
                return pack_return((x, (k, v)))
            else:
                return pack_return((x,))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.mlp_hidden_dim if config.mlp_hidden_dim is not None else int(config.n_embd * config.mlp_ratio)
        self.act = ActivationFunction(config.act_type)
        self.fc1 = nn.Linear(config.n_embd, self.hidden_dim, bias=False)
        self.fc2 = ZeroInitLinear(self.hidden_dim // 2 if config.act_type.lower() == "swiglu" else self.hidden_dim, config.n_embd)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.norm_pos = config.norm_pos
        self.attn_norm = RMSNorm(config)
        self.attn = MultiHeadAttention(config, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)
        self.resid_drop = Dropout(config.residual_dropout)
        self.layer_idx = layer_idx
        self.config = config

    def forward(self, x, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None, 
                window_size=None, v_1=None, q_1=None, k_1=None, g_1=None,
                output_attentions=False, output_gating_scores=False):
        residual = x
        
        if self.norm_pos in {"before", "both"}:
            x = self.attn_norm(x)
        
        attn_out = self.attn(
            x,
            past_kv=past_kv,
            use_cache=use_cache,
            cu_doc_len=cu_doc_len,
            max_doc_len=max_doc_len,
            window_size=window_size,
            v_1=v_1,
            q_1=q_1,
            k_1=k_1,
            g_1=g_1,
            output_attentions=output_attentions,
            output_gating_scores=output_gating_scores
        )

        extras_count = int(output_attentions) + int(output_gating_scores)
        if extras_count > 0:
            attn_content = attn_out[:-extras_count]
            extras_tuple = attn_out[-extras_count:]
        else:
            attn_content = attn_out
            extras_tuple = ()
        
        idx = 0
        attn_weights = None
        gate_scores = None
        if output_attentions:
            attn_weights = extras_tuple[idx]
            idx += 1
        if output_gating_scores:
            gate_scores = extras_tuple[idx]
            idx += 1

        if self.layer_idx == 0:
            if use_cache:
                x, new_kv, q_1_out, k_1_out, v_1_out, g_1_out = attn_content
            else:
                x, q_1_out, k_1_out, v_1_out, g_1_out = attn_content
                new_kv = None
        else:
            if use_cache:
                x, new_kv = attn_content
            else:
                if isinstance(attn_content, tuple):
                    x = attn_content[0]
                else:
                    x = attn_content
                new_kv = None
            q_1_out = k_1_out = v_1_out = g_1_out = None
        
        if self.norm_pos in {"after", "both"}:
            x = self.attn_norm(x)
        
        x = residual + self.resid_drop(x)
        
        residual = x
        
        if self.norm_pos in {"before", "both"}:
            x = self.mlp_norm(x)
        
        x = self.mlp(x)
        
        if self.norm_pos in {"after", "both"}:
            x = self.mlp_norm(x)
        
        x = residual + self.resid_drop(x)
        
        if self.layer_idx == 0:
            if use_cache:
                res = (x, new_kv, q_1_out, k_1_out, v_1_out, g_1_out)
            else:
                res = (x, q_1_out, k_1_out, v_1_out, g_1_out)
        else:
            if use_cache:
                res = (x, new_kv)
            else:
                res = (x,)
        
        if output_attentions:
            res = res + (attn_weights,)
        if output_gating_scores:
            res = res + (gate_scores,)
            
        return res


def logit_soft_cap(logits, cap):
    return cap * torch.tanh(logits / cap)


class OBPM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        
        if config.per_layer_backout:
            self.backout_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            emb_drop=Dropout(config.embedding_dropout),
            layers=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            final_norm=RMSNorm(config)
        ))
        
        if not config.weight_tying:
            self.lm_head = ZeroInitLinear(config.n_embd, config.vocab_size, bias=False)
            
        if config.smear_gate_enabled:
            self.smear_gate = nn.Linear(config.smear_gate_dim, 1, bias=False)
            self.smear_lambda = nn.Parameter(torch.zeros(1))
        
        self.apply(partial(self._init_weights, std=config.init_std, init_cutoff_factor=config.init_cutoff_factor))
    
    def to_mixed_precision(self, dtype=torch.bfloat16):
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                module.to(dtype=dtype)
        if hasattr(self, 'smear_lambda'):
            self.smear_lambda.data = self.smear_lambda.data.to(dtype)
        return self
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def reset_yarn(self):
        if not getattr(self.config, "yarn_enabled", False):
            return
        for block in self.transformer.layers:
            if hasattr(block.attn, "rope") and hasattr(block.attn.rope, "reset_yarn"):
                block.attn.rope.reset_yarn()

    def apply_yarn(self, old_window: int, new_window: int):
        if not getattr(self.config, "yarn_enabled", False):
            return
        for block in self.transformer.layers:
            if hasattr(block.attn, "rope") and hasattr(block.attn.rope, "apply_yarn"):
                block.attn.rope.apply_yarn(old_window, new_window)
    
    def _init_weights(self, module, std=0.02, init_cutoff_factor=None):
        if isinstance(module, ZeroInitLinear):
            return
        if isinstance(module, nn.Linear):
            if init_cutoff_factor is not None:
                cutoff = init_cutoff_factor * std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff, b=cutoff)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            if init_cutoff_factor is not None:
                cutoff = init_cutoff_factor * std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff, b=cutoff)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(self, idx, past_kv=None, use_cache=False, cu_doc_len=None, max_doc_len=None, 
                window_size=None, output_hidden_states=False, output_attentions=False, output_gating_scores=False):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Token length {T} exceeds max sequence length {self.config.block_size}"
        
        x = self.transformer.wte(idx)
        
        if self.config.smear_gate_enabled and T > 1:
            gate = torch.sigmoid(self.smear_gate(x[:, 1:, :self.config.smear_gate_dim]))
            gate = self.smear_lambda * gate
            x_smear = x[:, 1:] + gate * x[:, :-1]
            x = torch.cat([x[:, :1], x_smear], dim=1)
            
        x = self.transformer.emb_drop(x)

        if past_kv is None:
            past_kv = [None] * len(self.transformer.layers)
        new_kv = [] if use_cache else None
        
        layer_outputs = [] if self.config.per_layer_backout else None
        
        v_1 = q_1 = k_1 = g_1 = None
        
        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        all_attentions = () if output_attentions else None
        
        all_gating_scores = () if output_gating_scores else None
        
        for layer_idx, block in enumerate(self.transformer.layers):
            block_out = block(
                x,
                past_kv=past_kv[layer_idx],
                use_cache=use_cache,
                cu_doc_len=cu_doc_len,
                max_doc_len=max_doc_len,
                window_size=window_size,
                v_1=v_1,
                q_1=q_1,
                k_1=k_1,
                g_1=g_1,
                output_attentions=output_attentions,
                output_gating_scores=output_gating_scores,
            )
            
            extras_count = int(output_attentions) + int(output_gating_scores)
            if extras_count > 0:
                block_content = block_out[:-extras_count]
                extras_tuple = block_out[-extras_count:]
            else:
                block_content = block_out
                extras_tuple = ()
            
            idx_extra = 0
            attn_weights = None
            gate_scores = None
            if output_attentions:
                attn_weights = extras_tuple[idx_extra]
                idx_extra += 1
            if output_gating_scores:
                gate_scores = extras_tuple[idx_extra]
                idx_extra += 1

            if layer_idx == 0:
                if use_cache:
                    x, present_kv, q_1, k_1, v_1, g_1 = block_content
                    new_kv.append(present_kv)
                else:
                    x, q_1, k_1, v_1, g_1 = block_content
            else:
                if use_cache:
                    x, present_kv = block_content
                    new_kv.append(present_kv)
                else:
                    x = block_content[0]
            
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)
                
            if output_gating_scores:
                all_gating_scores = all_gating_scores + (gate_scores,)

            if self.config.per_layer_backout:
                layer_outputs.append(x.clone())

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
        
        if self.config.per_layer_backout:
            for i in range(len(self.transformer.layers)):
                x = x - self.backout_lambdas[i] * layer_outputs[i]
        
        x = self.transformer.final_norm(x)
        
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.lm_head(x)
        
        if self.config.logit_soft_cap is not None:
            logits = logit_soft_cap(logits, self.config.logit_soft_cap)
        
        if use_cache:
            return logits, new_kv, all_hidden_states, all_attentions, all_gating_scores
        return logits, all_hidden_states, all_attentions, all_gating_scores
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, max_context=None):
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)
        B, T = idx.size()

        if max_context is None:
            if getattr(self.config, "yarn_enabled", False):
                max_context = getattr(self.config, "yarn_max_seq_len", self.config.block_size)
            else:
                max_context = self.config.block_size

        if T > max_context:
            idx = idx[:, -max_context:]
            T = idx.size(1)

        past_kv = None

        if T > 0:
            start = 0
            while start < T:
                end = min(start + self.config.block_size, T)
                idx_cond = idx[:, start:end]
                logits, past_kv = self(idx_cond, past_kv=past_kv, use_cache=True)
                start = end

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:] if idx.size(1) > 0 else idx
            logits, past_kv = self(idx_cond, past_kv=past_kv, use_cache=True)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if idx.size(1) > max_context:
                idx = idx[:, -max_context:]

        return idx