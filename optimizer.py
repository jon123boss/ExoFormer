# optimizer.py
import torch
from dataclasses import dataclass
from typing import List
from muon.muon import Muon


@dataclass
class OptimizerConfig:
    muon_lr: float = 0.03
    adamw_lr: float = 0.008
    muon_weight_decay: float = 0.0
    adamw_weight_decay: float = 0.0
    cautious: bool = True
    beta1: float = 0.9
    beta2: float = 0.95
    muon_momentum: float = 0.95


def configure_optimizers(model, config: OptimizerConfig):
    muon_params = []
    adamw_params = []
    lambda_params = []
    
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        for layer_idx, block in enumerate(model.transformer.layers):
            for name, p in block.named_parameters():
                if 'lambda_' in name:
                    lambda_params.append(p)
                elif 'dynamic_mixing' in name:
                    if p.ndim >= 2:
                        muon_params.append(p)
                    else:
                        adamw_params.append(p)
                elif p.ndim >= 2:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

    if hasattr(model.transformer, "wte"):
        for p in model.transformer.wte.parameters():
            adamw_params.append(p)
    
    if hasattr(model, "lm_head") and model.lm_head is not None:
        for p in model.lm_head.parameters():
            adamw_params.append(p)
    
    if hasattr(model.transformer, "final_norm"):
        for p in model.transformer.final_norm.parameters():
            if p.ndim < 2: 
                adamw_params.append(p)
    
    if hasattr(model, "smear_gate"):
        for p in model.smear_gate.parameters():
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
    
    if hasattr(model, "smear_lambda"):
        adamw_params.append(model.smear_lambda)
    
    if hasattr(model, "anchor_proj"):
        for p in model.anchor_proj.parameters():
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
    
    if hasattr(model, "anchor_q_norm"):
        for p in model.anchor_q_norm.parameters():
            adamw_params.append(p)
    if hasattr(model, "anchor_k_norm"):
        for p in model.anchor_k_norm.parameters():
            adamw_params.append(p)
    if hasattr(model, "anchor_v_norm"):
        for p in model.anchor_v_norm.parameters():
            adamw_params.append(p)
    if hasattr(model, "anchor_g_norm"):
        for p in model.anchor_g_norm.parameters():
            adamw_params.append(p)
    
    if hasattr(model, "alpha1"):
        adamw_params.append(model.alpha1)
    if hasattr(model, "alpha2"):
        adamw_params.append(model.alpha2)
    
    if hasattr(model, "backout_lambdas"):
        adamw_params.append(model.backout_lambdas)
    
    optimizers = []

    if muon_params:
        muon = Muon(
            muon_params,
            lr=config.muon_lr,
            weight_decay=config.muon_weight_decay,
            momentum=config.muon_momentum,
            cautious=config.cautious,
        )
        optimizers.append(muon)
    
    use_cuda = torch.cuda.is_available()
    
    adamw_groups = []
    
    if lambda_params:
        adamw_groups.append({
            'params': lambda_params,
            'lr': config.adamw_lr * 1,
            'weight_decay': config.adamw_weight_decay,
        })
    
    if adamw_params:
        adamw_groups.append({
            'params': adamw_params,
            'lr': config.adamw_lr,
            'weight_decay': config.adamw_weight_decay,
        })
    

    if adamw_groups:
        adamw = torch.optim.AdamW(
            adamw_groups,
            betas=(config.beta1, config.beta2),
            fused=use_cuda,
            capturable=use_cuda,
        )
        optimizers.append(adamw)
    
    print(f"Muon optimizer: {len(muon_params)} parameters")
    print(f"AdamW optimizer (lambda params): {len(lambda_params)} parameters")
    print(f"AdamW optimizer (other params): {len(adamw_params)} parameters")
    
    return optimizers


def get_optimizers(config, model):
    optimizer_config = OptimizerConfig(
        muon_lr=config["muon_lr"],
        adamw_lr=config["adamw_lr"],
        muon_weight_decay=config["muon_weight_decay"],
        adamw_weight_decay=config["adamw_weight_decay"],
        cautious=config["cautious"],
        beta1=config["beta1"],
        beta2=config["beta2"],
        muon_momentum=config["muon_momentum"],
    )

    optimizers = configure_optimizers(model, optimizer_config)
    
    return optimizers
