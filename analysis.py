import os
import sys
import argparse
import importlib
import gc
from dataclasses import asdict, is_dataclass, fields
from pathlib import Path
from collections import defaultdict
import warnings

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    from dataloader import DataLoaderConfig, create_dataloaders, warmup_boundaries
    DATALOADER_AVAILABLE = True
except ImportError:
    DATALOADER_AVAILABLE = False
    print("Warning: dataloader module not found. Will fall back to random data if no data available.")


def format_display_name(name):
    return name.replace("_", " ")


def cleanup_memory(verbose=False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  [Memory] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def _strip_compiled_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_sd[k[len("_orig_mod."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def _unwrap_compiled_model(m):
    if hasattr(m, "_orig_mod"):
        return m._orig_mod
    return m


def compute_layer_to_layer_similarity_matrices(hidden_states):
    layer_indices = sorted(hidden_states.keys())
    n_layers = len(layer_indices)
    
    normalized_states = {}
    for idx in layer_indices:
        h = hidden_states[idx]
        h_flat = h.flatten(0, 1)
        h_norm = F.normalize(h_flat, dim=-1).cpu().float()
        normalized_states[idx] = h_norm
        del h_flat
    
    cleanup_memory()
    
    sim_matrix = np.zeros((n_layers, n_layers), dtype=np.float32)
    
    for i, idx_i in enumerate(layer_indices):
        h_i_norm = normalized_states[idx_i]
        
        for j, idx_j in enumerate(layer_indices):
            if j >= i:
                h_j_norm = normalized_states[idx_j]
                sim = (h_i_norm * h_j_norm).sum(dim=-1).mean().item()
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
    
    del normalized_states
    gc.collect()
    
    return sim_matrix, layer_indices


def plot_layer_similarity_heatmaps(layer_sim_matrices, model_names, output_dir):
    heatmap_dir = os.path.join(output_dir, "hidden_state_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    print(f"\nGenerating layer similarity heatmaps in {heatmap_dir}/")
    
    for name in model_names:
        if name not in layer_sim_matrices:
            continue
        
        display_name = format_display_name(name)
        sim_matrix, layer_indices = layer_sim_matrices[name]
        n_layers = len(layer_indices)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(sim_matrix, cmap="RdBu_r", aspect="auto", vmin=0, vmax=1)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine Similarity", fontsize=12)
        
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Layer Index", fontsize=12)
        ax.set_title(f"{display_name}\nLayer-to-Layer Hidden State Similarity", fontsize=14)
        
        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(n_layers))
        ax.set_xticklabels(layer_indices, rotation=45)
        ax.set_yticklabels(layer_indices)
        
        threshold = 0.5
        for i in range(n_layers):
            for j in range(n_layers):
                value = sim_matrix[i, j]
                if value > threshold:
                    ax.text(j, i, f"{value:.2f}", 
                           ha="center", va="center", 
                           color="white" if value > 0.7 else "black",
                           fontsize=8)
        
        plt.tight_layout()
        
        filename = f"layer_similarity_{name}.png"
        filepath = os.path.join(heatmap_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        np_filename = f"layer_similarity_{name}.npy"
        np_filepath = os.path.join(heatmap_dir, np_filename)
        np.save(np_filepath, sim_matrix)
        
        indices_filename = f"layer_indices_{name}.txt"
        indices_filepath = os.path.join(heatmap_dir, indices_filename)
        with open(indices_filepath, 'w') as f:
            f.write(f"Layer indices: {layer_indices}\n")
            f.write(f"Matrix shape: {sim_matrix.shape}\n")
    
    if len(layer_sim_matrices) > 1:
        n_models = len(model_names)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        if n_models == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        all_values = []
        for name in model_names:
            if name in layer_sim_matrices:
                sim_matrix, _ = layer_sim_matrices[name]
                all_values.extend(sim_matrix.flatten())
        vmin, vmax = min(all_values), max(all_values)
        del all_values
        
        plot_idx = 0
        for name in model_names:
            if name not in layer_sim_matrices:
                continue
            
            display_name = format_display_name(name)
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            sim_matrix, layer_indices = layer_sim_matrices[name]
            n_layers = len(layer_indices)
            
            im = ax.imshow(sim_matrix, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_title(f"{display_name}", fontsize=11)
            ax.set_xlabel("Layer", fontsize=9)
            ax.set_ylabel("Layer", fontsize=9)
            
            if n_layers > 10:
                tick_step = max(1, n_layers // 5)
                ticks = list(range(0, n_layers, tick_step))
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels([layer_indices[i] for i in ticks], fontsize=8)
                ax.set_yticklabels([layer_indices[i] for i in ticks], fontsize=8)
            else:
                ax.set_xticks(range(n_layers))
                ax.set_yticks(range(n_layers))
                ax.set_xticklabels(layer_indices, fontsize=8)
                ax.set_yticklabels(layer_indices, fontsize=8)
            
            plot_idx += 1
        
        for idx in range(plot_idx, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")
        
        fig.suptitle("Layer-to-Layer Hidden State Similarity Comparison", fontsize=14, y=1.02)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        summary_path = os.path.join(heatmap_dir, "comparison_grid.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if len(layer_sim_matrices) == 2:
            names = [n for n in model_names if n in layer_sim_matrices]
            if len(names) == 2:
                name1, name2 = names
                display_name1 = format_display_name(name1)
                display_name2 = format_display_name(name2)
                mat1, indices1 = layer_sim_matrices[name1]
                mat2, indices2 = layer_sim_matrices[name2]
                
                if mat1.shape == mat2.shape:
                    diff_matrix = mat1 - mat2
                    
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
                    
                    im1 = ax1.imshow(mat1, cmap="RdBu_r", aspect="auto", vmin=0, vmax=1)
                    ax1.set_title(f"{display_name1}", fontsize=12)
                    ax1.set_xlabel("Layer", fontsize=10)
                    ax1.set_ylabel("Layer", fontsize=10)
                    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                    
                    im2 = ax2.imshow(mat2, cmap="RdBu_r", aspect="auto", vmin=0, vmax=1)
                    ax2.set_title(f"{display_name2}", fontsize=12)
                    ax2.set_xlabel("Layer", fontsize=10)
                    ax2.set_ylabel("Layer", fontsize=10)
                    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                    
                    max_abs_diff = np.max(np.abs(diff_matrix))
                    im3 = ax3.imshow(diff_matrix, cmap="RdBu", aspect="auto", 
                                     vmin=-max_abs_diff, vmax=max_abs_diff)
                    ax3.set_title(f"Difference ({display_name1} - {display_name2})", fontsize=12)
                    ax3.set_xlabel("Layer", fontsize=10)
                    ax3.set_ylabel("Layer", fontsize=10)
                    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
                    
                    for i in range(diff_matrix.shape[0]):
                        for j in range(diff_matrix.shape[1]):
                            value = diff_matrix[i, j]
                            if abs(value) > 0.1:
                                ax3.text(j, i, f"{value:+.2f}", 
                                       ha="center", va="center", 
                                       color="white" if abs(value) > 0.3 else "black",
                                       fontsize=8)
                    
                    plt.suptitle("Layer Similarity Comparison", fontsize=14, y=1.05)
                    plt.tight_layout()
                    
                    diff_path = os.path.join(heatmap_dir, "difference_comparison.png")
                    plt.savefig(diff_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    np.save(os.path.join(heatmap_dir, "difference_matrix.npy"), diff_matrix)
                    del diff_matrix
    
    print(f"  Saved layer similarity heatmaps to {heatmap_dir}/")
    gc.collect()


def _extract_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"], list(checkpoint.keys())
    if isinstance(checkpoint, dict):
        return checkpoint, list(checkpoint.keys())
    raise ValueError("Checkpoint format not recognized (expected dict).")


def _detect_model_type(state_dict):
    has_anchor = False
    has_v_norm = False
    has_dynamic = False
    
    for k in state_dict.keys():
        if "anchor" in k:
            has_anchor = True
        if "v_norm" in k:
            has_v_norm = True
        if "dynamic" in k:
            has_dynamic = True
    
    if has_dynamic:
        return "model4"
    elif has_anchor:
        return "model3"
    elif has_v_norm:
        return "model2"
    else:
        return "model1"


def _detect_anchor_enabled(model_config, state_dict):
    if hasattr(model_config, 'anchor_enabled') and model_config.anchor_enabled:
        return True
    if hasattr(model_config, 'decouple_anchor') and model_config.decouple_anchor:
        return True
    
    for k in state_dict.keys():
        if 'anchor' in k.lower() and ('proj' in k.lower() or 'norm' in k.lower()):
            return True
    
    layer0_has_lambda = any(
        'layers.0.attn.lambda_' in k 
        for k in state_dict.keys()
    )
    
    return layer0_has_lambda


def _import_model_module(model_type: str):
    if model_type == "model3":
        fallback_order = ["oldmodels.model3", "oldmodels.model2", "oldmodels.model1", "oldmodels.model4"]
    elif model_type == "model2":
        fallback_order = ["oldmodels.model2", "oldmodels.model3", "oldmodels.model1", "oldmodels.model4"]
    elif model_type == "model4":
        fallback_order = ["oldmodels.model4", "oldmodels.model3", "oldmodels.model2", "oldmodels.model1"]
    else:
        fallback_order = ["oldmodels.model1", "oldmodels.model2", "oldmodels.model3", "oldmodels.model4"]
    
    for mod_name in fallback_order:
        try:
            mod = importlib.import_module(mod_name)
            if mod_name != fallback_order[0]:
                print(f"Warning: Using '{mod_name}' as fallback for '{fallback_order[0]}'")
            return mod, mod_name
        except Exception as e:
            print(f"Warning: failed to import '{mod_name}': {e}")
            continue
    
    raise ImportError(f"Failed to import any model module from: {fallback_order}")

def _filter_kwargs_for_dataclass(dc_type, kwargs: dict):
    if not is_dataclass(dc_type):
        return kwargs
    allowed = {f.name for f in fields(dc_type)}
    return {k: v for k, v in kwargs.items() if k in allowed}


def _build_config(ModelConfigType, checkpoint):
    if isinstance(checkpoint, dict) and "model_args" in checkpoint and isinstance(checkpoint["model_args"], dict):
        model_args = checkpoint["model_args"]
        model_args = _filter_kwargs_for_dataclass(ModelConfigType, model_args)
        return ModelConfigType(**model_args)

    if isinstance(checkpoint, dict) and "config" in checkpoint and isinstance(checkpoint["config"], dict):
        cfg = checkpoint["config"]

        mapped = dict(
            block_size=cfg.get("block_size", 2048),
            vocab_size=cfg.get("vocab_size", 57601),
            n_layer=cfg.get("n_layer", 29),
            n_head=cfg.get("n_head", 16),
            n_embd=cfg.get("n_embd", 1024),
            mlp_hidden_dim=cfg.get("mlp_hidden_dim", None),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            weight_tying=cfg.get("weight_tying", False),
            act_type=cfg.get("act_type", "swiglu"),
            rope_theta=cfg.get("rope_theta", 500000.0),
            rmsnorm_eps=cfg.get("rmsnorm_eps", 1e-6),
            rmsnorm_use_weight=cfg.get("rmsnorm_use_weight", True),
            rmsnorm_use_bias=cfg.get("rmsnorm_use_bias", False),
            embedding_dropout=cfg.get("embedding_dropout", 0.0),
            residual_dropout=cfg.get("residual_dropout", 0.0),
            attention_dropout=cfg.get("attention_dropout", 0.0),
            norm_pos=cfg.get("norm_pos", "before"),
            qk_norm=cfg.get("qk_norm", True),
            clip_qkv=cfg.get("clip_qkv", None),
            flash_attention=False,
            init_std=cfg.get("init_std", 0.02),
            init_cutoff_factor=cfg.get("init_cutoff_factor", 3),
            logit_soft_cap=cfg.get("logit_soft_cap", None),
            smear_gate_enabled=cfg.get("smear_gate_enabled", False),
            smear_gate_dim=cfg.get("smear_gate_dim", 1024),
            value_res_enabled=cfg.get("value_res_enabled", True),
            value_res_lambda_init=cfg.get("value_res_lambda_init", 0.5),
            query_res_enabled=cfg.get("query_res_enabled", True),
            query_res_lambda_init=cfg.get("query_res_lambda_init", 0.5),
            key_res_enabled=cfg.get("key_res_enabled", True),
            key_res_lambda_init=cfg.get("key_res_lambda_init", 0.5),
            per_layer_backout=cfg.get("per_layer_backout", False),
            residual_mode=cfg.get("residual_mode", "headwise"),
            gated_attention_enabled=cfg.get("gated_attention_enabled", True),
            gate_res_enabled=cfg.get("gate_res_enabled", True),
            gate_res_lambda_init=cfg.get("gate_res_lambda_init", 0.5),
            anchor_enabled=cfg.get("anchor_enabled", cfg.get("decouple_anchor", False)),
            decouple_anchor=cfg.get("decouple_anchor", cfg.get("anchor_enabled", False)),
            q_residual_norm_enabled=cfg.get("q_residual_norm_enabled", True),
            k_residual_norm_enabled=cfg.get("k_residual_norm_enabled", True),
            v_residual_norm_enabled=cfg.get("v_residual_norm_enabled", True),
            g_residual_norm_enabled=cfg.get("g_residual_norm_enabled", True),
        )
        mapped = _filter_kwargs_for_dataclass(ModelConfigType, mapped)
        return ModelConfigType(**mapped)

    raise ValueError("Cannot find model configuration in checkpoint (expected 'model_args' or 'config').")


@torch.no_grad()
def plot_and_save_lambda_ratio_heatmaps(
    model,
    model_config,
    save_dir: str,
    tag: str = "",
    eps: float = 1e-12,
    percentile_clip=(1.0, 99.0),
):
    os.makedirs(save_dir, exist_ok=True)

    m = _unwrap_compiled_model(model)
    layers = m.transformer.layers
    n_layer = len(layers)

    n_head = int(getattr(model_config, "n_head", 0))
    n_embd = int(getattr(model_config, "n_embd", 0))
    head_dim = n_embd // n_head if (n_head and n_embd) else None
    
    anchor_enabled = getattr(model_config, 'anchor_enabled', False) or getattr(model_config, 'decouple_anchor', False)
    
    layer0_has_lambda = hasattr(layers[0].attn, 'lambda_v1') or hasattr(layers[0].attn, 'lambda_q1')
    
    if anchor_enabled or layer0_has_lambda:
        print(f"[lambda-heatmap] Anchor/decouple mode detected: Layer 0 has lambda parameters")
    else:
        print(f"[lambda-heatmap] Original mode: Layer 0 provides reference (no lambda parameters)")

    display_tag = format_display_name(tag)

    def collect(kind_char: str):
        rows = []
        inferred_mode = None
        inferred_xdim = None

        for li in range(n_layer):
            attn = layers[li].attn
            name1 = f"lambda_{kind_char}1"
            name2 = f"lambda_{kind_char}2"

            if hasattr(attn, name1) and hasattr(attn, name2):
                l1_param = getattr(attn, name1)
                l2_param = getattr(attn, name2)
                
                if not isinstance(l1_param, torch.Tensor) or not isinstance(l2_param, torch.Tensor):
                    rows.append(None)
                    continue
                
                l1 = l1_param.detach().cpu().float()
                l2 = l2_param.detach().cpu().float()

                ratio = (l1 / (l2 + eps))

                if ratio.ndim == 0:
                    mode = "scalar"
                    v = np.array([ratio.item()], dtype=np.float32)
                elif ratio.ndim == 1:
                    mode = "headwise"
                    v = ratio.numpy().astype(np.float32)
                elif ratio.ndim == 2:
                    mode = "elementwise"
                    v = ratio.reshape(-1).numpy().astype(np.float32)
                else:
                    mode = f"unknown_ndim_{ratio.ndim}"
                    v = ratio.reshape(-1).numpy().astype(np.float32)

                if inferred_mode is None:
                    inferred_mode = mode
                    inferred_xdim = int(v.size)

                if inferred_xdim is not None and v.size != inferred_xdim:
                    vv = np.full((inferred_xdim,), np.nan, dtype=np.float32)
                    ncopy = min(inferred_xdim, v.size)
                    vv[:ncopy] = v[:ncopy]
                    v = vv

                rows.append(v)
                del l1, l2, ratio
            else:
                if inferred_xdim is None:
                    rows.append(None)
                else:
                    rows.append(np.full((inferred_xdim,), np.nan, dtype=np.float32))

        if inferred_xdim is None:
            return None, None

        for i in range(len(rows)):
            if rows[i] is None:
                rows[i] = np.full((inferred_xdim,), np.nan, dtype=np.float32)
            elif isinstance(rows[i], torch.Tensor):
                rows[i] = rows[i].cpu().numpy().astype(np.float32)
            elif isinstance(rows[i], np.ndarray):
                if rows[i].dtype != np.float32:
                    rows[i] = rows[i].astype(np.float32)
            else:
                rows[i] = np.array(rows[i], dtype=np.float32)

        mat = np.stack(rows, axis=0).astype(np.float32)
        return mat, inferred_mode

    mats = {}
    modes = {}
    found_any_lambda = False
    for title, k in [("Value", "v"), ("Key", "k"), ("Query", "q"), ("Gate", "g")]:
        mat, mode = collect(k)
        if mat is not None:
            found_any_lambda = True
            if isinstance(mat, np.ndarray) and mat.dtype != np.float32:
                mat = mat.astype(np.float32)
            elif not isinstance(mat, np.ndarray):
                mat = np.array(mat, dtype=np.float32)
        mats[title] = mat
        modes[title] = mode
    
    if not found_any_lambda:
        print("[lambda-heatmap] No lambda parameters found in model; skipping plot.")
        return None

    xdim = None
    for t in ["Value", "Key", "Query", "Gate"]:
        if mats[t] is not None and isinstance(mats[t], np.ndarray) and mats[t].ndim == 2:
            xdim = mats[t].shape[1]
            break
    
    if xdim is None:
        xdim = 1
    
    for t in ["Value", "Key", "Query", "Gate"]:
        if mats[t] is None:
            mats[t] = np.full((n_layer, xdim), np.nan, dtype=np.float32)
            if modes[t] is None:
                for other_t in ["Value", "Key", "Query", "Gate"]:
                    if modes[other_t] is not None:
                        modes[t] = modes[other_t]
                        break
                if modes[t] is None:
                    modes[t] = "unknown"

    def x_label_from(mat, mode_guess):
        if mat is None:
            return "?"
        xdim = mat.shape[1]
        if xdim == 1:
            return "scalar"
        if n_head and xdim == n_head:
            return "head"
        if n_embd and xdim == n_embd:
            return "dimension (flattened head*dim)"
        if head_dim and n_head and xdim == n_head * head_dim:
            return "dimension (flattened head*dim)"
        return f"index (xdim={xdim})"

    all_finite = []
    for t, mat in mats.items():
        if mat is None:
            continue
        try:
            if isinstance(mat, np.ndarray):
                if mat.dtype == object:
                    mat_float = np.array([float(x) if x is not None else np.nan for x in mat.flat], dtype=np.float32).reshape(mat.shape)
                else:
                    mat_float = mat.astype(np.float32)
            else:
                mat_float = np.array(mat, dtype=np.float32)
            vals = mat_float[np.isfinite(mat_float)]
            if vals.size:
                all_finite.append(vals)
            del mat_float
        except (ValueError, TypeError):
            continue
    if all_finite:
        all_finite = np.concatenate(all_finite, axis=0)
        lo, hi = np.percentile(all_finite, percentile_clip)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(all_finite)), float(np.nanmax(all_finite))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = 0.0, 1.0
        del all_finite
    else:
        lo, hi = 0.0, 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    panels = [("Value", axes[0, 0]), ("Key", axes[0, 1]), ("Query", axes[1, 0]), ("Gate", axes[1, 1])]

    for title, ax in panels:
        mat = mats[title]
        mode = modes[title]

        try:
            if mat is None:
                mat = np.full((n_layer, 1), np.nan, dtype=np.float32)
            else:
                if isinstance(mat, np.ndarray):
                    if mat.dtype == object:
                        mat = np.array([float(x) if x is not None else np.nan for x in mat.flat], dtype=np.float32).reshape(mat.shape)
                    else:
                        mat = mat.astype(np.float32)
                else:
                    mat = np.array(mat, dtype=np.float32)
            
            masked = np.ma.masked_invalid(mat)
            im = ax.imshow(masked, aspect="auto", origin="lower", vmin=lo, vmax=hi, cmap="viridis")
        except Exception as e:
            print(f"[lambda-heatmap] Warning: Error plotting {title}: {e}")
            print(f"  mat type: {type(mat)}, dtype: {getattr(mat, 'dtype', 'N/A')}, shape: {getattr(mat, 'shape', 'N/A')}")
            blank = np.full((n_layer, 1), np.nan, dtype=np.float32)
            im = ax.imshow(np.ma.masked_invalid(blank), aspect="auto", origin="lower", vmin=lo, vmax=hi, cmap="viridis")
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes, fontsize=8)

        ax.set_title(f"{title}  (λ1/λ2)", fontsize=30)
        ax.set_ylabel("layer (n_layer)")

        ax.set_xlabel(x_label_from(mat, mode))

        if n_layer <= 32:
            ax.set_yticks(np.arange(n_layer))
        else:
            ticks = np.linspace(0, n_layer - 1, 9).round().astype(int)
            ax.set_yticks(ticks)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("λ1/λ2")

    anchor_status = "Anchor/Decouple Mode (Layer 0 has lambdas)" if (anchor_enabled or layer0_has_lambda) else "Original Mode (Layer 0 = reference)"
    fig.suptitle(f"Lambda Ratio Heatmaps - {anchor_status}", fontsize=12, y=1.02)

    suffix = f"_{tag}" if tag else ""
    out_path = os.path.join(save_dir, f"lambda_ratio_heatmaps{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[lambda-heatmap] Saved: {out_path}")
    
    gc.collect()
    
    return {
        "mats": mats,
        "modes": modes,
        "color_limits": (lo, hi),
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "anchor_enabled": anchor_enabled or layer0_has_lambda,
    }


def plot_lambda_ratio_comparison(all_lambda_data, model_names, output_dir):
    valid_models = [name for name in model_names if name in all_lambda_data and all_lambda_data[name] is not None]
    
    if len(valid_models) < 2:
        print("  Not enough models with lambda data for comparison plots.")
        return
    
    lambda_dir = os.path.join(output_dir, "lambda_heatmaps")
    os.makedirs(lambda_dir, exist_ok=True)
    
    residual_types = ["Value", "Key", "Query", "Gate"]
    
    for res_type in residual_types:
        has_data = any(
            all_lambda_data[name]["mats"].get(res_type) is not None 
            for name in valid_models
        )
        if not has_data:
            continue
        
        n_models = len(valid_models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_models == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        all_vals = []
        for name in valid_models:
            mat = all_lambda_data[name]["mats"].get(res_type)
            if mat is not None:
                vals = mat[np.isfinite(mat)]
                if vals.size:
                    all_vals.extend(vals)
        
        if all_vals:
            lo, hi = np.percentile(all_vals, [1, 99])
        else:
            lo, hi = 0, 1
        del all_vals
        
        for idx, name in enumerate(valid_models):
            display_name = format_display_name(name)
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            mat = all_lambda_data[name]["mats"].get(res_type)
            anchor_mode = all_lambda_data[name].get("anchor_enabled", False)
            mode_str = "(anchor)" if anchor_mode else "(original)"
            
            if mat is not None:
                masked = np.ma.masked_invalid(mat)
                im = ax.imshow(masked, aspect="auto", origin="lower", vmin=lo, vmax=hi, cmap="viridis")
                ax.set_title(f"{display_name} {mode_str}", fontsize=11)
                ax.set_ylabel("Layer")
                ax.set_xlabel("Head/Dim Index")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{display_name} (No {res_type} lambdas)")
        
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        fig.suptitle(f"{res_type} Lambda Ratios (λ1/λ2) Comparison", fontsize=14, y=1.02)
        plt.tight_layout()
        
        out_path = os.path.join(lambda_dir, f"lambda_comparison_{res_type.lower()}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_models)))
    
    for ax, res_type in zip(axes.flat, residual_types):
        for idx, name in enumerate(valid_models):
            display_name = format_display_name(name)
            mat = all_lambda_data[name]["mats"].get(res_type)
            anchor_mode = all_lambda_data[name].get("anchor_enabled", False)
            label_suffix = " (anchor)" if anchor_mode else ""
            
            if mat is not None:
                mean_per_layer = np.nanmean(mat, axis=1)
                layers = np.arange(len(mean_per_layer))
                
                if anchor_mode and not np.isnan(mean_per_layer[0]):
                    ax.plot([0], [mean_per_layer[0]], marker='s', markersize=8,
                           color=colors[idx], label=f"{display_name}{label_suffix} (L0)")
                    ax.plot(layers[1:], mean_per_layer[1:], marker='o', 
                           color=colors[idx], linewidth=2, markersize=4)
                else:
                    ax.plot(layers, mean_per_layer, marker='o', label=display_name + label_suffix, 
                           color=colors[idx], linewidth=2, markersize=4)
        
        ax.set_title(f"{res_type} λ1/λ2 (Mean per Layer)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean λ1/λ2")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, linestyle='--', color='gray', alpha=0.5)
    
    fig.suptitle("Lambda Ratio Summary: Mean per Layer", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(lambda_dir, "lambda_summary_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved lambda comparison plots to {lambda_dir}/")
    gc.collect()


def compute_pca_core_features(hidden_states, variance_threshold=0.99):
    """Memory-optimized PCA computation - processes one layer at a time."""
    core_features = {}
    explained_variance_ratios = {}
    cumulative_variance = {}
    
    layer_indices = sorted(k for k in hidden_states.keys() if k != "final")
    
    for layer_idx in layer_indices:
        h = hidden_states[layer_idx]
        B, T, C = h.shape
        
        h_flat = h.reshape(-1, C).float().cpu().numpy()
        
        h_centered = h_flat - h_flat.mean(axis=0, keepdims=True)
        del h_flat
        
        N = h_centered.shape[0]
        
        try:
            _, S, _ = np.linalg.svd(h_centered, full_matrices=False)
            del h_centered
            gc.collect()
            
            eigenvalues = (S ** 2) / (N - 1)
            del S
            
            total_var = eigenvalues.sum()
            if total_var > 0:
                var_ratios = eigenvalues / total_var
            else:
                var_ratios = np.zeros_like(eigenvalues)
            del eigenvalues
            
            cum_var = np.cumsum(var_ratios)
            
            n_components = np.searchsorted(cum_var, variance_threshold) + 1
            n_components = min(n_components, len(cum_var))
            
            core_features[layer_idx] = int(n_components)
            explained_variance_ratios[layer_idx] = var_ratios
            cumulative_variance[layer_idx] = cum_var
            
        except Exception as e:
            print(f"  Warning: PCA failed for layer {layer_idx}: {e}")
            core_features[layer_idx] = C
            explained_variance_ratios[layer_idx] = np.ones(C) / C
            cumulative_variance[layer_idx] = np.linspace(1/C, 1.0, C)
        
        gc.collect()
    
    return core_features, explained_variance_ratios, cumulative_variance


def plot_pca_core_features(results, model_names, output_dir, variance_threshold=0.99):
    pca_dir = os.path.join(output_dir, "pca_analysis")
    os.makedirs(pca_dir, exist_ok=True)
    
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    if len(model_names) > len(colors):
        cmap = plt.cm.get_cmap("tab20")
        colors = [cmap(i / max(1, len(model_names))) for i in range(len(model_names))]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, name in enumerate(model_names):
        if name not in results or "pca_core_features" not in results[name]:
            continue
        
        display_name = format_display_name(name)
        core_features = results[name]["pca_core_features"]
        layers = sorted(core_features.keys())
        values = [core_features[l] for l in layers]
        
        ax.plot(layers, values, marker="o", label=display_name, linewidth=2.5, 
                markersize=8, color=colors[i % len(colors)])
        
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, linestyle="--", alpha=0.3, color=colors[i % len(colors)])
    
    ax.set_xlabel("Layer Index", fontsize=25)
    ax.set_ylabel(f"Core Features", fontsize=25)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    
    for name in model_names:
        if name in results and "config" in results[name]:
            n_embd = results[name]["config"].get("n_embd", None)
            if n_embd:
                ax.axhline(y=n_embd, linestyle=":", alpha=0.5, color="gray", 
                          label=f"Full dim ({n_embd})")
                break
    
    plt.tight_layout()
    plt.savefig(os.path.join(pca_dir, "pca_core_features.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, name in enumerate(model_names):
        if name not in results or "pca_core_features" not in results[name]:
            continue
        
        display_name = format_display_name(name)
        core_features = results[name]["pca_core_features"]
        n_embd = results[name]["config"].get("n_embd", 1)
        
        layers = sorted(core_features.keys())
        percentages = [(core_features[l] / n_embd) * 100 for l in layers]
        
        ax.plot(layers, percentages, marker="s", label=display_name, linewidth=2.5, 
                markersize=8, color=colors[i % len(colors)])
    
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel(f"Core Features (% of embedding dim)", fontsize=14)
    ax.set_title(f"Intrinsic Dimensionality as Percentage of Full Dimension", fontsize=16)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(pca_dir, "pca_core_features_percentage.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    for name in model_names:
        if name not in results or "pca_cumulative_variance" not in results[name]:
            continue
        
        display_name = format_display_name(name)
        cum_var = results[name]["pca_cumulative_variance"]
        n_layers = len(cum_var)
        
        layer_indices = sorted(cum_var.keys())
        if len(layer_indices) > 6:
            selected = [
                layer_indices[0],
                layer_indices[len(layer_indices)//4],
                layer_indices[len(layer_indices)//2],
                layer_indices[3*len(layer_indices)//4],
                layer_indices[-1]
            ]
            selected = list(dict.fromkeys(selected))
        else:
            selected = layer_indices
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        layer_colors = plt.cm.viridis(np.linspace(0, 1, len(selected)))
        
        for idx, layer_idx in enumerate(selected):
            cv = cum_var[layer_idx]
            n_components = min(200, len(cv))
            ax.plot(range(1, n_components + 1), cv[:n_components], 
                   label=f"Layer {layer_idx}", linewidth=2, color=layer_colors[idx])
        
        ax.axhline(y=variance_threshold, linestyle="--", color="red", alpha=0.7, 
                  label=f"{variance_threshold*100:.0f}% threshold")
        ax.set_xlabel("Number of Principal Components", fontsize=14)
        ax.set_ylabel("Cumulative Explained Variance", fontsize=14)
        ax.set_title(f"{display_name}: Cumulative Explained Variance by Layer", fontsize=16)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, n_components)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(pca_dir, f"cumulative_variance_{name}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    for name in model_names:
        if name not in results or "pca_explained_variance" not in results[name]:
            continue
        
        display_name = format_display_name(name)
        var_ratios = results[name]["pca_explained_variance"]
        layer_indices = sorted(var_ratios.keys())
        
        top_k = 50
        n_layers = len(layer_indices)
        
        var_matrix = np.zeros((n_layers, top_k))
        for i, layer_idx in enumerate(layer_indices):
            vr = var_ratios[layer_idx]
            k = min(top_k, len(vr))
            var_matrix[i, :k] = vr[:k]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(var_matrix, aspect="auto", cmap="viridis", 
                      vmin=0, vmax=np.percentile(var_matrix[var_matrix > 0], 95))
        
        ax.set_xlabel("Principal Component Index", fontsize=14)
        ax.set_ylabel("Layer Index", fontsize=14)
        ax.set_title(f"{display_name}: Explained Variance Ratio per Component", fontsize=16)
        
        if n_layers > 20:
            tick_step = max(1, n_layers // 10)
            yticks = list(range(0, n_layers, tick_step))
            ax.set_yticks(yticks)
            ax.set_yticklabels([layer_indices[i] for i in yticks])
        else:
            ax.set_yticks(range(n_layers))
            ax.set_yticklabels(layer_indices)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Explained Variance Ratio", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(pca_dir, f"variance_heatmap_{name}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        del var_matrix
    
    if len([n for n in model_names if n in results and "pca_core_features" in results[n]]) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_data = []
        for name in model_names:
            if name in results and "pca_core_features" in results[name]:
                display_name = format_display_name(name)
                core_features = results[name]["pca_core_features"]
                mean_cf = np.mean(list(core_features.values()))
                std_cf = np.std(list(core_features.values()))
                min_cf = min(core_features.values())
                max_cf = max(core_features.values())
                model_data.append((display_name, mean_cf, std_cf, min_cf, max_cf))
        
        x = np.arange(len(model_data))
        width = 0.6
        
        means = [d[1] for d in model_data]
        stds = [d[2] for d in model_data]
        names = [d[0] for d in model_data]
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, 
                     color=[colors[i % len(colors)] for i in range(len(model_data))],
                     alpha=0.8, edgecolor='black')
        
        for i, (name, mean, std, min_v, max_v) in enumerate(model_data):
            ax.plot([i, i], [min_v, max_v], 'k-', linewidth=2, alpha=0.5)
            ax.plot(i, min_v, 'k_', markersize=10)
            ax.plot(i, max_v, 'k_', markersize=10)
        
        ax.set_xlabel("Model", fontsize=14)
        ax.set_ylabel(f"Core Features (PCs for {variance_threshold*100:.0f}% variance)", fontsize=14)
        ax.set_title("Mean Core Features Comparison Across Models", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{mean:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha="center", va="bottom", fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(pca_dir, "pca_comparison_bar.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved PCA analysis plots to {pca_dir}/")
    gc.collect()


def compute_all_attention_weights(model, input_ids, dtype):
    model.eval()
    attention_weights = {}
    hidden_states = {}
    gating_scores = {}

    with torch.no_grad():
        input_ids = input_ids.to(next(model.parameters()).device)
        
        all_gating_scores = None
        
        try:
            logits, all_hidden_states, all_attention, all_gating_scores = model(
                input_ids, 
                output_hidden_states=True, 
                output_attentions=True, 
                output_gating_scores=True
            )
        except (ValueError, TypseError):
            try:
                logits, all_hidden_states, all_attention = model(
                    input_ids, 
                    output_hidden_states=True, 
                    output_attentions=True
                )
            except Exception as e:
                raise RuntimeError(f"Model forward pass failed. Ensure model supports output_hidden_states and output_attentions.\nError: {e}") from e

        for i, h in enumerate(all_hidden_states):
            hidden_states[i] = h.clone().cpu()

        for i, attn in enumerate(all_attention):
            try:
                attention_weights[i] = attn.cpu()
            except:
                pass
            
        if all_gating_scores is not None:
            for i, gate in enumerate(all_gating_scores):
                try:
                    gating_scores[i] = gate.cpu()
                except:
                    pass
        
    return attention_weights, hidden_states, gating_scores


def compute_first_token_attention_score(attention_weights):
    first_token_scores = {}
    for layer_idx, attn in attention_weights.items():
        first_token_attn = attn[:, :, :, 0]
        first_token_scores[layer_idx] = first_token_attn.mean().item()
        del first_token_attn
    return first_token_scores


def compute_token_importance(attention_weights):
    token_importance = {}
    for layer_idx, attn in attention_weights.items():
        importance = attn.mean(dim=(0, 1)).sum(dim=0)
        importance = importance / (importance.sum() + 1e-12)
        token_importance[layer_idx] = importance.float().numpy()
        del importance
    return token_importance


def compute_attention_entropy(attention_weights):
    entropies = {}
    eps = 1e-10
    for layer_idx, attn in attention_weights.items():
        attn_clamped = attn.clamp(min=eps)
        entropy = -torch.sum(attn_clamped * torch.log(attn_clamped), dim=-1)
        entropies[layer_idx] = entropy.mean().item()
        del attn_clamped, entropy
    return entropies


def compute_massive_activations(hidden_states):
    max_activations = {}
    for layer_idx, h in hidden_states.items():
        if layer_idx == "final":
            continue
        max_activations[layer_idx] = h.abs().max().item()
    return max_activations


def compute_hidden_state_norms(hidden_states):
    norms = {}
    for layer_idx, h in hidden_states.items():
        if layer_idx == "final":
            continue
        token_norms = torch.norm(h, dim=-1)
        norms[layer_idx] = token_norms.mean().item()
        del token_norms
    return norms


def compute_hidden_state_similarity(hidden_states):
    similarities = {}
    layer_indices = sorted([k for k in hidden_states.keys() if k != "final"])
    for i in range(len(layer_indices) - 1):
        idx1, idx2 = layer_indices[i], layer_indices[i + 1]
        h1 = hidden_states[idx1].flatten(1)
        h2 = hidden_states[idx2].flatten(1)
        similarities[f"{idx1}->{idx2}"] = F.cosine_similarity(h1, h2, dim=-1).mean().item()
        del h1, h2
    return similarities


def compute_first_layer_similarity(hidden_states):
    similarities = {}
    layer_indices = sorted([k for k in hidden_states.keys() if k != "final"])
    if len(layer_indices) < 2:
        return similarities
    h0 = hidden_states[layer_indices[0]].flatten(1)
    for idx in layer_indices[1:]:
        h = hidden_states[idx].flatten(1)
        similarities[idx] = F.cosine_similarity(h0, h, dim=-1).mean().item()
        del h
    del h0
    return similarities


def compute_token_to_token_similarity(hidden_states):
    similarities = {}
    for layer_idx, h in hidden_states.items():
        if layer_idx == "final":
            continue
        B, T, C = h.shape
        h_norm = F.normalize(h, dim=-1)
        sim_matrix = torch.bmm(h_norm, h_norm.transpose(1, 2))
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        upper_sim = sim_matrix[:, mask]
        similarities[layer_idx] = upper_sim.mean().item()
        del h_norm, sim_matrix, mask, upper_sim
    return similarities


def compute_token_to_token_similarity_matrices(hidden_states):
    similarity_matrices = {}
    for layer_idx, h in hidden_states.items():
        if layer_idx == "final":
            continue
        B, T, C = h.shape
        h_norm = F.normalize(h, dim=-1)
        sim_matrix = torch.bmm(h_norm, h_norm.transpose(1, 2))
        avg_sim_matrix = sim_matrix.mean(dim=0).float().numpy()
        similarity_matrices[layer_idx] = avg_sim_matrix
        del h_norm, sim_matrix
    return similarity_matrices


def compute_token_to_token_similarity_std(hidden_states):
    similarity_stds = {}
    for layer_idx, h in hidden_states.items():
        if layer_idx == "final":
            continue
        B, T, C = h.shape
        h_norm = F.normalize(h, dim=-1)
        sim_matrix = torch.bmm(h_norm, h_norm.transpose(1, 2))
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        upper_sim = sim_matrix[:, mask]
        similarity_stds[layer_idx] = upper_sim.std().item()
        del h_norm, sim_matrix, mask, upper_sim
    return similarity_stds


def compute_top_k_token_importance(attention_weights, k_values=(1, 2, 3, 10)):
    top_k_importance = defaultdict(dict)
    for layer_idx, attn in attention_weights.items():
        attn_avg = attn.mean(dim=(0, 1))
        for k in k_values:
            if k > attn_avg.shape[-1]:
                continue
            top_k_scores, _ = torch.topk(attn_avg, k, dim=-1)
            top_k_importance[layer_idx][k] = top_k_scores.mean().item()
            del top_k_scores
        del attn_avg
    return top_k_importance


def compute_gating_metrics(all_gating_scores, threshold=0.2):
    means = {}
    sparsities = {}
    
    if not all_gating_scores:
        return means, sparsities

    for layer_idx, scores in all_gating_scores.items():
        if not scores.is_cpu:
            scores = scores.cpu()
            
        means[layer_idx] = scores.mean().item()
        
        sparsities[layer_idx] = (scores < threshold).float().mean().item()
        
    return means, sparsities


def analyze_value_residual_lambdas(model):
    lambda_stats = {}

    for layer_idx, block in enumerate(model.transformer.layers):
        attn = block.attn
        layer_stats = {}

        if hasattr(attn, "lambda_v1"):
            v1 = attn.lambda_v1.detach().cpu()
            v2 = attn.lambda_v2.detach().cpu()
            layer_stats["value"] = {
                "lambda1_mean": v1.mean().item(),
                "lambda2_mean": v2.mean().item(),
                "ratio": (v1.mean() / (v2.mean() + 1e-6)).item(),
            }
            del v1, v2

        if hasattr(attn, "lambda_q1"):
            q1 = attn.lambda_q1.detach().cpu()
            q2 = attn.lambda_q2.detach().cpu()
            layer_stats["query"] = {
                "lambda1_mean": q1.mean().item(),
                "lambda2_mean": q2.mean().item(),
                "ratio": (q1.mean() / (q2.mean() + 1e-6)).item(),
            }
            del q1, q2

        if hasattr(attn, "lambda_k1"):
            k1 = attn.lambda_k1.detach().cpu()
            k2 = attn.lambda_k2.detach().cpu()
            layer_stats["key"] = {
                "lambda1_mean": k1.mean().item(),
                "lambda2_mean": k2.mean().item(),
                "ratio": (k1.mean() / (k2.mean() + 1e-6)).item(),
            }
            del k1, k2

        if hasattr(attn, "lambda_g1"):
            g1 = attn.lambda_g1.detach().cpu()
            g2 = attn.lambda_g2.detach().cpu()
            layer_stats["gate"] = {
                "lambda1_mean": g1.mean().item(),
                "lambda2_mean": g2.mean().item(),
                "ratio": (g1.mean() / (g2.mean() + 1e-6)).item(),
            }
            del g1, g2

        if layer_stats:
            lambda_stats[layer_idx] = layer_stats

    return lambda_stats


def plot_token_similarity_heatmaps(similarity_matrices, model_name, output_dir):
    display_name = format_display_name(model_name)
    heatmap_dir = os.path.join(output_dir, f"token_similarity_heatmaps_{model_name}")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    n_layers = len(similarity_matrices)
    layers = sorted(similarity_matrices.keys())
    
    print(f"  Generating {n_layers} token similarity heatmaps for {display_name}...")
    
    for layer_idx in layers:
        sim_matrix = similarity_matrices[layer_idx]
        T = sim_matrix.shape[0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=-1, vmax=1)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine Similarity", fontsize=12)
        
        ax.set_xlabel("Token Position", fontsize=12)
        ax.set_ylabel("Token Position", fontsize=12)
        ax.set_title(f"{display_name} - Layer {layer_idx}\nToken-to-Token Similarity", fontsize=14)
        
        if T > 20:
            tick_step = max(1, T // 10)
            ticks = list(range(0, T, tick_step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(ticks)
            ax.set_yticklabels(ticks)
        
        plt.tight_layout()
        
        filename = f"layer_{layer_idx:02d}_token_similarity.png"
        filepath = os.path.join(heatmap_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_layers == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, layer_idx in enumerate(layers):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        sim_matrix = similarity_matrices[layer_idx]
        im = ax.imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=-1, vmax=1)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.set_xlabel("Token", fontsize=8)
        ax.set_ylabel("Token", fontsize=8)
    
    for idx in range(n_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")
    
    fig.suptitle(f"{display_name}: Token-to-Token Similarity Across All Layers", fontsize=14, y=1.02)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    summary_path = os.path.join(heatmap_dir, "all_layers_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved heatmaps to {heatmap_dir}/")
    gc.collect()


def plot_attention_analysis(results, model_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("default")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 10

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    if len(model_names) > len(colors):
        cmap = plt.cm.get_cmap("tab20")
        colors = [cmap(i / max(1, len(model_names))) for i in range(len(model_names))]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        if name in results:
            display_name = format_display_name(name)
            scores = results[name]["first_token_scores"]
            layers = sorted(scores.keys())
            values = [scores[l] for l in layers]
            ax.plot(layers, values, marker="o", label=display_name, linewidth=2, color=colors[i % len(colors)])
            ax.axhline(y=float(np.mean(values)), linestyle="--", alpha=0.4, color=colors[i % len(colors)])
    ax.set_xlabel("Layer Index", fontsize=25)
    ax.set_ylabel("First Token Attention Score", fontsize=25)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "first_token_attention.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        if name in results:
            display_name = format_display_name(name)
            entropies = results[name]["entropies"]
            layers = sorted(entropies.keys())
            values = [entropies[l] for l in layers]
            ax.plot(layers, values, marker="s", label=display_name, linewidth=2, color=colors[i % len(colors)])
    ax.set_xlabel("Layer Index", fontsize=25)
    ax.set_ylabel("Attention Entropy", fontsize=25)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_entropy.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        if name in results:
            display_name = format_display_name(name)
            max_acts = results[name]["max_activations"]
            layers = sorted(max_acts.keys())
            values = [max_acts[l] for l in layers]
            ax.plot(layers, values, marker="^", label=display_name, linewidth=2, color=colors[i % len(colors)])
    ax.set_xlabel("Layer Index", fontsize=25)
    ax.set_ylabel("Maximum Activation Value", fontsize=25)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "massive_activations.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        if name in results and "token_similarity" in results[name]:
            display_name = format_display_name(name)
            sims = results[name]["token_similarity"]
            layers = sorted(sims.keys())
            values = [sims[l] for l in layers]
            ax.plot(layers, values, marker="d", label=display_name, linewidth=2, color=colors[i % len(colors)])
    ax.set_xlabel("Layer Index", fontsize=25)
    ax.set_ylabel("Mean Token-to-Token Similarity", fontsize=25)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "token_similarity.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        if name in results and "token_similarity_std" in results[name]:
            display_name = format_display_name(name)
            sims_std = results[name]["token_similarity_std"]
            layers = sorted(sims_std.keys())
            values = [sims_std[l] for l in layers]
            ax.plot(layers, values, marker="d", label=display_name, linewidth=2, color=colors[i % len(colors)])
    ax.set_xlabel("Layer Index", fontsize=25)
    ax.set_ylabel("Token-to-Token Similarity Std", fontsize=25)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "token_similarity_std.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        if name in results and "first_layer_similarity" in results[name]:
            display_name = format_display_name(name)
            sims = results[name]["first_layer_similarity"]
            layers = sorted(sims.keys())
            values = [sims[l] for l in layers]
            ax.plot(layers, values, marker="o", label=display_name, linewidth=2, color=colors[i % len(colors)])
    ax.set_xlabel("Layer Index", fontsize=25)
    ax.set_ylabel("Cosine Similarity to Layer 0", fontsize=25)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "first_layer_similarity.png"), dpi=150)
    plt.close()

    has_gating_data = any(
        results.get(n, {}).get("gating_means") 
        for n in model_names 
        if n in results
    )
    if has_gating_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, name in enumerate(model_names):
            if name in results and "gating_means" in results[name] and results[name]["gating_means"]:
                display_name = format_display_name(name)
                gating_means = results[name]["gating_means"]
                layers = sorted(gating_means.keys())
                values = [gating_means[l] for l in layers]
                ax.plot(layers, values, marker="o", label=display_name, linewidth=2, color=colors[i % len(colors)])
        ax.set_xlabel("Layer Index", fontsize=25)
        ax.set_ylabel("Mean Gating Score", fontsize=25)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gating_mean.png"), dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, name in enumerate(model_names):
            if name in results and "gating_sparsities" in results[name] and results[name]["gating_sparsities"]:
                display_name = format_display_name(name)
                gating_sparsities = results[name]["gating_sparsities"]
                layers = sorted(gating_sparsities.keys())
                values = [gating_sparsities[l] for l in layers]
                ax.plot(layers, values, marker="s", label=display_name, linewidth=2, color=colors[i % len(colors)])
                
        ax.set_xlabel("Layer Index", fontsize=25)
        ax.set_ylabel("Gating Sparsity (<0.2)", fontsize=25)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gating_sparsity.png"), dpi=150)
        plt.close()

    has_lambdas = any(results.get(name, {}).get("lambda_stats") for name in model_names)
    if has_lambdas:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        residual_types = ["value", "query", "key", "gate"]

        for ax, res_type in zip(axes.flat, residual_types):
            for i, name in enumerate(model_names):
                if name in results and results[name].get("lambda_stats"):
                    display_name = format_display_name(name)
                    lambda_stats = results[name]["lambda_stats"]
                    layers = sorted(lambda_stats.keys())

                    ratios = []
                    for l in layers:
                        if res_type in lambda_stats[l]:
                            ratios.append(lambda_stats[l][res_type]["ratio"])
                        else:
                            ratios.append(np.nan)

                    if not all(np.isnan(r) for r in ratios):
                        ax.plot(layers, ratios, marker="o", label=display_name, color=colors[i % len(colors)])

            ax.set_title(f"{res_type.capitalize()} Residual λ1/λ2 Ratio per Layer", fontsize=12)
            ax.set_xlabel("Layer")
            ax.set_ylabel("λ1/λ2 Ratio")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1, linestyle="--", alpha=0.5, color="gray")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lambda_analysis.png"), dpi=150)
        plt.close()

    for name in model_names:
        if name in results and "attention_weights" in results[name]:
            display_name = format_display_name(name)
            attn = results[name]["attention_weights"]
            n_layers = len(attn)

            layer_indices = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
            layer_indices = [l for l in layer_indices if l in attn]

            fig, axes = plt.subplots(1, len(layer_indices), figsize=(4 * len(layer_indices), 4))
            if len(layer_indices) == 1:
                axes = [axes]

            for ax, layer_idx in zip(axes, layer_indices):
                size = min(64, attn[layer_idx].shape[-1])
                attn_map = attn[layer_idx].mean(dim=(0, 1))[:size, :size].float().numpy()
                im = ax.imshow(attn_map, cmap="viridis", aspect="auto")
                ax.set_xlabel("Key Position")
                ax.set_ylabel("Query Position")
                ax.set_title(f"Layer {layer_idx}")
                plt.colorbar(im, ax=ax, fraction=0.046)

            fig.suptitle(f"{display_name}: Attention Maps Across Layers", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"attention_heatmap_{name}.png"), dpi=150)
            plt.close()

    n_models = sum(1 for name in model_names if name in results)
    if n_models > 0:
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        plot_idx = 0
        for name in model_names:
            if name in results:
                display_name = format_display_name(name)
                ax = axes_flat[plot_idx]
                token_imp = results[name]["token_importance"]
                last_layer = max(token_imp.keys())
                importance = token_imp[last_layer][:50]
                ax.bar(range(len(importance)), importance, alpha=0.7, color=colors[plot_idx % len(colors)])
                ax.set_xlabel("Token Position", fontsize=25)
                ax.set_ylabel("Token Importance", fontsize=25)
                ax.set_title(display_name, fontsize=12)
                ax.grid(True, alpha=0.3)
                plot_idx += 1

        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "token_importance.png"), dpi=150)
        plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    markers = ["o", "s", "^", "d"]

    for i, name in enumerate(model_names):
        if name in results and "top_k_importance" in results[name]:
            display_name = format_display_name(name)
            top_k = results[name]["top_k_importance"]
            layers = sorted(top_k.keys())

            for j, k in enumerate([1, 3, 10]):
                values = [top_k[l].get(k, 0) for l in layers]
                linestyle = "-" if i == 0 else "--"
                ax.plot(
                    layers,
                    values,
                    marker=markers[j],
                    label=f"{display_name} Top-{k}",
                    linestyle=linestyle,
                    color=colors[j % len(colors)],
                    alpha=0.7 if i > 0 else 1.0,
                )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Mean Top-K Attention Score", fontsize=12)
    ax.set_title("Top-K Token Importance per Layer", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_k_importance.png"), dpi=150)
    plt.close()

    analyzed_count = sum(1 for name in model_names if name in results)
    if analyzed_count >= 1:
        metrics = ["Mean First Token Attn", "Mean Entropy", "Max Activation (log)"]
        fig_width = max(10, 4 + 2 * analyzed_count)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        x = np.arange(len(metrics))
        width = 0.8 / analyzed_count

        bar_idx = 0
        for i, name in enumerate(model_names):
            if name in results:
                display_name = format_display_name(name)
                values = [
                    float(np.mean(list(results[name]["first_token_scores"].values()))),
                    float(np.mean(list(results[name]["entropies"].values()))),
                    float(np.log10(max(results[name]["max_activations"].values()) + 1)),
                ]
                offset = (bar_idx - analyzed_count / 2 + 0.5) * width
                ax.bar(x + offset, values, width, label=display_name, color=colors[i % len(colors)])
                bar_idx += 1

        ax.set_ylabel("Value")
        ax.set_title("Summary Comparison Across Models")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "summary_comparison.png"), dpi=150)
        plt.close()

    for name in model_names:
        if name in results and "token_similarity_matrices" in results[name]:
            plot_token_similarity_heatmaps(
                results[name]["token_similarity_matrices"],
                name,
                output_dir
            )
            del results[name]["token_similarity_matrices"]
            gc.collect()
    
    layer_sim_matrices = {}
    for name in model_names:
        if name in results and "layer_similarity_matrix" in results[name]:
            layer_sim_matrices[name] = (
                results[name]["layer_similarity_matrix"],
                results[name]["layer_indices"]
            )
    
    if layer_sim_matrices:
        plot_layer_similarity_heatmaps(layer_sim_matrices, model_names, output_dir)
        del layer_sim_matrices
        gc.collect()
    
    all_lambda_data = {}
    for name in model_names:
        if name in results and "lambda_heatmap_data" in results[name]:
            all_lambda_data[name] = results[name]["lambda_heatmap_data"]
    
    if all_lambda_data:
        plot_lambda_ratio_comparison(all_lambda_data, model_names, output_dir)
        del all_lambda_data
        gc.collect()
    
    has_pca = any(
        name in results and "pca_core_features" in results[name]
        for name in model_names
    )
    if has_pca:
        plot_pca_core_features(results, model_names, output_dir)

    print(f"\nPlots saved to {output_dir}/")
    gc.collect()


def print_summary(results, model_names):
    print("\n" + "=" * 80)
    print("ATTENTION PATTERN ANALYSIS SUMMARY")
    print("=" * 80)

    for name in model_names:
        if name not in results:
            continue

        display_name = format_display_name(name)
        print(f"\n{'=' * 40}")
        print(f"Model: {display_name}")
        if "model_impl" in results[name]:
            print(f"Impl:  {results[name]['model_impl']}")
        if "anchor_enabled" in results[name]:
            anchor_status = "Yes (Layer 0 has lambdas)" if results[name]["anchor_enabled"] else "No (Layer 0 = reference)"
            print(f"Anchor Mode: {anchor_status}")
        if "data_source" in results[name]:
            print(f"Data Source: {results[name]["data_source"]}")
        print(f"{'=' * 40}")

        first_token = results[name]["first_token_scores"]
        mean_first = float(np.mean(list(first_token.values())))
        max_first = float(max(first_token.values()))
        max_layer = max(first_token, key=first_token.get)

        print("\n📊 Attention Sink Analysis:")
        print(f"   Mean First Token Attention: {mean_first:.4f}")
        print(f"   Max First Token Attention:  {max_first:.4f} (Layer {max_layer})")

        entropies = results[name]["entropies"]
        mean_entropy = float(np.mean(list(entropies.values())))
        min_entropy = float(min(entropies.values()))
        min_layer = min(entropies, key=entropies.get)

        print("\n📊 Attention Entropy:")
        print(f"   Mean Entropy: {mean_entropy:.4f}")
        print(f"   Min Entropy:  {min_entropy:.4f} (Layer {min_layer}) [Most concentrated]")

        max_acts = results[name]["max_activations"]
        overall_max = float(max(max_acts.values()))
        max_act_layer = max(max_acts, key=max_acts.get)

        print("\n📊 Massive Activations:")
        print(f"   Maximum Activation: {overall_max:.2f} (Layer {max_act_layer})")

        if "token_similarity" in results[name]:
            token_sim = results[name]["token_similarity"]
            if token_sim:
                mean_sim = float(np.mean(list(token_sim.values())))
                max_sim = float(max(token_sim.values()))
                max_sim_layer = max(token_sim, key=token_sim.get)
                print("\n📊 Over-smoothing (Token Similarity):")
                print(f"   Mean Token Similarity: {mean_sim:.4f}")
                print(f"   Max Token Similarity:  {max_sim:.4f} (Layer {max_sim_layer})")
        
        if "token_similarity_std" in results[name]:
            token_sim_std = results[name]["token_similarity_std"]
            if token_sim_std:
                mean_std = float(np.mean(list(token_sim_std.values())))
                print(f"   Mean Token Similarity Std: {mean_std:.4f}")
        
        if "layer_similarity_matrix" in results[name]:
            layer_sim = results[name]["layer_similarity_matrix"]
            diag_mean = float(np.mean(np.diag(layer_sim)))
            off_diag = layer_sim.copy()
            np.fill_diagonal(off_diag, np.nan)
            off_diag_mean = float(np.nanmean(off_diag))
            
            print("\n📊 Layer-to-Layer Similarity:")
            print(f"   Self-similarity (diagonal): {diag_mean:.4f} (should be ~1.0)")
            print(f"   Cross-layer similarity: {off_diag_mean:.4f}")
            
            n_layers = layer_sim.shape[0]
            first_last_sim = layer_sim[0, -1] if n_layers > 1 else 1.0
            print(f"   First-to-last layer similarity: {first_last_sim:.4f}")
            del off_diag

        if "pca_core_features" in results[name]:
            pca_cf = results[name]["pca_core_features"]
            n_embd = results[name]["config"].get("n_embd", 1)
            mean_cf = float(np.mean(list(pca_cf.values())))
            min_cf = min(pca_cf.values())
            max_cf = max(pca_cf.values())
            min_cf_layer = min(pca_cf, key=pca_cf.get)
            max_cf_layer = max(pca_cf, key=pca_cf.get)
            
            print("\n📊 PCA Core Features (99% variance):")
            print(f"   Mean Core Features: {mean_cf:.1f} ({mean_cf/n_embd*100:.1f}% of {n_embd})")
            print(f"   Min Core Features:  {min_cf} (Layer {min_cf_layer})")
            print(f"   Max Core Features:  {max_cf} (Layer {max_cf_layer})")

        if "gating_means" in results[name] and results[name]["gating_means"]:
            gating_means = results[name]["gating_means"]
            mean_gate = float(np.mean(list(gating_means.values())))
            print(f"\n📊 Gating Scores:")
            print(f"   Mean Gating Score: {mean_gate:.4f}")
            
        if "gating_sparsities" in results[name] and results[name]["gating_sparsities"]:
            gating_sparsities = results[name]["gating_sparsities"]
            mean_sparsity = float(np.mean(list(gating_sparsities.values())))
            print(f"   Mean Gating Sparsity (<0.2): {mean_sparsity:.4f}")

        if results[name].get("lambda_stats"):
            print("\n📊 Residual Lambda Analysis:")
            lambda_stats = results[name]["lambda_stats"]
            
            if 0 in lambda_stats:
                print("   [Anchor Mode: Layer 0 has lambda parameters]")
            
            for res_type in ["value", "query", "key", "gate"]:
                ratios = [lambda_stats[l][res_type]["ratio"] for l in lambda_stats if res_type in lambda_stats[l]]
                if ratios:
                    print(f"   {res_type.capitalize()} λ1/λ2 ratio: {float(np.mean(ratios)):.4f}")

    if len(model_names) > 1:
        print(f"\n{'=' * 40}")
        print("COMPARISON")
        print(f"{'=' * 40}")

        analyzed_models = [name for name in model_names if name in results]

        first_tokens = {name: float(np.mean(list(results[name]["first_token_scores"].values()))) for name in analyzed_models}
        best_sink = min(first_tokens, key=first_tokens.get)
        worst_sink = max(first_tokens, key=first_tokens.get)

        print(f"\n🏆 Lowest Attention Sink: {format_display_name(best_sink)} ({first_tokens[best_sink]:.4f})")
        print(f"   Highest Attention Sink: {format_display_name(worst_sink)} ({first_tokens[worst_sink]:.4f})")

        max_acts_all = {name: float(max(results[name]["max_activations"].values())) for name in analyzed_models}
        best_act = min(max_acts_all, key=max_acts_all.get)
        worst_act = max(max_acts_all, key=max_acts_all.get)

        print(f"\n🏆 Lowest Max Activation: {format_display_name(best_act)} ({max_acts_all[best_act]:.2f})")
        print(f"   Highest Max Activation: {format_display_name(worst_act)} ({max_acts_all[worst_act]:.2f})")

        pca_models = [name for name in analyzed_models if "pca_core_features" in results[name]]
        if pca_models:
            pca_means = {name: float(np.mean(list(results[name]["pca_core_features"].values()))) for name in pca_models}
            lowest_dim = min(pca_means, key=pca_means.get)
            highest_dim = max(pca_means, key=pca_means.get)
            
            print(f"\n🏆 Lowest Intrinsic Dim: {format_display_name(lowest_dim)} ({pca_means[lowest_dim]:.1f} PCs)")
            print(f"   Highest Intrinsic Dim: {format_display_name(highest_dim)} ({pca_means[highest_dim]:.1f} PCs)")

        if len(analyzed_models) >= 2 and first_tokens[worst_sink] != 0:
            sink_reduction = (first_tokens[worst_sink] - first_tokens[best_sink]) / first_tokens[worst_sink] * 100
            print(f"\n📈 Attention Sink Reduction ({format_display_name(worst_sink)} → {format_display_name(best_sink)}): {sink_reduction:.1f}%")

        if len(analyzed_models) >= 2 and max_acts_all[worst_act] != 0:
            act_reduction = (max_acts_all[worst_act] - max_acts_all[best_act]) / max_acts_all[worst_act] * 100
            print(f"📈 Max Activation Reduction ({format_display_name(worst_act)} → {format_display_name(best_act)}): {act_reduction:.1f}%")

        if len(analyzed_models) >= 2:
            print(f"\n{'=' * 40}")
            print("PAIRWISE COMPARISON TABLE")
            print(f"{'=' * 40}")

            header = f"{'Metric':<30}"
            for name in analyzed_models:
                display_name = format_display_name(name)
                header += f" {display_name:>12}"
            print(header)
            print("-" * len(header))

            row = f"{'Attention Sink (mean)':<30}"
            for name in analyzed_models:
                row += f" {first_tokens[name]:>12.4f}"
            print(row)

            row = f"{'Max Activation':<30}"
            for name in analyzed_models:
                row += f" {max_acts_all[name]:>12.2f}"
            print(row)

            entropies_all = {name: float(np.mean(list(results[name]["entropies"].values()))) for name in analyzed_models}
            row = f"{'Attention Entropy (mean)':<30}"
            for name in analyzed_models:
                row += f" {entropies_all[name]:>12.4f}"
            print(row)

            token_sim_all = {}
            for name in analyzed_models:
                if "token_similarity" in results[name] and results[name]["token_similarity"]:
                    token_sim_all[name] = float(np.mean(list(results[name]["token_similarity"].values())))
            if token_sim_all:
                row = f"{'Token Similarity (mean)':<30}"
                for name in analyzed_models:
                    row += f" {token_sim_all.get(name, float('nan')):>12.4f}" if name in token_sim_all else f" {'N/A':>12}"
                print(row)

            token_sim_std_all = {}
            for name in analyzed_models:
                if "token_similarity_std" in results[name] and results[name]["token_similarity_std"]:
                    token_sim_std_all[name] = float(np.mean(list(results[name]["token_similarity_std"].values())))
            if token_sim_std_all:
                row = f"{'Token Similarity Std (mean)':<30}"
                for name in analyzed_models:
                    row += f" {token_sim_std_all.get(name, float('nan')):>12.4f}" if name in token_sim_std_all else f" {'N/A':>12}"
                print(row)

            if pca_models:
                row = f"{'PCA Core Features (mean)':<30}"
                for name in analyzed_models:
                    if name in pca_means:
                        row += f" {pca_means[name]:>12.1f}"
                    else:
                        row += f" {'N/A':>12}"
                print(row)
            
            gating_models = [name for name in analyzed_models if "gating_means" in results[name] and results[name]["gating_means"]]
            if gating_models:
                row = f"{'Gating Mean (mean)':<30}"
                for name in analyzed_models:
                    if name in gating_models:
                         val = float(np.mean(list(results[name]["gating_means"].values())))
                         row += f" {val:>12.4f}"
                    else:
                        row += f" {'N/A':>12}"
                print(row)
                
                row = f"{'Gating Sparsity (mean)':<30}"
                for name in analyzed_models:
                    if name in gating_models:
                         val = float(np.mean(list(results[name]["gating_sparsities"].values())))
                         row += f" {val:>12.4f}"
                    else:
                        row += f" {'N/A':>12}"
                print(row)


def load_model_from_checkpoint(ckpt_path, device):
    print(f"Loading checkpoint from {ckpt_path}...")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    raw_state_dict, _ = _extract_state_dict_from_checkpoint(checkpoint)
    raw_state_dict = _strip_compiled_prefix(raw_state_dict)

    model_type = _detect_model_type(raw_state_dict)
    model_mod, mod_name = _import_model_module(model_type)

    if not hasattr(model_mod, "OBPM") or not hasattr(model_mod, "ModelConfig"):
        raise AttributeError(f"Selected module '{mod_name}' does not expose OBPM and ModelConfig.")
    OBPM = model_mod.OBPM
    ModelConfig = model_mod.ModelConfig

    config = _build_config(ModelConfig, checkpoint)
    
    has_anchor = any("anchor" in k for k in raw_state_dict.keys())
    has_v_norm = any("v_norm" in k for k in raw_state_dict.keys())
    anchor_enabled = _detect_anchor_enabled(config, raw_state_dict)
    
    del checkpoint
    gc.collect()

    model = OBPM(config)
    model.load_state_dict(raw_state_dict, strict=False)
    
    del raw_state_dict
    gc.collect()
    
    model.to(device)

    if device.type == "cuda":
        if hasattr(model, "to_mixed_precision"):
            model.to_mixed_precision(dtype=torch.bfloat16)
        else:
            model = model.to(dtype=torch.bfloat16)
        cleanup_memory()

    model.eval()

    print(f"  Model implementation: {mod_name}.py (anchor={has_anchor}, v_norm={has_v_norm})")
    print(f"  Model loaded: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print(f"  Gated Attention: {getattr(config, 'gated_attention_enabled', 'N/A')}")
    print(f"  Value Residual: {getattr(config, 'value_res_enabled', 'N/A')}")
    
    if anchor_enabled:
        print(f"  Anchor/Decouple Mode: ENABLED (Layer 0 has lambda parameters)")
    else:
        print(f"  Anchor/Decouple Mode: DISABLED (Layer 0 provides reference)")

    return model, config, mod_name, anchor_enabled


def generate_sample_input(batch_size, seq_length, vocab_size, device):
    return torch.randint(0, vocab_size, (batch_size, seq_length), device=device)


def create_validation_dataloader(dataset_dir, block_size, batch_size, use_doc_masking=True, 
                                  doc_separator_token=50256, num_workers=4):
    if not DATALOADER_AVAILABLE:
        raise ImportError("Dataloader module not available. Cannot load validation data.")
    
    dataloader_config = DataLoaderConfig(
        data_dir=dataset_dir,
        batch_size=batch_size,
        block_size=block_size,
        grad_accum_steps=1,
        use_doc_masking=use_doc_masking,
        doc_separator_token=doc_separator_token,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=False,
    )
    
    _, val_loader = create_dataloaders(dataloader_config)
    return val_loader


def get_validation_batches(val_loader, num_batches, device):
    batches = []
    val_iter = iter(val_loader)
    
    for _ in range(num_batches):
        try:
            batch = next(val_iter)
            if len(batch) == 4:
                x, y, cu_seqlens, max_seqlen = batch
            else:
                x, y = batch[:2]
            
            x = x.to(device)
            batches.append(x)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)
            if len(batch) == 4:
                x, y, cu_seqlens, max_seqlen = batch
            else:
                x, y = batch[:2]
            x = x.to(device)
            batches.append(x)
    
    return batches


def get_combined_input(batches):
    return torch.cat(batches, dim=0)


def analyze_single_model(name, ckpt_path, input_ids, analysis_dtype, output_dir, pca_variance_threshold, data_source):
    display_name = format_display_name(name)
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {display_name}")
    print(f"{'=' * 60}")

    try:
        model, config, impl_name, anchor_enabled = load_model_from_checkpoint(ckpt_path, device)

        if input_ids.max() >= config.vocab_size:
            print(f"⚠ Warning: Token IDs exceed vocab size. Clamping to valid range.")
            input_ids = input_ids.clamp(0, config.vocab_size - 1)

        print("Computing attention weights, hidden states and gating scores via model forward pass...")
        attention_weights, hidden_states, gating_scores = compute_all_attention_weights(
            model, input_ids, dtype=analysis_dtype
        )
        
        cleanup_memory()
        
        print("Computing layer-to-layer similarity matrices...")
        layer_sim_matrix, layer_indices = compute_layer_to_layer_similarity_matrices(hidden_states)

        print("Analyzing attention patterns...")
        first_token_scores = compute_first_token_attention_score(attention_weights)
        token_importance = compute_token_importance(attention_weights)
        entropies = compute_attention_entropy(attention_weights)
        top_k_importance = compute_top_k_token_importance(attention_weights)
        
        attention_weights_for_viz = {}
        n_layers = len(attention_weights)
        viz_layers = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
        viz_layers = list(set(l for l in viz_layers if l in attention_weights))
        for l in viz_layers:
            attention_weights_for_viz[l] = attention_weights[l]
        del attention_weights
        gc.collect()
        
        print("Computing hidden state metrics...")
        max_activations = compute_massive_activations(hidden_states)
        hidden_norms = compute_hidden_state_norms(hidden_states)
        similarities = compute_hidden_state_similarity(hidden_states)
        first_layer_sim = compute_first_layer_similarity(hidden_states)
        token_similarity = compute_token_to_token_similarity(hidden_states)
        token_similarity_std = compute_token_to_token_similarity_std(hidden_states)
        token_similarity_matrices = compute_token_to_token_similarity_matrices(hidden_states)
        
        print(f"Computing PCA core features ({pca_variance_threshold*100:.0f}% variance threshold)...")
        pca_core_features, pca_explained_variance, pca_cumulative_variance = compute_pca_core_features(
            hidden_states, variance_threshold=pca_variance_threshold
        )
        
        del hidden_states
        gc.collect()
        cleanup_memory()

        print("Analyzing gating scores...")
        gating_means, gating_sparsities = compute_gating_metrics(gating_scores)

        print("Analyzing residual lambda parameters...")
        lambda_stats = analyze_value_residual_lambdas(model)
        
        print("Generating lambda ratio heatmaps...")
        lambda_heatmap_dir = os.path.join(output_dir, "lambda_heatmaps")
        lambda_heatmap_data = plot_and_save_lambda_ratio_heatmaps(
            model, config, lambda_heatmap_dir, tag=name
        )

        del model
        cleanup_memory()

        result = {
            "model_impl": impl_name,
            "anchor_enabled": anchor_enabled,
            "data_source": data_source,
            "first_token_scores": first_token_scores,
            "token_importance": token_importance,
            "entropies": entropies,
            "max_activations": max_activations,
            "hidden_norms": hidden_norms,
            "similarities": similarities,
            "first_layer_similarity": first_layer_sim,
            "token_similarity": token_similarity,
            "token_similarity_std": token_similarity_std,
            "token_similarity_matrices": token_similarity_matrices,
            "layer_similarity_matrix": layer_sim_matrix,
            "layer_indices": layer_indices,
            "top_k_importance": top_k_importance,
            "gating_stats": None,
            "gating_means": gating_means,
            "gating_sparsities": gating_sparsities,
            "lambda_stats": lambda_stats,
            "lambda_heatmap_data": lambda_heatmap_data,
            "pca_core_features": pca_core_features,
            "pca_explained_variance": pca_explained_variance,
            "pca_cumulative_variance": pca_cumulative_variance,
            "attention_weights": attention_weights_for_viz,
            "config": asdict(config) if is_dataclass(config) else dict(config.__dict__),
        }
        
        cleanup_memory()
        return result

    except Exception as e:
        print(f"Error analyzing {display_name}: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attention patterns in transformer checkpoints using validation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aa.py

  python aa.py --checkpoints out/baseline.pt out/resgated.pt --dataset-dir finewebedu10B

  python aa.py --checkpoints out/v1.pt out/v2.pt --names "Baseline" "ResGated"

  python aa.py --output results/analysis --batch-size 8

  python aa.py --use-random-data
        """,
    )
    parser.add_argument(
        "--checkpoints",
        "-c",
        nargs="+",
        type=str,
        help="Paths to checkpoint files. If not provided, auto-discovers .pt files in --input-dir",
    )
    parser.add_argument(
        "--names",
        "-n",
        nargs="+",
        type=str,
        help="Custom names for each checkpoint (must match number of checkpoints)",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default="out",
        help="Directory to search for checkpoints (default: out)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="attention_analysis_results",
        help="Output directory for results (default: attention_analysis_results)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="Batch size for analysis (default: 4)",
    )
    parser.add_argument(
        "--seq-length",
        "-s",
        type=int,
        default=2048,
        help="Sequence length for analysis (default: 2048)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of validation batches to use (default: 1)",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="+",
        type=str,
        default=[],
        help="Patterns to exclude when auto-discovering checkpoints",
    )
    parser.add_argument(
        "--dataset-dir",
        "-d",
        type=str,
        default="finewebedu10B",
        help="Directory containing training/validation data (default: finewebedu10B)",
    )
    parser.add_argument(
        "--use-doc-masking",
        action="store_true",
        default=True,
        help="Use document masking in dataloader (default: True)",
    )
    parser.add_argument(
        "--no-doc-masking",
        action="store_true",
        help="Disable document masking",
    )
    parser.add_argument(
        "--doc-separator-token",
        type=int,
        default=50256,
        help="Document separator token ID (default: 50256)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--use-random-data",
        action="store_true",
        help="Use random data instead of validation data (fallback mode)",
    )
    parser.add_argument(
        "--pca-variance-threshold",
        type=float,
        default=0.99,
        help="Variance threshold for PCA core features (default: 0.99 for 99%%)",
    )

    args = parser.parse_args()
    
    use_doc_masking = args.use_doc_masking and not args.no_doc_masking

    print("=" * 80)
    print("ATTENTION PATTERN ANALYSIS (Memory Optimized)")
    print("Compatible with modelnew.py, model.py, and oldmodel.py checkpoints (auto-detected)")
    print("=" * 80)

    models_to_analyze = {}

    if args.checkpoints:
        for i, ckpt_path in enumerate(args.checkpoints):
            path = Path(ckpt_path)
            if path.exists():
                name = args.names[i] if args.names and i < len(args.names) else path.stem
                models_to_analyze[name] = path
                print(f"\n✓ Found checkpoint: {path} (as '{format_display_name(name)}')")
            else:
                print(f"\n✗ Checkpoint not found: {ckpt_path}")
    else:
        input_dir = Path(args.input_dir)
        if input_dir.exists():
            print(f"\nAuto-discovering checkpoints in {input_dir}...")
            pt_files = sorted(input_dir.glob("*.pt"))
            for pt_file in pt_files:
                excluded = any(pattern in pt_file.name for pattern in args.exclude)
                if not excluded:
                    name = pt_file.stem
                    models_to_analyze[name] = pt_file
                    print(f"  ✓ Found: {pt_file} (as '{format_display_name(name)}')")
                else:
                    print(f"  ✗ Excluded: {pt_file}")
        else:
            print(f"\n✗ Input directory not found: {input_dir}")

    if not models_to_analyze:
        print("\nError: No checkpoints found.")
        print("Use --checkpoints to specify checkpoint paths, or ensure .pt files exist in the input directory.")
        return

    print(f"\nFound {len(models_to_analyze)} checkpoint(s) to analyze")

    batch_size = args.batch_size
    seq_length = args.seq_length
    num_batches = args.num_batches
    output_dir = Path(args.output)
    pca_variance_threshold = args.pca_variance_threshold

    analysis_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    val_loader = None
    use_validation_data = not args.use_random_data
    input_ids = None
    data_source = "random"
    
    if use_validation_data and DATALOADER_AVAILABLE:
        try:
            print(f"\nLoading validation data from {args.dataset_dir}...")
            val_loader = create_validation_dataloader(
                dataset_dir=args.dataset_dir,
                block_size=seq_length,
                batch_size=batch_size,
                use_doc_masking=use_doc_masking,
                doc_separator_token=args.doc_separator_token,
                num_workers=args.num_workers,
            )
            
            if use_doc_masking:
                print("Warming up document boundary cache...")
                warmup_boundaries(val_loader.dataset, num_shards=min(4, len(val_loader.dataset.shards)))
                print("Boundary warmup complete.")
            
            print(f"✓ Validation dataloader created successfully")
            print(f"  Dataset size: {len(val_loader.dataset):,} sequences")
            print(f"  Using {num_batches} batch(es) of {batch_size} sequences each")
            
            batches = get_validation_batches(val_loader, num_batches, device)
            input_ids = get_combined_input(batches)
            del batches
            data_source = f"validation ({args.dataset_dir})"
            print(f"  Combined input shape: {input_ids.shape}")
            
        except Exception as e:
            print(f"\n⚠ Warning: Failed to load validation data: {e}")
            print("  Falling back to random data...")
            val_loader = None
            use_validation_data = False
    elif use_validation_data and not DATALOADER_AVAILABLE:
        print("\n⚠ Warning: Dataloader module not available. Using random data.")
        use_validation_data = False
    else:
        print("\nUsing random data (--use-random-data flag set)")

    results = {}
    model_names = list(models_to_analyze.keys())

    for name, ckpt_path in models_to_analyze.items():
        if input_ids is None:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if "config" in checkpoint:
                vocab_size = checkpoint["config"].get("vocab_size", 57601)
            elif "model_args" in checkpoint:
                vocab_size = checkpoint["model_args"].get("vocab_size", 57601)
            else:
                vocab_size = 57601
            del checkpoint
            gc.collect()
            
            print(f"\nGenerating random sample input (batch={batch_size}, seq_len={seq_length})...")
            input_ids = generate_sample_input(batch_size * num_batches, seq_length, vocab_size, device)
        
        result = analyze_single_model(
            name, ckpt_path, input_ids, analysis_dtype, output_dir, pca_variance_threshold, data_source
        )
        
        if result is not None:
            results[name] = result
        
        cleanup_memory(verbose=True)

    del input_ids
    cleanup_memory()

    if not results:
        print("\nNo models were successfully analyzed.")
        return

    print_summary(results, model_names)

    print("\n" + "=" * 60)
    print("Generating visualization plots...")
    print("=" * 60)

    plot_attention_analysis(results, model_names, output_dir)

    results_file = output_dir / "analysis_results.txt"
    os.makedirs(output_dir, exist_ok=True)

    with open(results_file, "w") as f:
        f.write("ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write("Compatible with modelnew.py, model.py, and oldmodel.py checkpoints (auto-detected)\n")
        f.write("=" * 80 + "\n\n")

        for name in model_names:
            if name not in results:
                continue

            display_name = format_display_name(name)
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Model: {display_name}\n")
            f.write(f"Impl:  {results[name].get('model_impl', 'unknown')}\n")
            f.write(f"Data Source: {results[name].get('data_source', 'unknown')}\n")
            anchor_status = "ENABLED (Layer 0 has lambdas)" if results[name].get('anchor_enabled', False) else "DISABLED (Layer 0 = reference)"
            f.write(f"Anchor Mode: {anchor_status}\n")
            f.write(f"{'=' * 60}\n")

            cfg = results[name]["config"]
            f.write("\nConfiguration:\n")
            f.write(f"  Layers: {cfg.get('n_layer')}\n")
            f.write(f"  Heads: {cfg.get('n_head')}\n")
            f.write(f"  Embedding Dim: {cfg.get('n_embd')}\n")
            f.write(f"  Gated Attention: {cfg.get('gated_attention_enabled')}\n")
            f.write(f"  Value Residual: {cfg.get('value_res_enabled')}\n")
            f.write(f"  Anchor Enabled: {cfg.get('anchor_enabled', cfg.get('decouple_anchor', False))}\n")

            f.write("\n" + "-" * 40 + "\n")
            f.write("First Token Attention Scores (Attention Sink):\n")
            f.write("-" * 40 + "\n")
            for layer, score in sorted(results[name]["first_token_scores"].items()):
                f.write(f"  Layer {layer:2d}: {score:.6f}\n")
            mean_sink = float(np.mean(list(results[name]["first_token_scores"].values())))
            f.write(f"  Mean:     {mean_sink:.6f}\n")

            f.write("\n" + "-" * 40 + "\n")
            f.write("Attention Entropy:\n")
            f.write("-" * 40 + "\n")
            for layer, entropy in sorted(results[name]["entropies"].items()):
                f.write(f"  Layer {layer:2d}: {entropy:.6f}\n")
            mean_entropy = float(np.mean(list(results[name]["entropies"].values())))
            f.write(f"  Mean:     {mean_entropy:.6f}\n")

            f.write("\n" + "-" * 40 + "\n")
            f.write("Maximum Activations (Massive Activations):\n")
            f.write("-" * 40 + "\n")
            for layer, max_act in sorted(results[name]["max_activations"].items()):
                f.write(f"  Layer {layer:2d}: {max_act:.2f}\n")

            if results[name].get("token_similarity"):
                f.write("\n" + "-" * 40 + "\n")
                f.write("Token-to-Token Similarity (Over-smoothing):\n")
                f.write("-" * 40 + "\n")
                for layer, sim in sorted(results[name]["token_similarity"].items()):
                    f.write(f"  Layer {layer:2d}: {sim:.6f}\n")

            if results[name].get("token_similarity_std"):
                f.write("\n" + "-" * 40 + "\n")
                f.write("Token-to-Token Similarity Std:\n")
                f.write("-" * 40 + "\n")
                for layer, sim_std in sorted(results[name]["token_similarity_std"].items()):
                    f.write(f"  Layer {layer:2d}: {sim_std:.6f}\n")

            if results[name].get("pca_core_features"):
                f.write("\n" + "-" * 40 + "\n")
                f.write(f"PCA Core Features ({pca_variance_threshold*100:.0f}% variance):\n")
                f.write("-" * 40 + "\n")
                n_embd = cfg.get('n_embd', 1)
                for layer, cf in sorted(results[name]["pca_core_features"].items()):
                    pct = cf / n_embd * 100
                    f.write(f"  Layer {layer:2d}: {cf:4d} ({pct:5.1f}% of {n_embd})\n")
                mean_cf = float(np.mean(list(results[name]["pca_core_features"].values())))
                f.write(f"  Mean:     {mean_cf:.1f} ({mean_cf/n_embd*100:.1f}%)\n")

            if results[name].get("gating_means") and results[name]["gating_means"]:
                f.write("\n" + "-" * 40 + "\n")
                f.write("Gating Scores:\n")
                f.write("-" * 40 + "\n")
                for layer, score in sorted(results[name]["gating_means"].items()):
                    f.write(f"  Layer {layer:2d}: {score:.6f}\n")
                mean_gate = float(np.mean(list(results[name]["gating_means"].values())))
                f.write(f"  Mean:     {mean_gate:.6f}\n")

            if results[name].get("gating_sparsities") and results[name]["gating_sparsities"]:
                f.write("\n" + "-" * 40 + "\n")
                f.write("Gating Sparsity (<0.2):\n")
                f.write("-" * 40 + "\n")
                for layer, sparsity in sorted(results[name]["gating_sparsities"].items()):
                    f.write(f"  Layer {layer:2d}: {sparsity:.6f}\n")
                mean_sparsity = float(np.mean(list(results[name]["gating_sparsities"].values())))
                f.write(f"  Mean:     {mean_sparsity:.6f}\n")

            if results[name].get("lambda_stats"):
                f.write("\n" + "-" * 40 + "\n")
                f.write("Residual Lambda Statistics:\n")
                if 0 in results[name]["lambda_stats"]:
                    f.write("[Anchor Mode: Layer 0 has lambda parameters]\n")
                f.write("-" * 40 + "\n")
                for layer, stats in sorted(results[name]["lambda_stats"].items()):
                    f.write(f"  Layer {layer:2d}:\n")
                    for res_type, values in stats.items():
                        f.write(
                            f"    {res_type}: λ1={values['lambda1_mean']:.4f}, "
                            f"λ2={values['lambda2_mean']:.4f}, ratio={values['ratio']:.4f}\n"
                        )

    print(f"\nNumerical results saved to {results_file}")
    print("\n✓ Analysis complete!")
    
    del results
    cleanup_memory()


if __name__ == "__main__":
    main()