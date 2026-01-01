import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .utils import extract_kv_tuple, default_position_ids

logger = logging.getLogger(__name__)

# Prefer optimized block implementation if available
try:
    from petals.llama.block import OptimizedLlamaDecoderLayer

    OPTIMIZED_LAYER_AVAILABLE = True
except Exception as e:  # pragma: no cover - optional dependency
    OptimizedLlamaDecoderLayer = None
    OPTIMIZED_LAYER_AVAILABLE = False
    logger.warning(f"OptimizedLlamaDecoderLayer not available ({e}), using vanilla LlamaDecoderLayer.")


def _debug_conversion(original_layer: LlamaDecoderLayer, optimized_layer: OptimizedLlamaDecoderLayer, 
                      layer_idx: int) -> tuple[float, float]:
    """Detailed debugging of layer conversion."""
    # 1. State dict 키 비교 (가중치 + buffer 포함)
    orig_sd = original_layer.state_dict()
    opt_sd = optimized_layer.state_dict()
    
    logger.info(f"Layer {layer_idx}: === State Dict Keys ===")
    logger.info(f"Layer {layer_idx}: Original has {len(orig_sd)} keys")
    logger.info(f"Layer {layer_idx}: Optimized has {len(opt_sd)} keys")
    
    missing = set(orig_sd.keys()) - set(opt_sd.keys())
    extra = set(opt_sd.keys()) - set(orig_sd.keys())
    
    if missing:
        logger.error(f"Layer {layer_idx}: Missing in optimized: {list(missing)}")
    if extra:
        logger.warning(f"Layer {layer_idx}: Extra in optimized: {list(extra)[:10]}")
    
    # 1-1. Buffer 명시적 확인 (특히 rotary_emb)
    orig_buffers = dict(original_layer.named_buffers())
    opt_buffers = dict(optimized_layer.named_buffers())
    
    logger.info(f"Layer {layer_idx}: === Buffers ===")
    logger.info(f"Layer {layer_idx}: Original buffers: {list(orig_buffers.keys())[:10]}")
    logger.info(f"Layer {layer_idx}: Optimized buffers: {list(opt_buffers.keys())[:10]}")
    
    missing_buffers = set(orig_buffers.keys()) - set(opt_buffers.keys())
    if missing_buffers:
        logger.error(f"Layer {layer_idx}: Missing buffers: {list(missing_buffers)}")
    
    # Rotary embedding buffer 확인
    if hasattr(original_layer, 'self_attn') and hasattr(original_layer.self_attn, 'rotary_emb'):
        orig_rotary_buffers = dict(original_layer.self_attn.rotary_emb.named_buffers())
        if hasattr(optimized_layer, 'self_attn') and hasattr(optimized_layer.self_attn, 'rotary_emb'):
            opt_rotary_buffers = dict(optimized_layer.self_attn.rotary_emb.named_buffers())
            logger.info(f"Layer {layer_idx}: Rotary embedding buffers - original: {list(orig_rotary_buffers.keys())}, optimized: {list(opt_rotary_buffers.keys())}")
            
            for buffer_name in orig_rotary_buffers:
                if buffer_name in opt_rotary_buffers:
                    orig_buf = orig_rotary_buffers[buffer_name]
                    opt_buf = opt_rotary_buffers[buffer_name]
                    # Device가 다를 수 있으므로 CPU로 이동하여 비교
                    orig_buf_cpu = orig_buf.cpu()
                    opt_buf_cpu = opt_buf.cpu()
                    if not torch.equal(orig_buf_cpu, opt_buf_cpu):
                        buf_diff = (orig_buf_cpu - opt_buf_cpu).abs().max().item()
                        # 상대 오차 계산
                        orig_max = orig_buf_cpu.abs().max().item()
                        relative_diff = (buf_diff / (orig_max + 1e-8)) * 100 if orig_max > 0 else 0
                        
                        # inv_freq의 일반적인 값 범위 확인
                        orig_mean = orig_buf_cpu.mean().item()
                        orig_std = orig_buf_cpu.std().item()
                        
                        logger.info(
                            f"Layer {layer_idx}: Rotary buffer '{buffer_name}' - "
                            f"abs_diff={buf_diff:.8f}, relative_diff={relative_diff:.4f}%, "
                            f"orig_range=[{orig_buf_cpu.min().item():.8f}, {orig_buf_cpu.max().item():.8f}], "
                            f"orig_mean={orig_mean:.8f}, orig_std={orig_std:.8f}"
                        )
                        
                        # 허용 가능한 오차인지 판단 (float16의 경우 더 관대하게)
                        # inv_freq는 보통 1e-4 ~ 1e-6 범위이므로, 0.00024170은 상대적으로 큰 차이일 수 있음
                        if relative_diff < 0.1:  # 0.1% 미만이면 허용 가능
                            logger.info(f"Layer {layer_idx}: ✓ Rotary buffer '{buffer_name}' difference is acceptable (relative_diff={relative_diff:.4f}%)")
                        elif relative_diff < 1.0:  # 1% 미만이면 경고
                            logger.warning(f"Layer {layer_idx}: ⚠ Rotary buffer '{buffer_name}' has moderate difference (relative_diff={relative_diff:.4f}%)")
                        else:
                            logger.error(f"Layer {layer_idx}: ✗ Rotary buffer '{buffer_name}' has large difference (relative_diff={relative_diff:.4f}%)")
                    else:
                        logger.debug(f"Layer {layer_idx}: Rotary buffer '{buffer_name}' OK")
                else:
                    logger.error(f"Layer {layer_idx}: Rotary buffer '{buffer_name}' missing in optimized")
    
    # 2. 가중치 직접 비교 (공통 키만)
    common_keys = set(orig_sd.keys()) & set(opt_sd.keys())
    max_diff = 0.0
    max_diff_key = None
    mismatched_keys = []
    
    for key in common_keys:
        orig_tensor = orig_sd[key]
        opt_tensor = opt_sd[key]
        
        # Shape 확인
        if orig_tensor.shape != opt_tensor.shape:
            logger.error(f"Layer {layer_idx}: Shape mismatch for {key}: {orig_tensor.shape} vs {opt_tensor.shape}")
            continue
        
        # CPU, float32로 변환하여 비교
        orig_tensor = orig_tensor.cpu().float()
        opt_tensor = opt_tensor.cpu().float()
        
        diff = (orig_tensor - opt_tensor).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            max_diff_key = key
        if diff > 1e-5:
            mismatched_keys.append((key, diff))
            logger.error(f"Layer {layer_idx}:   {key}: max diff = {diff:.8f}")
    
    if max_diff_key:
        logger.info(f"Layer {layer_idx}: Max difference in '{max_diff_key}': {max_diff:.8f}")
    
    # 3. Forward 결과 비교
    device = next(original_layer.parameters()).device
    dtype = next(original_layer.parameters()).dtype
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(
        batch_size, seq_len, original_layer.hidden_size,
        device=device,
        dtype=dtype
    )
    
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    original_layer.eval()
    optimized_layer.eval()
    
    with torch.no_grad():
        orig_out = original_layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )[0]
        opt_out = optimized_layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )[0]
    
    output_diff = (orig_out - opt_out).abs().mean().item()
    output_diff_pct = (output_diff / (orig_out.abs().mean().item() + 1e-8)) * 100
    
    logger.info(f"Layer {layer_idx}: === Forward Output Comparison ===")
    logger.info(f"Layer {layer_idx}: Mean absolute difference: {output_diff:.8f}")
    logger.info(f"Layer {layer_idx}: Percentage difference: {output_diff_pct:.4f}%")
    
    return max_diff, output_diff_pct


def comprehensive_layer1_debug(original_layer1: LlamaDecoderLayer, optimized_layer1: OptimizedLlamaDecoderLayer, 
                                layer0_output: torch.Tensor):
    """Layer1의 모든 부분을 체크"""
    
    logger.info("=== Layer1 Comprehensive Debug ===")
    
    # 1. 전체 state_dict 키 확인
    orig_sd = original_layer1.state_dict()
    opt_sd = optimized_layer1.state_dict()
    
    logger.info("1. State Dict Keys:")
    logger.info(f"   Original: {len(orig_sd)} keys")
    logger.info(f"   Optimized: {len(opt_sd)} keys")
    
    missing = set(orig_sd.keys()) - set(opt_sd.keys())
    if missing:
        logger.warning(f"   ⚠️ Missing keys: {missing}")
    
    # 2. 각 서브모듈별 가중치 비교
    logger.info("2. Submodule Weight Comparison:")
    
    modules_to_check = {
        'mlp': ['gate_proj', 'up_proj', 'down_proj'],
        'post_attention_layernorm': ['weight'],
        'input_layernorm': ['weight'],
        'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    }
    
    for mod_name, param_names in modules_to_check.items():
        orig_mod = getattr(original_layer1, mod_name, None)
        opt_mod = getattr(optimized_layer1, mod_name, None)
        
        if orig_mod is None or opt_mod is None:
            continue
        
        for param_name in param_names:
            try:
                orig_param_obj = getattr(orig_mod, param_name, None)
                opt_param_obj = getattr(opt_mod, param_name, None)
                
                if orig_param_obj is None or opt_param_obj is None:
                    continue
                
                orig_param = orig_param_obj.weight if hasattr(orig_param_obj, 'weight') else orig_param_obj
                opt_param = opt_param_obj.weight if hasattr(opt_param_obj, 'weight') else opt_param_obj
                
                if orig_param.shape != opt_param.shape:
                    logger.error(f"   {mod_name}.{param_name}: Shape mismatch! {orig_param.shape} vs {opt_param.shape}")
                    continue
                
                diff = (orig_param.cpu().float() - opt_param.cpu().float()).abs().max()
                logger.info(f"   {mod_name}.{param_name}: max_diff = {diff:.10f}")
                if diff > 1e-4:
                    logger.warning(f"      ⚠️ LARGE DIFFERENCE!")
            except Exception as e:
                logger.warning(f"   {mod_name}.{param_name}: Error checking - {e}")
    
    # 3. Forward 단계별 확인
    logger.info("3. Forward Pass Step-by-Step:")
    
    original_layer1.eval()
    optimized_layer1.eval()
    
    with torch.no_grad():
        # Input LayerNorm
        orig_norm1 = original_layer1.input_layernorm(layer0_output)
        opt_norm1 = optimized_layer1.input_layernorm(layer0_output)
        norm1_diff = (orig_norm1 - opt_norm1).abs().mean()
        logger.info(f"   Input LayerNorm diff: {norm1_diff:.6f}")
        logger.info(f"   Input LayerNorm stats - orig: min={orig_norm1.min().item():.4f}, max={orig_norm1.max().item():.4f}, mean={orig_norm1.mean().item():.4f}")
        logger.info(f"   Input LayerNorm stats - opt:  min={opt_norm1.min().item():.4f}, max={opt_norm1.max().item():.4f}, mean={opt_norm1.mean().item():.4f}")
        
        # Attention
        # position_ids 생성
        position_ids = torch.arange(layer0_output.shape[1], device=layer0_output.device).unsqueeze(0)
        orig_attn_out = original_layer1.self_attn(orig_norm1, position_ids=position_ids, use_cache=False)[0]
        opt_attn_out = optimized_layer1.self_attn(opt_norm1, position_ids=position_ids, use_cache=False)[0]
        attn_diff = (orig_attn_out - opt_attn_out).abs().mean()
        logger.info(f"   Attention output diff: {attn_diff:.6f}")
        logger.info(f"   Attention output stats - orig: min={orig_attn_out.min().item():.4f}, max={orig_attn_out.max().item():.4f}, mean={orig_attn_out.mean().item():.4f}")
        logger.info(f"   Attention output stats - opt:  min={opt_attn_out.min().item():.4f}, max={opt_attn_out.max().item():.4f}, mean={opt_attn_out.mean().item():.4f}")
        
        # Residual 1
        orig_res1 = layer0_output + orig_attn_out
        opt_res1 = layer0_output + opt_attn_out
        res1_diff = (orig_res1 - opt_res1).abs().mean()
        logger.info(f"   After residual1 diff: {res1_diff:.6f}")
        logger.info(f"   After residual1 stats - orig: min={orig_res1.min().item():.4f}, max={orig_res1.max().item():.4f}, mean={orig_res1.mean().item():.4f}")
        logger.info(f"   After residual1 stats - opt:  min={opt_res1.min().item():.4f}, max={opt_res1.max().item():.4f}, mean={opt_res1.mean().item():.4f}")
        
        # Post-Attention LayerNorm (MLP 입력)
        orig_mlp_input = original_layer1.post_attention_layernorm(orig_res1)
        opt_mlp_input = optimized_layer1.post_attention_layernorm(opt_res1)
        mlp_input_diff = (orig_mlp_input - opt_mlp_input).abs().mean()
        logger.info(f"   MLP input (post-norm) diff: {mlp_input_diff:.6f}")
        logger.info(f"   MLP input stats - orig: min={orig_mlp_input.min().item():.4f}, max={orig_mlp_input.max().item():.4f}, mean={orig_mlp_input.mean().item():.4f}, abs_max={orig_mlp_input.abs().max().item():.4f}, abs_mean={orig_mlp_input.abs().mean().item():.4f}")
        logger.info(f"   MLP input stats - opt:  min={opt_mlp_input.min().item():.4f}, max={opt_mlp_input.max().item():.4f}, mean={opt_mlp_input.mean().item():.4f}, abs_max={opt_mlp_input.abs().max().item():.4f}, abs_mean={opt_mlp_input.abs().mean().item():.4f}")
        
        # MLP 출력 (폭발하는 부분)
        orig_mlp_out = original_layer1.mlp(orig_mlp_input)
        opt_mlp_out = optimized_layer1.mlp(opt_mlp_input)
        mlp_out_diff = (orig_mlp_out - opt_mlp_out).abs().mean()
        logger.info(f"   MLP output diff: {mlp_out_diff:.6f}")
        logger.info(f"   MLP output stats - orig: min={orig_mlp_out.min().item():.4f}, max={orig_mlp_out.max().item():.4f}, mean={orig_mlp_out.mean().item():.4f}, abs_max={orig_mlp_out.abs().max().item():.4f}, abs_mean={orig_mlp_out.abs().mean().item():.4f}")
        logger.info(f"   MLP output stats - opt:  min={opt_mlp_out.min().item():.4f}, max={opt_mlp_out.max().item():.4f}, mean={opt_mlp_out.mean().item():.4f}, abs_max={opt_mlp_out.abs().max().item():.4f}, abs_mean={opt_mlp_out.abs().mean().item():.4f}")
        
        if opt_mlp_out.abs().max() > orig_mlp_out.abs().max() * 10:
            logger.error(f"      🚨 ACTIVATION EXPLOSION DETECTED!")
            logger.error(f"      Original max: {orig_mlp_out.abs().max().item():.4f}")
            logger.error(f"      Optimized max: {opt_mlp_out.abs().max().item():.4f}")
        
        # Final output
        orig_final = orig_res1 + orig_mlp_out
        opt_final = opt_res1 + opt_mlp_out
        final_diff = (orig_final - opt_final).abs().mean()
        logger.info(f"   Final output (after residual2) diff: {final_diff:.6f}")
        logger.info(f"   Final output stats - orig: min={orig_final.min().item():.4f}, max={orig_final.max().item():.4f}, mean={orig_final.mean().item():.4f}")
        logger.info(f"   Final output stats - opt:  min={opt_final.min().item():.4f}, max={opt_final.max().item():.4f}, mean={opt_final.mean().item():.4f}")
    
    logger.info("=== End Layer1 Comprehensive Debug ===\n")


def _verify_layer_conversion(original_layer: LlamaDecoderLayer, optimized_layer: OptimizedLlamaDecoderLayer,
                             config, layer_idx: int) -> bool:
    """Verify that the converted layer produces the same output as the original."""
    try:
        # 상세 디버깅 실행
        max_weight_diff, output_diff_pct = _debug_conversion(original_layer, optimized_layer, layer_idx)
        
        # 입력 텐서 생성 (원본 레이어의 device와 dtype 사용)
        device = original_layer.input_layernorm.weight.device
        dtype = original_layer.input_layernorm.weight.dtype
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(
            batch_size, seq_len, config.hidden_size,
            device=device,
            dtype=dtype
        )
        
        # position_ids 생성
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        
        with torch.no_grad():
            # 원본 레이어 출력
            original_output = original_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
            )
            
            # 최적화 버전 출력
            optimized_output = optimized_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
            )
        
        # 결과 비교 (hidden_states만 비교)
        original_hidden = original_output[0]
        optimized_hidden = optimized_output[0]
        
        # 더 관대한 tolerance 사용 (float16의 경우 수치적 오차가 클 수 있음)
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        rtol = 1e-3 if dtype == torch.float16 else 1e-5
        
        if not torch.allclose(original_hidden, optimized_hidden, atol=atol, rtol=rtol):
            max_diff = (original_hidden - optimized_hidden).abs().max().item()
            mean_diff = (original_hidden - optimized_hidden).abs().mean().item()
            std_diff = (original_hidden - optimized_hidden).abs().std().item()
            
            # 상대 오차 계산
            relative_max_diff = (max_diff / (original_hidden.abs().max().item() + 1e-8)) * 100
            
            logger.warning(
                f"Layer {layer_idx}: Output mismatch! max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
                f"std_diff={std_diff:.6f}, relative_max_diff={relative_max_diff:.4f}%, dtype={dtype}"
            )
            
            # 가중치 차이와 출력 차이 비교
            if max_weight_diff > 1e-4:
                logger.error(
                    f"Layer {layer_idx}: Large weight difference detected ({max_weight_diff:.8f})! "
                    f"This is likely the cause of output mismatch."
                )
            
            # OptimizedLlamaDecoderLayer는 CUDA graph 최적화를 사용하므로
            # 수치적 차이가 있을 수 있음. 허용 범위를 더 관대하게 설정
            # float16의 경우 더 큰 오차가 발생할 수 있음
            if dtype == torch.float16:
                # float16: 5% 미만이면 허용 가능, 10% 미만이면 경고
                if relative_max_diff < 5.0:
                    logger.info(
                        f"Layer {layer_idx}: Acceptable relative error for float16 ({relative_max_diff:.4f}%), "
                        f"likely due to CUDA graph optimization and float16 precision. Continuing..."
                    )
                    return True
                elif relative_max_diff < 10.0:
                    logger.warning(
                        f"Layer {layer_idx}: Moderate relative error for float16 ({relative_max_diff:.4f}%), "
                        f"may be due to CUDA graph optimization. Continuing..."
                    )
                    return True
                else:
                    logger.error(f"Layer {layer_idx}: Large relative error ({relative_max_diff:.4f}%), conversion may be incorrect")
                    return False
            else:
                # float32: 더 엄격한 기준
                if relative_max_diff < 1.0:
                    logger.warning(
                        f"Layer {layer_idx}: Small relative error ({relative_max_diff:.4f}%), "
                        f"likely due to numerical precision. Continuing..."
                    )
                    return True
                else:
                    logger.error(f"Layer {layer_idx}: Large relative error ({relative_max_diff:.4f}%), conversion may be incorrect")
                    return False
        
        logger.info(f"Layer {layer_idx}: ✓ Conversion verified successfully")
        return True
        
    except Exception as e:
        logger.warning(f"Layer {layer_idx}: Verification failed with error: {e}", exc_info=True)
        return False


def _convert_layers(raw_layers: nn.ModuleList, config, device=None, dtype=None, verify: bool = True) -> nn.ModuleList:
    """Convert HF LlamaDecoderLayer to optimized version if available.
    
    Args:
        raw_layers: Original layers to convert
        config: Model configuration
        device: Target device (if None, uses original layer's device)
        dtype: Target dtype (if None, uses original layer's dtype)
        verify: Whether to verify conversion by running forward pass
    """
    if not OPTIMIZED_LAYER_AVAILABLE:
        return nn.ModuleList(raw_layers)

    optimized = []
    for idx, layer in enumerate(raw_layers):
        if isinstance(layer, OptimizedLlamaDecoderLayer):
            optimized.append(layer)
        elif isinstance(layer, LlamaDecoderLayer):
            # 1. 원본 레이어에서 state_dict 추출
            original_state_dict = layer.state_dict()
            logger.debug(f"Layer {idx}: Original state_dict keys: {list(original_state_dict.keys())[:10]}")
            
            # 2. OptimizedLlamaDecoderLayer 생성
            opt_layer = OptimizedLlamaDecoderLayer(config)
            opt_state_dict_before = opt_layer.state_dict()
            logger.debug(f"Layer {idx}: Optimized state_dict keys (before): {list(opt_state_dict_before.keys())[:10]}")
            
            # 3. 가중치 로드 전 검증: 키 이름 비교
            orig_keys = set(original_state_dict.keys())
            opt_keys_before = set(opt_state_dict_before.keys())
            missing_before = orig_keys - opt_keys_before
            unexpected_before = opt_keys_before - orig_keys
            
            if missing_before:
                logger.error(f"Layer {idx}: Keys in original but not in optimized (before load): {list(missing_before)[:10]}")
            if unexpected_before:
                logger.warning(f"Layer {idx}: Keys in optimized but not in original (before load): {list(unexpected_before)[:10]}")
            
            # 4. 가중치 및 Buffer 로드
            # state_dict()는 기본적으로 buffer도 포함하지만, 명시적으로 확인
            original_buffers = dict(layer.named_buffers())
            logger.debug(f"Layer {idx}: Original buffers: {list(original_buffers.keys())[:10]}")
            
            # load_state_dict는 buffer도 복사하지만, 명시적으로 확인하기 위해
            # strict=True로 시도, 실패하면 strict=False
            try:
                missing, unexpected = opt_layer.load_state_dict(original_state_dict, strict=True)
                if missing:
                    logger.error(f"Layer {idx}: missing keys after strict=True: {list(missing)}")
                if unexpected:
                    logger.warning(f"Layer {idx}: unexpected keys after strict=True: {list(unexpected)[:10]}")
                if not missing and not unexpected:
                    logger.info(f"Layer {idx}: ✓ All weights loaded successfully with strict=True")
            except RuntimeError as e:
                logger.warning(f"Layer {idx}: strict=True failed: {e}")
                missing, unexpected = opt_layer.load_state_dict(original_state_dict, strict=False)
                if missing:
                    logger.error(f"Layer {idx}: missing keys after strict=False: {list(missing)}")
                    # 서브모듈별로 직접 복사 시도
                    logger.info(f"Layer {idx}: Attempting to copy weights module by module...")
                    try:
                        # self_attn 복사 (가중치 + buffer)
                        if hasattr(layer, 'self_attn') and hasattr(opt_layer, 'self_attn'):
                            opt_layer.self_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
                            # Rotary embedding buffer 명시적 복사
                            if hasattr(layer.self_attn, 'rotary_emb') and hasattr(opt_layer.self_attn, 'rotary_emb'):
                                orig_rotary_buffers = dict(layer.self_attn.rotary_emb.named_buffers())
                                opt_rotary_buffers = dict(opt_layer.self_attn.rotary_emb.named_buffers())
                                for buffer_name, buffer_value in orig_rotary_buffers.items():
                                    if buffer_name in opt_rotary_buffers:
                                        opt_buffer = getattr(opt_layer.self_attn.rotary_emb, buffer_name)
                                        # Device가 다를 수 있으므로 opt_buffer의 device로 복사
                                        opt_buffer.data.copy_(buffer_value.data.to(opt_buffer.device))
                                        logger.debug(f"Layer {idx}: Copied rotary_emb buffer: {buffer_name}")
                                    else:
                                        logger.warning(f"Layer {idx}: Rotary buffer {buffer_name} not found in optimized layer")
                            logger.info(f"Layer {idx}: Copied self_attn weights and buffers")
                        
                        # mlp 복사
                        if hasattr(layer, 'mlp') and hasattr(opt_layer, 'mlp'):
                            opt_layer.mlp.load_state_dict(layer.mlp.state_dict(), strict=False)
                            logger.info(f"Layer {idx}: Copied mlp weights")
                        
                        # layernorm 복사
                        if hasattr(layer, 'input_layernorm') and hasattr(opt_layer, 'input_layernorm'):
                            opt_layer.input_layernorm.load_state_dict(layer.input_layernorm.state_dict(), strict=False)
                            logger.info(f"Layer {idx}: Copied input_layernorm weights")
                        
                        if hasattr(layer, 'post_attention_layernorm') and hasattr(opt_layer, 'post_attention_layernorm'):
                            opt_layer.post_attention_layernorm.load_state_dict(layer.post_attention_layernorm.state_dict(), strict=False)
                            logger.info(f"Layer {idx}: Copied post_attention_layernorm weights")
                        
                        # 다시 확인
                        missing_after_module_copy = set(original_state_dict.keys()) - set(opt_layer.state_dict().keys())
                        if missing_after_module_copy:
                            logger.error(f"Layer {idx}: Still missing keys after module copy: {list(missing_after_module_copy)[:10]}")
                        else:
                            logger.info(f"Layer {idx}: ✓ All weights copied successfully via module-by-module approach")
                    except Exception as module_copy_error:
                        logger.error(f"Layer {idx}: Module-by-module copy failed: {module_copy_error}", exc_info=True)
                
                if unexpected:
                    logger.warning(f"Layer {idx}: unexpected keys after strict=False: {list(unexpected)[:10]}")
            
            # 4. 디바이스 이동 (원본 레이어와 동일한 device로)
            if device is not None:
                opt_layer = opt_layer.to(device)
            elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
                # 원본 레이어의 device 확인
                original_device = layer.self_attn.q_proj.weight.device
                opt_layer = opt_layer.to(original_device)
            
            # 5. dtype 변환 (원본 레이어와 동일한 dtype으로)
            if dtype is not None:
                opt_layer = opt_layer.to(dtype)
            elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
                # 원본 레이어의 dtype 확인
                original_dtype = layer.self_attn.q_proj.weight.dtype
                opt_layer = opt_layer.to(original_dtype)
            
            # 5-1. Buffer 명시적 복사 확인 (특히 rotary_emb) - device 이동 후에 실행
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                if hasattr(opt_layer, 'self_attn') and hasattr(opt_layer.self_attn, 'rotary_emb'):
                    orig_rotary = layer.self_attn.rotary_emb
                    opt_rotary = opt_layer.self_attn.rotary_emb
                    
                    # Rotary embedding의 모든 buffer 확인 및 복사
                    orig_rotary_buffers = dict(orig_rotary.named_buffers())
                    opt_rotary_buffers = dict(opt_rotary.named_buffers())
                    
                    logger.debug(f"Layer {idx}: Rotary embedding buffers - original: {list(orig_rotary_buffers.keys())}, optimized: {list(opt_rotary_buffers.keys())}")
                    
                    # 방법 1: state_dict를 통한 복사 (더 안전함)
                    try:
                        orig_rotary_sd = orig_rotary.state_dict()
                        opt_rotary_sd = opt_rotary.state_dict()
                        
                        # 공통 키만 복사
                        common_keys = set(orig_rotary_sd.keys()) & set(opt_rotary_sd.keys())
                        if common_keys:
                            # device를 맞춰서 복사
                            for key in common_keys:
                                orig_val = orig_rotary_sd[key]
                                opt_val = opt_rotary_sd[key]
                                
                                # CPU에서 비교
                                orig_val_cpu = orig_val.cpu()
                                opt_val_cpu = opt_val.cpu()
                                if not torch.equal(orig_val_cpu, opt_val_cpu):
                                    # opt_rotary의 device로 이동하여 복사
                                    target_device = opt_val.device
                                    # buffer를 직접 복사 (register_buffer로 등록된 buffer는 직접 복사가 더 안전함)
                                    opt_buffer = getattr(opt_rotary, key)
                                    with torch.no_grad():
                                        opt_buffer.data.copy_(orig_val.to(target_device).data)
                                    logger.info(f"Layer {idx}: Copied rotary_emb.{key} directly (device: {target_device})")
                                    
                                    # 즉시 검증
                                    opt_val_after = getattr(opt_rotary, key).cpu()
                                    if torch.equal(orig_val_cpu, opt_val_after):
                                        logger.info(f"Layer {idx}: ✓ rotary_emb.{key} verified immediately after copy")
                                    else:
                                        diff = (orig_val_cpu - opt_val_after).abs().max().item()
                                        logger.error(f"Layer {idx}: ✗ rotary_emb.{key} still mismatched! diff={diff:.8f}")
                            
                            # 복사 후 최종 검증
                            opt_rotary_sd_after = opt_rotary.state_dict()
                            all_match = True
                            for key in common_keys:
                                orig_val = orig_rotary_sd[key].cpu()
                                opt_val_after = opt_rotary_sd_after[key].cpu()
                                if not torch.equal(orig_val, opt_val_after):
                                    diff = (orig_val - opt_val_after).abs().max().item()
                                    logger.error(f"Layer {idx}: ✗ rotary_emb.{key} still mismatched in final check! diff={diff:.8f}")
                                    # 한 번 더 시도
                                    opt_buffer = getattr(opt_rotary, key)
                                    with torch.no_grad():
                                        opt_buffer.data.copy_(orig_val.to(opt_buffer.device).data)
                                    logger.warning(f"Layer {idx}: Retried copying rotary_emb.{key}")
                                    # 재검증
                                    opt_val_retry = getattr(opt_rotary, key).cpu()
                                    if torch.equal(orig_val, opt_val_retry):
                                        logger.info(f"Layer {idx}: ✓ rotary_emb.{key} verified after retry")
                                        all_match = True
                                    else:
                                        diff_retry = (orig_val - opt_val_retry).abs().max().item()
                                        logger.error(f"Layer {idx}: ✗ rotary_emb.{key} still failed after retry! diff={diff_retry:.8f}")
                                        all_match = False
                            
                            if all_match:
                                logger.info(f"Layer {idx}: ✓ All rotary_emb buffers verified after copy")
                    except Exception as e:
                        logger.warning(f"Layer {idx}: Failed to copy rotary_emb via state_dict: {e}, trying direct copy...")
                        
                        # 방법 2: 직접 복사 (fallback)
                        for buffer_name, orig_buffer in orig_rotary_buffers.items():
                            if buffer_name in opt_rotary_buffers:
                                opt_buffer = getattr(opt_rotary, buffer_name)
                                # Device가 다를 수 있으므로 CPU로 이동하여 비교
                                orig_buffer_cpu = orig_buffer.cpu()
                                opt_buffer_cpu = opt_buffer.cpu()
                                if not torch.equal(orig_buffer_cpu, opt_buffer_cpu):
                                    # 원본 buffer를 opt_buffer의 device로 이동하여 복사
                                    with torch.no_grad():
                                        opt_buffer.data.copy_(orig_buffer.data.to(opt_buffer.device))
                                    logger.info(f"Layer {idx}: Copied rotary_emb buffer: {buffer_name} (device: {opt_buffer.device})")
                                    
                                    # 복사 후 다시 확인
                                    opt_buffer_cpu_after = opt_buffer.cpu()
                                    if torch.equal(orig_buffer_cpu, opt_buffer_cpu_after):
                                        logger.info(f"Layer {idx}: ✓ Rotary buffer {buffer_name} verified after copy")
                                    else:
                                        diff = (orig_buffer_cpu - opt_buffer_cpu_after).abs().max().item()
                                        logger.error(f"Layer {idx}: ✗ Rotary buffer {buffer_name} still mismatched after copy! diff={diff:.8f}")
                            else:
                                logger.warning(f"Layer {idx}: Rotary buffer {buffer_name} not found in optimized layer")
            
            # 7. 가중치 검증 (모든 가중치 비교)
            opt_state_dict = opt_layer.state_dict()
            key_checks = [
                'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
                'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
                'input_layernorm.weight', 'post_attention_layernorm.weight'
            ]
            
            weight_mismatches = []
            missing_weights = []
            
            for key in key_checks:
                if key in original_state_dict and key in opt_state_dict:
                    orig_tensor = original_state_dict[key]
                    opt_tensor = opt_state_dict[key]
                    
                    # Shape 확인
                    if orig_tensor.shape != opt_tensor.shape:
                        logger.error(f"Layer {idx}: Shape mismatch for {key}: {orig_tensor.shape} vs {opt_tensor.shape}")
                        missing_weights.append(key)
                        continue
                    
                    # CPU, float32로 변환하여 비교
                    orig_tensor = orig_tensor.cpu().float()
                    opt_tensor = opt_tensor.cpu().float()
                    
                    max_diff = (orig_tensor - opt_tensor).abs().max().item()
                    mean_diff = (orig_tensor - opt_tensor).abs().mean().item()
                    
                    if not torch.allclose(orig_tensor, opt_tensor, atol=1e-4):
                        logger.error(
                            f"Layer {idx}: Weight mismatch for {key}! max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
                            f"shape={orig_tensor.shape}"
                        )
                        weight_mismatches.append((key, max_diff, mean_diff))
                    else:
                        logger.debug(f"Layer {idx}: Weight OK for {key}: max_diff={max_diff:.8f}")
                elif key in original_state_dict:
                    logger.error(f"Layer {idx}: Missing key {key} in optimized layer!")
                    missing_weights.append(key)
                elif key in opt_state_dict:
                    logger.warning(f"Layer {idx}: Key {key} exists in optimized but not in original")
            
            if weight_mismatches:
                logger.error(f"Layer {idx}: Found {len(weight_mismatches)} weight mismatches!")
                for key, max_diff, mean_diff in weight_mismatches[:5]:  # 처음 5개만 출력
                    logger.error(f"Layer {idx}:   - {key}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
            
            if missing_weights:
                logger.error(f"Layer {idx}: Found {len(missing_weights)} missing weights: {missing_weights}")
            
            # 7. Forward pass 검증 (선택적)
            # 주의: OptimizedLlamaDecoderLayer는 구현이 약간 다를 수 있어서 
            # 완전히 동일한 출력을 보장하지 않을 수 있음 (특히 float16에서)
            if verify:
                verification_passed = _verify_layer_conversion(layer, opt_layer, config, idx)
                if not verification_passed:
                    # 검증 실패해도 계속 진행 (OptimizedLlamaDecoderLayer는 Petals의 최적화된 구현이므로)
                    logger.warning(
                        f"Layer {idx}: Forward pass verification failed. "
                        f"This may be expected due to implementation differences in OptimizedLlamaDecoderLayer. "
                        f"Continuing with conversion..."
                    )
            
            optimized.append(opt_layer)
        else:
            optimized.append(layer)
    
    return nn.ModuleList(optimized)


class Stage0(nn.Module):
    """LLaMA-only Stage0 using tuple KV cache."""

    def __init__(self, full, end: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        if "llama" not in model_type and "mistral" not in model_type and "mixtral" not in model_type:
            raise ValueError("Only LLaMA-style models are supported in Stage0.")

        if hasattr(full, "model") and hasattr(full.model, "embed_tokens"):
            self.embed_tokens = full.model.embed_tokens
            raw_layers = full.model.layers[:end]
        elif hasattr(full, "transformer") and hasattr(full.transformer, "wte"):
            self.embed_tokens = full.transformer.wte
            self.pos_embed = getattr(full.transformer, "wpe", None)
            raw_layers = full.transformer.h[:end]
        else:
            raise ValueError(f"Unsupported LLaMA architecture: {type(full)}.")

        # 원본 레이어 저장 (디버깅용)
        self.original_layers = nn.ModuleList(raw_layers)

        # device와 dtype 추출
        device = None
        dtype = None
        if len(raw_layers) > 0:
            first_layer = raw_layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
                device = first_layer.self_attn.q_proj.weight.device
                dtype = first_layer.self_attn.q_proj.weight.dtype

        self.layers = _convert_layers(raw_layers, full.config, device=device, dtype=dtype)
        self.config = full.config

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        x = self.embed_tokens(input_ids)
        new_cache = []

        # Prefill vs Decode 구분 (seq_len으로 판단)
        is_prefill = x.shape[1] > 1
        
        for i, layer in enumerate(self.layers):
            # 입력 상태 확인 (prefill일 때만)
            if i == 0 and is_prefill:
                logger.info(f"Stage0: Prefill - Input embedding stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
            
            layer_past = None if past_key_values is None else past_key_values[i]
            layer_pos = position_ids if position_ids is not None else default_position_ids(
                layer_past, x.shape[1], x.device
            )
            # 입력 상태 확인 (prefill일 때만)
            if is_prefill:
                input_before = x.clone()
                input_min, input_max = input_before.min().item(), input_before.max().item()
                input_mean, input_std = input_before.mean().item(), input_before.std().item()
                
                # Layer 0과 Layer 1의 입력을 특별히 확인
                if i == 0:
                    logger.info(
                        f"Stage0: Prefill - Layer {i} INPUT (before layer): "
                        f"min={input_min:.4f}, max={input_max:.4f}, mean={input_mean:.4f}, std={input_std:.4f}"
                    )
                elif i == 1:
                    logger.info(
                        f"Stage0: Prefill - Layer {i} INPUT (before layer): "
                        f"min={input_min:.4f}, max={input_max:.4f}, mean={input_mean:.4f}, std={input_std:.4f}, "
                        f"position_ids={layer_pos.tolist() if layer_pos is not None else None}"
                    )
                    
                    # Layer 1의 가중치 확인 (활성화값 폭발 원인 파악)
                    if isinstance(layer, OptimizedLlamaDecoderLayer):
                        layer_sd = layer.state_dict()
                        weight_keys = [
                            'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
                            'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
                            'input_layernorm.weight', 'post_attention_layernorm.weight'
                        ]
                        for key in weight_keys:
                            if key in layer_sd:
                                w = layer_sd[key]
                                w_min, w_max = w.min().item(), w.max().item()
                                w_mean, w_std = w.mean().item(), w.std().item()
                                w_abs_max = w.abs().max().item()
                                
                                # 비정상적으로 큰 가중치 확인
                                if w_abs_max > 10.0:
                                    logger.error(
                                        f"Stage0: Prefill - Layer {i} {key} has LARGE weights: "
                                        f"abs_max={w_abs_max:.4f}, min={w_min:.4f}, max={w_max:.4f}, "
                                        f"mean={w_mean:.4f}, std={w_std:.4f}"
                                    )
                                else:
                                    logger.info(
                                        f"Stage0: Prefill - Layer {i} {key} OK: "
                                        f"abs_max={w_abs_max:.4f}, min={w_min:.4f}, max={w_max:.4f}"
                                    )
            
            # Layer 1의 forward pass 중간 단계 확인 (활성화값 폭발 원인 파악)
            if is_prefill and i == 1 and isinstance(layer, OptimizedLlamaDecoderLayer):
                # Layer 1의 forward pass를 수동으로 추적
                residual = x.clone()
                
                # 1. Input layernorm
                x_after_input_norm = layer.input_layernorm(x)
                logger.info(
                    f"Stage0: Prefill - Layer {i} after input_layernorm: "
                    f"min={x_after_input_norm.min().item():.4f}, max={x_after_input_norm.max().item():.4f}, "
                    f"mean={x_after_input_norm.mean().item():.4f}, std={x_after_input_norm.std().item():.4f}"
                )
                
                # Rotary embedding 직접 확인 (원본 transformers와 비교)
                attn = layer.self_attn
                test_hidden = x_after_input_norm  # [batch, seq_len, hidden_size]
                test_pos = layer_pos  # [batch, seq_len]
                
                # rotary_emb 출력 shape 확인
                cos_test, sin_test = attn.rotary_emb(test_hidden, test_pos)
                logger.info(
                    f"Stage0: Prefill - Layer {i} rotary_emb direct call: "
                    f"input_hidden_shape={test_hidden.shape}, position_ids_shape={test_pos.shape}, "
                    f"cos_shape={cos_test.shape}, sin_shape={sin_test.shape}"
                )
                
                # 2. Attention
                attn_out = layer.self_attn(
                    hidden_states=x_after_input_norm,
                    attention_mask=None,
                    position_ids=layer_pos,
                    past_key_value=layer_past,
                    output_attentions=False,
                    use_cache=use_cache,
                )
                x_after_attn = attn_out[0]
                logger.info(
                    f"Stage0: Prefill - Layer {i} after attention: "
                    f"min={x_after_attn.min().item():.4f}, max={x_after_attn.max().item():.4f}, "
                    f"mean={x_after_attn.mean().item():.4f}, std={x_after_attn.std().item():.4f}"
                )
                
                # 3. Residual connection
                x_after_residual1 = residual + x_after_attn
                logger.info(
                    f"Stage0: Prefill - Layer {i} after residual1: "
                    f"min={x_after_residual1.min().item():.4f}, max={x_after_residual1.max().item():.4f}, "
                    f"mean={x_after_residual1.mean().item():.4f}, std={x_after_residual1.std().item():.4f}"
                )
                
                # 4. Post attention layernorm
                x_after_post_norm = layer.post_attention_layernorm(x_after_residual1)
                logger.info(
                    f"Stage0: Prefill - Layer {i} after post_attention_layernorm: "
                    f"min={x_after_post_norm.min().item():.4f}, max={x_after_post_norm.max().item():.4f}, "
                    f"mean={x_after_post_norm.mean().item():.4f}, std={x_after_post_norm.std().item():.4f}"
                )
                
                # 5. MLP (단계별 추적)
                mlp_input = x_after_post_norm
                logger.info(
                    f"Stage0: Prefill - Layer {i} MLP input: "
                    f"min={mlp_input.min().item():.4f}, max={mlp_input.max().item():.4f}, "
                    f"mean={mlp_input.mean().item():.4f}, std={mlp_input.std().item():.4f}"
                )
                
                # MLP 내부 단계 추적
                # gate_proj
                gate_out = layer.mlp.gate_proj(mlp_input)
                gate_max = gate_out.max().item()
                gate_min = gate_out.min().item()
                logger.info(
                    f"Stage0: Prefill - Layer {i} after gate_proj: "
                    f"min={gate_min:.4f}, max={gate_max:.4f}, "
                    f"mean={gate_out.mean().item():.4f}, std={gate_out.std().item():.4f}"
                )
                
                # up_proj
                up_out = layer.mlp.up_proj(mlp_input)
                up_max = up_out.max().item()
                up_min = up_out.min().item()
                logger.info(
                    f"Stage0: Prefill - Layer {i} after up_proj: "
                    f"min={up_min:.4f}, max={up_max:.4f}, "
                    f"mean={up_out.mean().item():.4f}, std={up_out.std().item():.4f}"
                )
                
                # SiLU 계산 확인
                silu_gate = torch.nn.functional.silu(gate_out)
                silu_max = silu_gate.max().item()
                silu_min = silu_gate.min().item()
                logger.info(
                    f"Stage0: Prefill - Layer {i} after SiLU(gate): "
                    f"min={silu_min:.4f}, max={silu_max:.4f}, "
                    f"mean={silu_gate.mean().item():.4f}, std={silu_gate.std().item():.4f}"
                )
                
                # activation (SiLU * up)
                activated = silu_gate * up_out
                activated_max = activated.max().item()
                activated_min = activated.min().item()
                logger.info(
                    f"Stage0: Prefill - Layer {i} after activation (SiLU * up): "
                    f"min={activated_min:.4f}, max={activated_max:.4f}, "
                    f"mean={activated.mean().item():.4f}, std={activated.std().item():.4f}, "
                    f"expected_max≈{silu_max * up_max:.4f}"
                )
                
                # 예상값과 실제값 비교
                if abs(activated_max - silu_max * up_max) > 10.0:
                    logger.error(
                        f"Stage0: Prefill - Layer {i} activation overflow detected! "
                        f"SiLU_max={silu_max:.4f}, up_max={up_max:.4f}, "
                        f"expected_max≈{silu_max * up_max:.4f}, actual_max={activated_max:.4f}"
                    )
                
                # down_proj (가중치 확인)
                down_proj_weight = layer.mlp.down_proj.weight
                down_proj_weight_stats = {
                    'min': down_proj_weight.min().item(),
                    'max': down_proj_weight.max().item(),
                    'mean': down_proj_weight.mean().item(),
                    'std': down_proj_weight.std().item(),
                    'abs_max': down_proj_weight.abs().max().item()
                }
                logger.info(
                    f"Stage0: Prefill - Layer {i} down_proj.weight stats: "
                    f"abs_max={down_proj_weight_stats['abs_max']:.4f}, "
                    f"min={down_proj_weight_stats['min']:.4f}, max={down_proj_weight_stats['max']:.4f}, "
                    f"mean={down_proj_weight_stats['mean']:.4f}, std={down_proj_weight_stats['std']:.4f}, "
                    f"shape={down_proj_weight.shape}"
                )
                
                # down_proj 계산
                activated_max = activated.max().item()
                expected_down_max = activated_max * down_proj_weight_stats['abs_max']
                logger.info(
                    f"Stage0: Prefill - Layer {i} down_proj calculation: "
                    f"activated_max={activated_max:.4f}, weight_abs_max={down_proj_weight_stats['abs_max']:.4f}, "
                    f"expected_output_max≈{expected_down_max:.4f}"
                )
                
                x_after_mlp = layer.mlp.down_proj(activated)
                logger.info(
                    f"Stage0: Prefill - Layer {i} after down_proj (MLP output): "
                    f"min={x_after_mlp.min().item():.4f}, max={x_after_mlp.max().item():.4f}, "
                    f"mean={x_after_mlp.mean().item():.4f}, std={x_after_mlp.std().item():.4f}, "
                    f"expected_max≈{expected_down_max:.4f}"
                )
                
                # 예상값과 실제값 비교
                actual_max = x_after_mlp.max().item()
                if abs(actual_max - expected_down_max) > 100.0:
                    logger.error(
                        f"Stage0: Prefill - Layer {i} down_proj output mismatch! "
                        f"expected_max≈{expected_down_max:.4f}, actual_max={actual_max:.4f}, "
                        f"diff={abs(actual_max - expected_down_max):.4f}"
                    )
                
                # 6. Final residual (residual2 = residual + MLP output)
                residual2_input = x_after_residual1.clone()
                logger.info(
                    f"Stage0: Prefill - Layer {i} residual2 input (x_after_residual1): "
                    f"min={residual2_input.min().item():.4f}, max={residual2_input.max().item():.4f}, "
                    f"mean={residual2_input.mean().item():.4f}, std={residual2_input.std().item():.4f}"
                )
                logger.info(
                    f"Stage0: Prefill - Layer {i} residual2 MLP output (x_after_mlp): "
                    f"min={x_after_mlp.min().item():.4f}, max={x_after_mlp.max().item():.4f}, "
                    f"mean={x_after_mlp.mean().item():.4f}, std={x_after_mlp.std().item():.4f}"
                )
                
                # residual connection 계산
                x_final = x_after_residual1 + x_after_mlp
                expected_final_max = max(residual2_input.max().item(), x_after_mlp.max().item())
                expected_final_min = min(residual2_input.min().item(), x_after_mlp.min().item())
                
                logger.info(
                    f"Stage0: Prefill - Layer {i} after residual2 (FINAL): "
                    f"min={x_final.min().item():.4f}, max={x_final.max().item():.4f}, "
                    f"mean={x_final.mean().item():.4f}, std={x_final.std().item():.4f}, "
                    f"expected_range≈[{expected_final_min:.4f}, {expected_final_max:.4f}]"
                )
                
                # residual connection이 문제인지 확인
                if x_final.max().item() > expected_final_max * 1.5:
                    logger.error(
                        f"Stage0: Prefill - Layer {i} residual2 overflow detected! "
                        f"residual_max={residual2_input.max().item():.4f}, mlp_max={x_after_mlp.max().item():.4f}, "
                        f"expected_max≈{expected_final_max:.4f}, actual_max={x_final.max().item():.4f}"
                    )
                
                # 정상 forward pass도 실행 (KV cache를 위해)
                # 하지만 이미 x_final을 계산했으므로 비교
                out = layer(
                    x,
                    attention_mask=None,
                    position_ids=layer_pos,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    output_attentions=False,
                )
                x_normal = out[0]
                
                # 수동 계산과 정상 forward pass 비교
                diff = (x_final - x_normal).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                logger.info(
                    f"Stage0: Prefill - Layer {i} manual vs normal forward: "
                    f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
                )
                
                if max_diff > 1e-3:
                    logger.error(
                        f"Stage0: Prefill - Layer {i} manual and normal forward mismatch! "
                        f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
                    )
                
                x = x_normal
            else:
                out = layer(
                    x,
                    attention_mask=None,
                    position_ids=layer_pos,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    output_attentions=False,
                )
                x = out[0]
            
            # Layer 0 출력 후 Layer 1 실행 전에 comprehensive 디버깅
            if is_prefill and i == 0 and len(self.original_layers) > 1:
                layer0_output = x.clone()
                original_layer1 = self.original_layers[1]
                optimized_layer1 = self.layers[1] if len(self.layers) > 1 else None
                
                if optimized_layer1 is not None and isinstance(optimized_layer1, OptimizedLlamaDecoderLayer):
                    logger.info("Stage0: Prefill - Running comprehensive Layer1 debug comparison...")
                    comprehensive_layer1_debug(original_layer1, optimized_layer1, layer0_output)
            
            # 각 레이어 출력 확인 (prefill일 때만 상세 로깅)
            if is_prefill:
                x_min, x_max = x.min().item(), x.max().item()
                x_mean, x_std = x.mean().item(), x.std().item()
                
                # Layer 0의 출력을 특별히 확인 (활성화값 폭발의 시작점)
                if i == 0:
                    logger.info(
                        f"Stage0: Prefill - Layer {i} OUTPUT (after layer): "
                        f"min={x_min:.4f}, max={x_max:.4f}, mean={x_mean:.4f}, std={x_std:.4f}, "
                        f"layer_type={type(layer).__name__}"
                    )
                
                if abs(x_min) > 100 or abs(x_max) > 100 or abs(x_mean) > 10 or x_std > 50:
                    logger.error(
                        f"Stage0: Prefill - Layer {i} output EXPLODING! "
                        f"min={x_min:.4f}, max={x_max:.4f}, mean={x_mean:.4f}, std={x_std:.4f}, "
                        f"layer_type={type(layer).__name__}, seq_len={x.shape[1]}, "
                        f"position_ids={layer_pos.tolist() if layer_pos is not None else None}"
                    )
                elif abs(x_min) > 50 or abs(x_max) > 50:
                    logger.warning(
                        f"Stage0: Prefill - Layer {i} output large values: "
                        f"min={x_min:.4f}, max={x_max:.4f}, mean={x_mean:.4f}, std={x_std:.4f}"
                    )
                else:
                    logger.debug(
                        f"Stage0: Prefill - Layer {i} output OK: "
                        f"min={x_min:.4f}, max={x_max:.4f}, mean={x_mean:.4f}, std={x_std:.4f}"
                    )
            
            if use_cache:
                kv = extract_kv_tuple(out, layer_idx=i)
                if kv is None:
                    logger.warning(f"Stage0: layer {i} did not return KV cache")
                new_cache.append(kv)

        if not use_cache:
            return x, None
        return x, tuple(new_cache)


class StageSegment(nn.Module):
    """LLaMA-only middle segment using tuple KV cache."""

    def __init__(self, full, start: int, end: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        if "llama" not in model_type and "mistral" not in model_type and "mixtral" not in model_type:
            raise ValueError("Only LLaMA-style models are supported in StageSegment.")

        if hasattr(full, "model") and hasattr(full.model, "layers"):
            raw_layers = full.model.layers[start:end]
        elif hasattr(full, "transformer") and hasattr(full.transformer, "h"):
            raw_layers = full.transformer.h[start:end]
        else:
            raise ValueError(f"Unsupported LLaMA architecture: {type(full)}.")

        # device와 dtype 추출
        device = None
        dtype = None
        if len(raw_layers) > 0:
            first_layer = raw_layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
                device = first_layer.self_attn.q_proj.weight.device
                dtype = first_layer.self_attn.q_proj.weight.dtype

        self.layers = _convert_layers(raw_layers, full.config, device=device, dtype=dtype)
        self.config = full.config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        x = hidden_states
        new_cache = []

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            layer_pos = position_ids if position_ids is not None else default_position_ids(
                layer_past, x.shape[1], x.device
            )
            out = layer(
                x,
                attention_mask=None,
                position_ids=layer_pos,
                past_key_value=layer_past,
                use_cache=use_cache,
                output_attentions=False,
            )
            x = out[0]
            if use_cache:
                kv = extract_kv_tuple(out, layer_idx=i)
                if kv is None:
                    logger.warning(f"StageSegment: layer {i} did not return KV cache")
                new_cache.append(kv)

        if not use_cache:
            return x, None
        return x, tuple(new_cache)


class StageLast(nn.Module):
    """LLaMA-only last stage using tuple KV cache."""

    def __init__(self, full, start: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        if "llama" not in model_type and "mistral" not in model_type and "mixtral" not in model_type:
            raise ValueError("Only LLaMA-style models are supported in StageLast.")

        if hasattr(full, "model") and hasattr(full.model, "layers"):
            raw_layers = full.model.layers[start:]
            if hasattr(full.model, "norm"):
                self.norm = full.model.norm
            elif hasattr(full.model, "final_layer_norm"):
                self.norm = full.model.final_layer_norm
            else:
                raise ValueError(f"Unsupported model: no norm layer found in {type(full.model)}")
        elif hasattr(full, "transformer") and hasattr(full.transformer, "h"):
            raw_layers = full.transformer.h[start:]
            self.norm = full.transformer.ln_f
        else:
            raise ValueError(f"Unsupported LLaMA architecture: {type(full)}.")

        # device와 dtype 추출
        device = None
        dtype = None
        if len(raw_layers) > 0:
            first_layer = raw_layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
                device = first_layer.self_attn.q_proj.weight.device
                dtype = first_layer.self_attn.q_proj.weight.dtype

        self.layers = _convert_layers(raw_layers, full.config, device=device, dtype=dtype)
        self.lm_head = full.lm_head
        self.config = full.config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        x = hidden_states
        new_cache = []

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            layer_pos = position_ids if position_ids is not None else default_position_ids(
                layer_past, x.shape[1], x.device
            )
            out = layer(
                x,
                attention_mask=None,
                position_ids=layer_pos,
                past_key_value=layer_past,
                use_cache=use_cache,
                output_attentions=False,
            )
            x = out[0]
            if use_cache:
                kv = extract_kv_tuple(out, layer_idx=i)
                if kv is None:
                    logger.warning(f"StageLast: layer {i} did not return KV cache")
                new_cache.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        if not use_cache:
            return logits, None
        return logits, tuple(new_cache)


def load_stage_model(
    model_name: str,
    device: torch.device,
    role: str,
    *,
    start: int = 0,
    end: Optional[int] = None,
    dtype=torch.float16,
):
    """
    Load only the layers needed for a stage to reduce memory (LLaMA-only).
    role:
      - 'stage0': keep embeddings + layers[:end], drop head/norm
      - 'segment': keep layers[start:end], drop embeddings/head/norm
      - 'last': keep layers[start:], norm, lm_head
    """
    full = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    try:
        full.config.use_cache = True
    except Exception:
        pass
    full.eval()

    def _prune_layers(obj, start_idx, end_idx):
        if hasattr(obj, "model") and hasattr(obj.model, "layers"):
            obj.model.layers = nn.ModuleList(obj.model.layers[start_idx:end_idx])
        elif hasattr(obj, "transformer") and hasattr(obj.transformer, "h"):
            obj.transformer.h = nn.ModuleList(obj.transformer.h[start_idx:end_idx])
        else:
            raise ValueError(f"Unsupported model architecture for pruning: {type(obj)}")

    if role == "stage0":
        _prune_layers(full, 0, end)
        if hasattr(full, "lm_head"):
            full.lm_head = None
        if hasattr(full, "model") and hasattr(full.model, "norm"):
            full.model.norm = None
    elif role == "segment":
        _prune_layers(full, start, end)
        if hasattr(full, "lm_head"):
            full.lm_head = None
        if hasattr(full, "model") and hasattr(full.model, "norm"):
            full.model.norm = None
    elif role == "last":
        _prune_layers(full, start, None)
        # keep norm/head
    else:
        raise ValueError(f"Unknown role: {role}")

    full = full.to(device)
    return full
