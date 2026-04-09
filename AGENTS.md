# AGENTS.md - Project Memory for vLLM (Lazada Fork)

## Project Overview

Customized fork of vLLM for Lazada's multimodal (image tagging) use cases, focused on **Qwen3-VL** model family optimizations.

- **Upstream base**: vLLM v0.16.x (releases/v0.16.0)
- **Active branch**: `lazada_image_tag_0_17`
- **Previous branch**: `lazada_image_tag_0_16` (stable)
- **GPU**: NVIDIA L20 (SM89, FlashAttention v2 only; v3 requires Hopper SM90+)

## Build & Development

- **Python**: >=3.10, <3.14 | **PyTorch**: 2.10.0
- **Install**: `pip install -e .`
- **Lint**: `ruff check vllm/` | **Format**: `ruff format --check vllm/` | **Type check**: `mypy vllm/`
- **Test**: `pytest tests/`
- **Startup**: `boot.sh` in app_source directory; kill existing processes before restart (EngineCore subprocess can hold ~36GB VRAM)

## Custom Development History

### Phase 1: Qwen3-VL Embedding Model

**Commits**: `83a845ab2`, `1761e1903`

- `qwen3_vl_embedding.py`: `Qwen3VLForEmbeddingModel` extending `Qwen3VLForConditionalGeneration`
  - `vision_projector` / `deepstack_projector_list` / `projector_256` (256-dim final embedding)
  - Pooling: LAST token + L2 normalization
- dtype fix: projectors must align with model precision (fp16 on V100 fallback from bf16)

### Phase 2: ViT torch.compile

**Commit**: `9c00da5f9`

- `@support_torch_compile` on `PatchEmbed`, `VisionBlock`, `PatchMerger` with `dynamic_arg_dims` on dim 0
- Controlled by `compile_mm_encoder: true` in `compilation_config`
- ViT uses packed sequence: `x.shape[0]` = sum of all images' patches, dim 0 marked dynamic, compiled graph reuses across sizes

**Bug fixed - AOT cache collision**: `merger` and `deepstack_merger_list` share `Qwen3_VisionPatchMerger` class but different shapes. AOT hash based on `forward.__qualname__` was identical. **Fix**: Created subclass `Qwen3_VisionDeepstackPatchMerger` with overridden `forward()` for distinct cache hash.

### Phase 3: Language Model FULL CUDA Graph for Pooling Models

**Key finding**: Default `max_cudagraph_capture_size` = `min(max_num_seqs * 2, 512)` = 256, too small for embedding models (no decode phase, every request is prefill with hundreds of tokens). Configured capture range 64-1024, 73 graphs.

**Attention backend for FULL graph**:
- FlashAttention v2: `UNIFORM_BATCH` only (no mixed-batch FULL graph)
- Triton Attention: `ALWAYS` (supports FULL graph) -> chosen for L20

**Attempted & reverted**: Changed attention type to ENCODER_ONLY + `CausalNoCacheAttention`. Caused `seq_lens` semantic mismatch during CUDA graph capture (Triton kernel OOB access). **Decision**: Keep standard DECODER attention with KV Cache (wastes some VRAM but avoids all backend complexity).

**Final minimal changes**:
- `vllm/config/vllm.py`: Removed pooling model FULL graph hard restriction
- `app_source/app.py`: `attention_config={"backend": "TRITON_ATTN"}` + `cudagraph_mode="FULL"` + `compile_mm_encoder: true`

### Phase 4: ViT CUDA Graph Bucket Wrapper

**Commits**: `9c00da5f9`, `fb6dbf3f1`

After FULL graph for backbone, **bottleneck shifted to ViT** (embedding model runs ViT every request, no encoder cache reuse like generation models). ViT runs via `EncoderRunner`, outside backbone's CUDA graph path.

**Challenge**: 2 dynamic dimensions (total patches AND number of segments), unlike backbone's 1 dimension. Solution: bucket-based wrapper.

- `vit_cudagraph_wrapper.py`: Splits ViT into eager part (patch_embed + pos_embed + rotary) and graph part (blocks + merger). Bucket key: `(num_patches, num_segments)`. `max_seqlen` NOT in key (kernels read actual lengths from cu_seqlens).
- `vit_cudagraph_integration.py`: `enable_vit_cudagraph(engine)` API, monkey-patches `_process_image_input`
- `vit_cudagraph_worker_extension.py`: For distributed setup via `collective_rpc`

## Key Architecture

```
Qwen3VLForConditionalGeneration
  +-- visual (patch_embed, blocks x28, merger, deepstack_merger_list, rotary_emb)
  +-- language_model (Qwen3LLMModel)

Qwen3VLForEmbeddingModel (extends above)
  +-- vision_projector / deepstack_projector_list / projector_256
  +-- pooler (LAST token + L2 normalize)
```

Pooling model uses **DECODER attention** (not ENCODER_ONLY) to avoid CUDA graph capture issues.

## File Map

| File | Description |
|------|-------------|
| `vllm/model_executor/models/qwen3_vl.py` | torch.compile decorators, DeepstackMerger subclass |
| `vllm/model_executor/models/qwen3_vl_embedding.py` | Embedding variant (new) |
| `vllm/model_executor/models/vit_cudagraph_wrapper.py` | ViT CUDA Graph bucket wrapper (new) |
| `vllm/model_executor/models/vit_cudagraph_integration.py` | CUDA Graph enable/disable API (new) |
| `vllm/model_executor/models/vit_cudagraph_worker_extension.py` | Worker extension (new) |
| `vllm/multimodal/inputs.py` | Concat fix, GPU transfer optimization |
| `vllm/v1/attention/ops/vit_attn_wrappers.py` | ViT attention (compile compat) |
| `vllm/config/vllm.py` | Removed pooling FULL graph restriction |

## Debugging Lessons

1. **AOT cache collision**: Same class + different init params -> same `__qualname__` -> same cache hash. Fix: subclass with overridden `forward()`.
2. **Cache invalidation**: After reverting structural code changes, must clear AOT cache before restart.
3. **ENCODER_ONLY + CUDA graph**: `seq_lens` semantics differ from DECODER; `fill_(max_query_len)` during capture causes kernel OOB. Keep DECODER type for pooling.
4. **`max_cudagraph_capture_size`**: Default 256 is for decode (1 token/req). Embedding models need explicit config.
5. **FA2 vs Triton**: FA2 only `UNIFORM_BATCH` for CUDA graph. Need Triton for FULL graph on L20.

## Constraints

- ViT CUDA Graph: FLASH_ATTN and TRITON_ATTN only (not SDPA/FLASHINFER)
- ViT CUDA Graph: single-GPU only
- torch.compile and CUDA Graph mutually exclusive for ViT
- FA v3 requires Hopper SM90+, unavailable on L20
- Pooling models must use DECODER attention type

## Potential Future Work

1. Piecewise CUDA Graph for ViT via Inductor partition (`maybe_use_cudagraph_partition_wrapper`)
2. Multi-GPU ViT CUDA Graph support
