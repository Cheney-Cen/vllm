"""
ViT CUDA Graph Bucket Wrapper for Qwen3-VL Vision Transformer.

This module provides CUDA Graph acceleration for the ViT (Vision Transformer)
component of Qwen3-VL models. It works by:

1. Splitting the ViT forward pass into:
   - Eager part: patch_embed + pos_embed + rotary_emb (dynamic shapes)
   - Graph part: transformer blocks + merger (fixed shapes per bucket)

2. Defining bucket combinations (num_patches x num_segments) and pre-capturing
   a CUDA Graph for each valid bucket during warmup.

3. At inference time, padding inputs to the nearest bucket and replaying
   the corresponding CUDA Graph.

Design decisions:
- Compatible with both FLASH_ATTN and TRITON_ATTN backends.
  TORCH_SDPA and FLASHINFER are NOT supported (SDPA uses Python loops
  with .tolist(); FlashInfer has cuDNN graph nesting issues).
- Uses ForwardContext.skip_compiled=True to bypass torch.compile during
  both CUDA Graph capture AND replay. The two acceleration paths
  (torch.compile vs CUDA Graph) are mutually exclusive.
- Single-GPU only (no special NCCL graph pool handling for now).
- max_seqlen is NOT included in the bucket descriptor key because
  attention kernels (Triton/Flash) read actual seq lengths from
  cu_seqlens tensor; max_seqlen only affects grid sizing and a
  conservative upper bound is always safe.
"""

from __future__ import annotations

import bisect
import dataclasses
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from vllm.forward_context import (
    ForwardContext,
    create_forward_context,
    override_forward_context,
)
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bucket descriptor (CUDA Graph cache key)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ViTBucketDescriptor:
    """Cache key for a captured CUDA Graph.

    max_seqlen is intentionally NOT included because:
    - Both Triton and Flash Attention use it only for grid sizing
    - The kernel reads actual seq lengths from cu_seqlens / b_seq_len
    - A captured graph with max_seqlen >= actual is always correct
      (just launches some extra no-op thread blocks)
    """
    num_patches: int    # padded total token count (dim-0 of hidden_states)
    num_segments: int   # padded segment count (len(cu_seqlens) - 1)


# ---------------------------------------------------------------------------
# Captured graph entry
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class CapturedViTGraph:
    """Holds a captured CUDA Graph and its associated static buffers."""
    graph: torch.cuda.CUDAGraph
    # Static input buffers (copy real data here before replay)
    static_hidden_states: torch.Tensor     # [num_patches, 1, hidden_size]
    static_cu_seqlens: torch.Tensor        # [num_segments + 1]
    static_rotary_cos: torch.Tensor        # [num_patches, rotary_dim]
    static_rotary_sin: torch.Tensor        # [num_patches, rotary_dim]
    static_max_seqlen: torch.Tensor        # scalar int32
    static_sequence_lengths: Optional[torch.Tensor]  # [num_segments] or None
    # Static output buffer (read results from here after replay)
    static_output: torch.Tensor            # [num_patches_merged, out_dim]


# ---------------------------------------------------------------------------
# Default bucket configuration
# ---------------------------------------------------------------------------
# Per-image token count after spatial merge (merge_size=2) is typically ~200.
# We bound the valid bucket space so that:
#   min_patches_per_image * num_segments <= num_patches
#       <= max_patches_per_image * num_segments
MAX_PATCHES_PER_IMAGE = 384
MIN_PATCHES_PER_IMAGE = 32

DEFAULT_PATCH_BUCKETS = [64, 128, 256, 384, 512, 768, 1024]
DEFAULT_SEGMENT_BUCKETS = [1, 2, 3, 4, 5]


def build_valid_bucket_combinations(
    patch_buckets: list[int],
    segment_buckets: list[int],
    max_patches_per_image: int = MAX_PATCHES_PER_IMAGE,
    min_patches_per_image: int = MIN_PATCHES_PER_IMAGE,
) -> list[ViTBucketDescriptor]:
    """Build valid (num_patches, num_segments) bucket combinations.

    Prunes unreachable combinations:
    - num_patches > num_segments * max_patches_per_image  (impossible)
    - num_patches < num_segments * min_patches_per_image  (impossible)
    """
    combos = []
    for seg in segment_buckets:
        max_feasible = seg * max_patches_per_image
        min_feasible = seg * min_patches_per_image
        for pat in patch_buckets:
            if pat > max_feasible:
                continue
            if pat < min_feasible:
                continue
            combos.append(ViTBucketDescriptor(num_patches=pat, num_segments=seg))
    return combos


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------
class ViTCUDAGraphBucketWrapper(nn.Module):
    """Wraps a Qwen3_VisionTransformer to add CUDA Graph bucket support.

    Usage::

        wrapper = ViTCUDAGraphBucketWrapper(model.visual, vllm_config)
        wrapper.warmup()  # call once at startup

        # Then replace model._process_image_input to use wrapper.forward()
    """

    def __init__(
        self,
        vision_transformer: nn.Module,
        vllm_config,
        patch_buckets: list[int] | None = None,
        segment_buckets: list[int] | None = None,
        max_patches_per_image: int = MAX_PATCHES_PER_IMAGE,
        min_patches_per_image: int = 100,
    ):
        super().__init__()
        self.vit = vision_transformer
        self.vllm_config = vllm_config

        self.patch_buckets = sorted(patch_buckets or DEFAULT_PATCH_BUCKETS)
        self.segment_buckets = sorted(segment_buckets or DEFAULT_SEGMENT_BUCKETS)
        self.max_patches_per_image = max_patches_per_image
        self.min_patches_per_image = min_patches_per_image

        # Build valid bucket combinations
        self.valid_buckets = build_valid_bucket_combinations(
            self.patch_buckets, self.segment_buckets,
            self.max_patches_per_image, self.min_patches_per_image,
        )

        # Graph cache
        self._graph_cache: dict[ViTBucketDescriptor, CapturedViTGraph] = {}
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._warmed_up = False

        # Validate attention backend compatibility
        self._validate_attn_backend()

    def _validate_attn_backend(self):
        """Check that the ViT attention backend is CUDA Graph compatible.

        Compatible:   FLASH_ATTN, TRITON_ATTN
        Incompatible: TORCH_SDPA  (Python loops + .tolist())
                      FLASHINFER  (cuDNN graph nesting)
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        attn_backend = getattr(self.vit, "attn_backend", None)
        if attn_backend is None:
            logger.warning(
                "[ViT CUDAGraph] Could not determine ViT attention backend. "
                "Proceeding anyway -- capture may fail at warmup."
            )
            return

        compatible = {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
        }
        if attn_backend not in compatible:
            raise ValueError(
                f"[ViT CUDAGraph] Attention backend {attn_backend} is not "
                f"compatible with CUDA Graph capture. "
                f"Compatible backends: {[b.name for b in compatible]}. "
                f"Hint: set mm_encoder_attn_backend='TRITON_ATTN' or "
                f"'FLASH_ATTN' in engine args, or ensure flash-attn / triton "
                f"is installed so auto-selection picks a compatible backend."
            )
        logger.info(
            "[ViT CUDAGraph] ViT attention backend: %s (compatible)",
            attn_backend.name,
        )

    # -----------------------------------------------------------------
    # Bucket selection helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _find_bucket(actual: int, buckets: list[int]) -> int | None:
        """Find the smallest bucket value >= actual."""
        idx = bisect.bisect_left(buckets, actual)
        if idx < len(buckets):
            return buckets[idx]
        return None

    def _compute_actual_merged_patches(self, grid_thw_list: list[list[int]]) -> int:
        """Compute total token count after spatial merge."""
        merge = self.vit.spatial_merge_size
        total = 0
        for t, h, w in grid_thw_list:
            total += t * (h // merge) * (w // merge)
        return total

    def _compute_actual_segments(self, grid_thw_list: list[list[int]]) -> int:
        """Compute number of segments (each temporal slice is one segment)."""
        return sum(t for t, _, _ in grid_thw_list)

    def _select_bucket(
        self, grid_thw_list: list[list[int]]
    ) -> ViTBucketDescriptor | None:
        """Select the best matching bucket for a given input.

        Returns None if the input exceeds all buckets or if no matching
        graph has been captured.
        """
        actual_patches = self._compute_actual_merged_patches(grid_thw_list)
        actual_segments = self._compute_actual_segments(grid_thw_list)

        bucket_patches = self._find_bucket(actual_patches, self.patch_buckets)
        bucket_segments = self._find_bucket(actual_segments, self.segment_buckets)

        if bucket_patches is None or bucket_segments is None:
            return None

        desc = ViTBucketDescriptor(
            num_patches=bucket_patches,
            num_segments=bucket_segments,
        )

        if desc not in self._graph_cache:
            return None

        return desc

    # -----------------------------------------------------------------
    # ViT blocks execution (shared between capture and eager fallback)
    # -----------------------------------------------------------------
    def _run_vit_blocks(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        max_seqlen: torch.Tensor | int,
        sequence_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Execute transformer blocks + merger.

        This is the "graphable" portion of the ViT forward pass.
        All inputs have fixed shapes per bucket.

        Called within a ForwardContext(skip_compiled=True) so that
        @support_torch_compile decorated blocks call .forward() directly.
        """
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.vit.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_cos,
                rotary_pos_emb_sin=rotary_sin,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )
            if layer_num in self.vit.deepstack_visual_indexes:
                idx = self.vit.deepstack_visual_indexes.index(layer_num)
                feat = self.vit.deepstack_merger_list[idx](hidden_states)
                deepstack_feature_lists.append(feat)

        hidden_states = self.vit.merger(hidden_states)
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists, dim=1
        )
        return hidden_states

    # -----------------------------------------------------------------
    # Graph capture
    # -----------------------------------------------------------------
    def _create_skip_compiled_context(self) -> ForwardContext:
        """Create a ForwardContext with skip_compiled=True.

        This ensures that @support_torch_compile decorated modules
        (VisionBlock, PatchEmbed, PatchMerger, etc.) bypass the
        torch.compile path and call .forward() directly.
        """
        return create_forward_context(
            attn_metadata=None,
            vllm_config=self.vllm_config,
            skip_compiled=True,
        )

    def _capture_one_graph(self, desc: ViTBucketDescriptor) -> CapturedViTGraph:
        """Capture a CUDA Graph for one bucket descriptor.

        Steps:
        1. Allocate static input/output buffers sized to the bucket.
        2. Run one warmup forward to let PyTorch allocate intermediates.
        3. Capture the graph.

        IMPORTANT: desc.num_patches is the POST-merge token count (after
        spatial merge). But transformer blocks process PRE-merge pixels.
        For spatial_merge_size=2, pre_merge = post_merge * 4.
        All input buffers (hidden_states, cu_seqlens, rotary) must use
        the pre-merge size. Only the output buffer uses post-merge size.
        """
        device = self.vit.device
        dtype = self.vit.dtype
        hidden_size = self.vit.hidden_size
        merge_size = self.vit.spatial_merge_size
        merge_factor = merge_size * merge_size  # typically 4
        num_pixels = desc.num_patches * merge_factor  # pre-merge input size

        # -- Allocate static input buffers (pre-merge sizes) --
        static_hidden = torch.zeros(
            num_pixels, 1, hidden_size,
            dtype=dtype, device=device,
        )

        # Build uniform cu_seqlens for the warmup/capture run
        # cu_seqlens tracks pre-merge pixel boundaries per segment
        static_cu_seqlens = torch.zeros(
            desc.num_segments + 1,
            dtype=torch.int32, device=device,
        )
        pixels_per_seg = num_pixels // max(desc.num_segments, 1)
        for i in range(desc.num_segments):
            static_cu_seqlens[i + 1] = min(
                (i + 1) * pixels_per_seg, num_pixels
            )
        static_cu_seqlens[-1] = num_pixels

        # Rotary embeddings (pre-merge: one per pixel)
        rotary_dim = self.vit.rotary_pos_emb.rotary_dim
        static_cos = torch.zeros(
            num_pixels, rotary_dim,
            dtype=dtype, device=device,
        )
        static_sin = torch.zeros(
            num_pixels, rotary_dim,
            dtype=dtype, device=device,
        )

        # max_seqlen: conservative upper bound = total pre-merge pixels.
        # IMPORTANT: We use a Python int (not a tensor) to avoid
        # GPU→CPU sync (.item()) during CUDA Graph capture.
        max_seqlen_int = num_pixels
        # Keep a tensor version for the struct (bookkeeping only)
        static_max_seqlen = torch.tensor(
            max_seqlen_int, dtype=torch.int32, device=device,
        )

        # sequence_lengths: needed for FlashInfer backend only.
        # Since we only support FLASH_ATTN and TRITON_ATTN, this is None.
        # But we keep the plumbing for future extensibility.
        static_sequence_lengths = MMEncoderAttention.maybe_compute_sequence_lengths(
            self.vit.attn_backend,
            # Need numpy cu_seqlens for the static method
            static_cu_seqlens.cpu().numpy(),
        )
        if static_sequence_lengths is not None:
            static_sequence_lengths = torch.from_numpy(
                static_sequence_lengths
            ).to(device, non_blocking=True)

        # Maybe recompute cu_seqlens (FlashInfer dual-segment format)
        # For FLASH_ATTN / TRITON_ATTN this is a no-op (returns unchanged).
        cu_seqlens_np = static_cu_seqlens.cpu().numpy()
        cu_seqlens_recomputed = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.vit.attn_backend,
            cu_seqlens_np,
            hidden_size,
            self.vit.tp_size,
        )
        static_cu_seqlens = torch.from_numpy(cu_seqlens_recomputed).to(
            device, non_blocking=True
        )

        # -- Create ForwardContext that skips torch.compile --
        skip_ctx = self._create_skip_compiled_context()

        # -- Monkey-patch attention wrappers to bypass torch.ops dispatch --
        # During CUDA Graph capture, we cannot go through torch.ops.vllm.*
        # because the custom op schema enforces Tensor? for max_seqlen,
        # but we need to pass a Python int to avoid .item() sync.
        # Patching the module-level names in mm_encoder_attention redirects
        # _forward_fa / _forward_triton to call the raw implementation.
        import vllm.model_executor.layers.attention.mm_encoder_attention as _mm_mod
        from vllm.v1.attention.ops.vit_attn_wrappers import (
            flash_attn_maxseqlen_wrapper as _raw_fa_impl,
            triton_attn_wrapper as _raw_triton_impl,
        )

        _saved_fa = _mm_mod.vit_flash_attn_wrapper
        _saved_triton = _mm_mod.vit_triton_attn_wrapper
        _mm_mod.vit_flash_attn_wrapper = _raw_fa_impl
        _mm_mod.vit_triton_attn_wrapper = _raw_triton_impl

        try:
            # -- Warmup run (allocates all intermediate buffers) --
            with torch.no_grad(), override_forward_context(skip_ctx):
                _ = self._run_vit_blocks(
                    static_hidden, static_cu_seqlens,
                    static_cos, static_sin,
                    max_seqlen_int,
                    static_sequence_lengths,
                )

            # -- Capture CUDA Graph --
            graph = torch.cuda.CUDAGraph()
            with (
                torch.no_grad(),
                override_forward_context(skip_ctx),
                torch.cuda.graph(graph, pool=self._graph_pool),
            ):
                static_output = self._run_vit_blocks(
                    static_hidden, static_cu_seqlens,
                    static_cos, static_sin,
                    max_seqlen_int,
                    static_sequence_lengths,
                )
        finally:
            # Restore original wrappers
            _mm_mod.vit_flash_attn_wrapper = _saved_fa
            _mm_mod.vit_triton_attn_wrapper = _saved_triton

        return CapturedViTGraph(
            graph=graph,
            static_hidden_states=static_hidden,
            static_cu_seqlens=static_cu_seqlens,
            static_rotary_cos=static_cos,
            static_rotary_sin=static_sin,
            static_max_seqlen=static_max_seqlen,
            static_sequence_lengths=static_sequence_lengths,
            static_output=static_output,
        )

    # -----------------------------------------------------------------
    # Warmup: pre-capture all valid bucket combinations
    # -----------------------------------------------------------------
    @torch.inference_mode()
    def warmup(self):
        """Pre-capture CUDA Graphs for all valid bucket combinations.

        Call this once after model loading, before serving any requests.
        """
        logger.info(
            "[ViT CUDAGraph] Starting warmup: %d bucket combinations",
            len(self.valid_buckets),
        )

        for i, desc in enumerate(self.valid_buckets):
            try:
                self._graph_cache[desc] = self._capture_one_graph(desc)
                logger.info(
                    "[ViT CUDAGraph] Captured %d/%d: patches=%d, segments=%d",
                    i + 1, len(self.valid_buckets),
                    desc.num_patches, desc.num_segments,
                )
            except Exception as e:
                logger.warning(
                    "[ViT CUDAGraph] Failed to capture patches=%d, segments=%d: %s",
                    desc.num_patches, desc.num_segments, e,
                )

        self._warmed_up = True
        logger.info(
            "[ViT CUDAGraph] Warmup complete. %d graphs cached.",
            len(self._graph_cache),
        )

    # -----------------------------------------------------------------
    # Eager path: patch_embed + pos_embed + rotary + cu_seqlens
    # -----------------------------------------------------------------
    def _run_eager_prefix(
        self,
        x: torch.Tensor,
        grid_thw_list: list[list[int]],
    ) -> tuple[
        torch.Tensor,  # hidden_states [num_patches, 1, hidden_size]
        torch.Tensor,  # cu_seqlens (after maybe_recompute for backend)
        torch.Tensor,  # rotary_cos
        torch.Tensor,  # rotary_sin
        torch.Tensor,  # max_seqlen (scalar)
        Optional[torch.Tensor],  # sequence_lengths (None for non-FlashInfer)
    ]:
        """Execute the dynamic prefix of the ViT forward pass.

        Mirrors Qwen3_VisionTransformer.forward() lines 583-619 exactly,
        producing the same tensors that transformer blocks expect.
        """
        device = self.vit.device
        dtype = self.vit.dtype

        hidden_states = x.to(device=device, dtype=dtype, non_blocking=True)
        hidden_states = self.vit.patch_embed(hidden_states)

        pos_embeds = self.vit.fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds

        rotary_cos, rotary_sin = self.vit.rot_pos_emb(grid_thw_list)

        # Compute cu_seqlens (numpy, matching upstream exactly)
        grid_thw_np = np.array(grid_thw_list, dtype=np.int32)
        cu_seqlens_np = np.repeat(
            grid_thw_np[:, 1] * grid_thw_np[:, 2], grid_thw_np[:, 0]
        ).cumsum(axis=0, dtype=np.int32)
        cu_seqlens_np = np.concatenate(
            [np.zeros(1, dtype=np.int32), cu_seqlens_np]
        )

        # sequence_lengths (FlashInfer only, None for FLASH_ATTN/TRITON_ATTN)
        sequence_lengths = MMEncoderAttention.maybe_compute_sequence_lengths(
            self.vit.attn_backend, cu_seqlens_np
        )
        if sequence_lengths is not None:
            sequence_lengths = torch.from_numpy(sequence_lengths).to(
                device, non_blocking=True
            )

        # max_seqlen
        max_seqlen_val = MMEncoderAttention.compute_max_seqlen(
            self.vit.attn_backend, cu_seqlens_np
        )
        max_seqlen = torch.tensor(
            max_seqlen_val, dtype=torch.int32, device=device,
        )

        # Maybe recompute cu_seqlens for FlashInfer format
        cu_seqlens_np = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.vit.attn_backend,
            cu_seqlens_np,
            self.vit.hidden_size,
            self.vit.tp_size,
        )
        cu_seqlens = torch.from_numpy(cu_seqlens_np).to(
            device, non_blocking=True,
        )

        hidden_states = hidden_states.unsqueeze(1)

        return hidden_states, cu_seqlens, rotary_cos, rotary_sin, max_seqlen, sequence_lengths

    # -----------------------------------------------------------------
    # Padding helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _pad_dim0(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad tensor along dim-0 with zeros to target_len."""
        actual = tensor.shape[0]
        if actual >= target_len:
            return tensor
        pad = torch.zeros(
            target_len - actual, *tensor.shape[1:],
            dtype=tensor.dtype, device=tensor.device,
        )
        return torch.cat([tensor, pad], dim=0)

    @staticmethod
    def _pad_cu_seqlens(
        cu_seqlens: torch.Tensor,
        target_segments: int,
    ) -> torch.Tensor:
        """Pad cu_seqlens to have (target_segments + 1) entries.

        Extra segments have zero length (start == previous end).
        """
        actual_segments = cu_seqlens.shape[0] - 1
        if actual_segments >= target_segments:
            return cu_seqlens
        last_val = cu_seqlens[-1]
        padding = last_val.expand(target_segments - actual_segments)
        return torch.cat([cu_seqlens, padding], dim=0)

    # -----------------------------------------------------------------
    # Forward: select graph path or eager fallback
    # -----------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
    ) -> torch.Tensor:
        """Drop-in replacement for Qwen3_VisionTransformer.forward().

        The entire forward is wrapped in a skip_compiled ForwardContext so
        that ALL @support_torch_compile-decorated sub-modules (patch_embed,
        VisionBlock, PatchMerger, etc.) bypass the torch.compile path.
        CUDA Graph replaces torch.compile as the acceleration mechanism.
        """
        if isinstance(grid_thw, torch.Tensor):
            grid_thw_list = grid_thw.tolist()
        else:
            grid_thw_list = grid_thw

        # Set skip_compiled=True for the entire forward pass.
        # This ensures:
        # 1. patch_embed (eager prefix) bypasses torch.compile
        # 2. VisionBlock/Merger (graph path) bypass torch.compile
        # 3. Eager fallback also bypasses torch.compile
        skip_ctx = self._create_skip_compiled_context()
        with override_forward_context(skip_ctx):
            return self._forward_inner(x, grid_thw_list)

    def _forward_inner(
        self,
        x: torch.Tensor,
        grid_thw_list: list[list[int]],
    ) -> torch.Tensor:
        """Inner forward logic, called within skip_compiled context."""
        # --- Eager prefix (always runs, dynamic shapes) ---
        (
            hidden_states, cu_seqlens, rotary_cos, rotary_sin,
            max_seqlen, sequence_lengths,
        ) = self._run_eager_prefix(x, grid_thw_list)

        actual_merged = self._compute_actual_merged_patches(grid_thw_list)

        # --- Try CUDA Graph path ---
        desc = self._select_bucket(grid_thw_list) if self._warmed_up else None

        if desc is not None:
            entry = self._graph_cache[desc]

            # desc.num_patches is POST-merge. Static buffers are allocated
            # with PRE-merge size = num_patches * merge_size^2.
            merge_size = self.vit.spatial_merge_size
            num_pixels = desc.num_patches * merge_size * merge_size

            # Copy actual data into static input buffers
            actual_pixels = hidden_states.shape[0]  # pre-merge pixel count

            entry.static_hidden_states[:actual_pixels].copy_(hidden_states)
            if actual_pixels < num_pixels:
                entry.static_hidden_states[actual_pixels:].zero_()

            # Pad and copy cu_seqlens
            padded_cu_seqlens = self._pad_cu_seqlens(
                cu_seqlens, desc.num_segments
            )
            entry.static_cu_seqlens[:padded_cu_seqlens.shape[0]].copy_(
                padded_cu_seqlens
            )

            # Pad and copy rotary embeddings (pre-merge pixel count)
            actual_rot_len = rotary_cos.shape[0]
            entry.static_rotary_cos[:actual_rot_len].copy_(rotary_cos)
            if actual_rot_len < num_pixels:
                entry.static_rotary_cos[actual_rot_len:].zero_()
            entry.static_rotary_sin[:actual_rot_len].copy_(rotary_sin)
            if actual_rot_len < num_pixels:
                entry.static_rotary_sin[actual_rot_len:].zero_()

            # max_seqlen is baked in at capture time (= num_pixels,
            # a conservative upper bound). The attention kernel reads
            # actual seq lengths from cu_seqlens, so this is safe.

            # sequence_lengths: for FLASH_ATTN / TRITON_ATTN this is None,
            # so no copy needed. If it were non-None (future FlashInfer
            # support), we'd need to copy here as well.

            # Replay the captured graph
            entry.graph.replay()

            # Extract output: only the actual merged patches, not padding
            output = entry.static_output[:actual_merged].clone()
            return output

        else:
            # --- Eager fallback ---
            logger.debug(
                "[ViT CUDAGraph] Fallback to eager: patches=%d, segments=%d",
                hidden_states.shape[0],
                cu_seqlens.shape[0] - 1,
            )
            output = self._run_vit_blocks(
                hidden_states, cu_seqlens,
                rotary_cos, rotary_sin,
                max_seqlen,
                sequence_lengths,
            )
            return output
