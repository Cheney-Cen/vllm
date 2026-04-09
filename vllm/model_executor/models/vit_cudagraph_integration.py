"""
Integration helpers for ViT CUDA Graph Bucket Wrapper.

Provides functions to enable CUDA Graph acceleration on the ViT component
of Qwen3-VL models served through vLLM.

Usage from app.py or similar entry point::

    from vllm.model_executor.models.vit_cudagraph_integration import (
        enable_vit_cudagraph,
    )

    # After engine is fully initialized:
    enable_vit_cudagraph(engine)

Or with custom bucket configuration::

    enable_vit_cudagraph(
        engine,
        patch_buckets=[64, 128, 256, 512],
        segment_buckets=[1, 2, 3],
        max_patches_per_image=300,
    )
"""

from __future__ import annotations

import logging
import types

logger = logging.getLogger(__name__)


def enable_vit_cudagraph(
    engine,
    patch_buckets: list[int] | None = None,
    segment_buckets: list[int] | None = None,
    max_patches_per_image: int = 384,
    min_patches_per_image: int = 100,
) -> bool:
    """Enable ViT CUDA Graph bucket acceleration on a vLLM engine.

    This function:
    1. Locates the vision transformer (``self.visual``) on the model.
    2. Creates a ``ViTCUDAGraphBucketWrapper`` around it.
    3. Runs warmup to pre-capture all valid bucket CUDA Graphs.
    4. Monkey-patches ``_process_image_input`` to use the wrapper.

    Args:
        engine: A vLLM ``AsyncLLM``, ``LLMEngine``, or ``LLM`` instance.
        patch_buckets: Custom patch bucket sizes. Default: [64..1024].
        segment_buckets: Custom segment bucket sizes. Default: [1..5].
        max_patches_per_image: Upper bound on tokens per image after
            spatial merge. Used for bucket pruning. Default: 384.
        min_patches_per_image: Lower bound on tokens per image. Default: 100.

    Returns:
        True if CUDA Graph was successfully enabled, False otherwise.
    """
    from vllm.model_executor.models.vit_cudagraph_wrapper import (
        ViTCUDAGraphBucketWrapper,
    )

    try:
        # Step 1: Get model instance from engine
        model = _get_model_from_engine(engine)
        if model is None:
            logger.warning(
                "[ViT CUDAGraph] Could not locate model instance from engine. "
                "Make sure the engine is fully initialized."
            )
            return False

        # Step 2: Find the vision transformer
        visual = getattr(model, "visual", None)
        if visual is None:
            logger.warning(
                "[ViT CUDAGraph] Model has no 'visual' attribute. "
                "Is this a multimodal model?"
            )
            return False

        # Step 3: Get vllm_config
        vllm_config = _get_vllm_config(engine, model)
        if vllm_config is None:
            logger.warning("[ViT CUDAGraph] Could not locate vllm_config.")
            return False

        # Step 4: Create wrapper
        wrapper = ViTCUDAGraphBucketWrapper(
            vision_transformer=visual,
            vllm_config=vllm_config,
            patch_buckets=patch_buckets,
            segment_buckets=segment_buckets,
            max_patches_per_image=max_patches_per_image,
            min_patches_per_image=min_patches_per_image,
        )

        logger.info(
            "[ViT CUDAGraph] Bucket config: %d valid combinations "
            "(patches=%s, segments=%s)",
            len(wrapper.valid_buckets),
            wrapper.patch_buckets,
            wrapper.segment_buckets,
        )

        # Step 5: Run warmup (pre-capture all graphs)
        wrapper.warmup()

        # Step 6: Monkey-patch _process_image_input on the model
        _patch_process_image_input(model, wrapper)

        logger.info("[ViT CUDAGraph] Successfully enabled on model.")
        return True

    except Exception as e:
        logger.error(
            "[ViT CUDAGraph] Failed to enable: %s", e, exc_info=True
        )
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model_from_engine(engine):
    """Extract the model instance from various vLLM engine types.

    Supports multiple internal API paths:
    - AsyncLLM (v1): engine.engine_core.model_executor.model
    - AsyncLLM (v1): engine.engine_core.model_executor.driver_worker.model_runner.model
    - LLMEngine: engine.model_executor.driver_worker.model_runner.model
    - LLM (sync wrapper): engine.llm_engine -> recurse
    """
    # Path 1: v1 AsyncLLM with direct model attribute
    if hasattr(engine, "engine_core"):
        ec = engine.engine_core
        if hasattr(ec, "model_executor"):
            me = ec.model_executor
            if hasattr(me, "model"):
                return me.model
            if hasattr(me, "driver_worker"):
                worker = me.driver_worker
                if hasattr(worker, "model_runner"):
                    mr = worker.model_runner
                    if hasattr(mr, "model"):
                        return mr.model

    # Path 2: LLMEngine
    if hasattr(engine, "model_executor"):
        me = engine.model_executor
        if hasattr(me, "driver_worker"):
            worker = me.driver_worker
            if hasattr(worker, "model_runner"):
                mr = worker.model_runner
                if hasattr(mr, "model"):
                    return mr.model

    # Path 3: sync LLM wrapper
    if hasattr(engine, "llm_engine"):
        return _get_model_from_engine(engine.llm_engine)

    return None


def _get_vllm_config(engine, model=None):
    """Extract VllmConfig from engine or model."""
    # Try engine first
    if hasattr(engine, "vllm_config"):
        return engine.vllm_config
    if hasattr(engine, "engine_config"):
        return engine.engine_config

    # Try model
    if model is not None and hasattr(model, "vllm_config"):
        return model.vllm_config

    # Try sync wrapper
    if hasattr(engine, "llm_engine"):
        return _get_vllm_config(engine.llm_engine, model)

    return None


def _patch_process_image_input(model, wrapper):
    """Monkey-patch _process_image_input to use the CUDA Graph wrapper.

    The original code in Qwen3VLForConditionalGeneration._process_image_input:

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        with set_forward_context(None, self.vllm_config):
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

    The patched version:

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        image_embeds = self._vit_cg_wrapper(pixel_values, grid_thw=grid_thw)

    The wrapper handles its own ForwardContext (skip_compiled=True) internally.
    """
    from vllm.forward_context import set_forward_context

    # Store wrapper on model instance
    model._vit_cg_wrapper = wrapper

    # Save original method for potential restore
    model._original_process_image_input = model._process_image_input

    def _patched_process_image_input(self, image_input):
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input.get("type") == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)

            # Check for data parallel mode -- CUDA Graph wrapper doesn't
            # support DP sharded vision model; fall back to original path.
            if getattr(self, "use_data_parallel", False):
                from vllm.model_executor.models.vision import (
                    run_dp_sharded_mrope_vision_model,
                )
                with set_forward_context(None, self.vllm_config):
                    return run_dp_sharded_mrope_vision_model(
                        self.visual, pixel_values, grid_thw.tolist(),
                        rope_type="rope_3d",
                    )

            # Use CUDA Graph wrapper instead of self.visual directly.
            # The wrapper manages its own ForwardContext (skip_compiled=True).
            image_embeds = self._vit_cg_wrapper(
                pixel_values, grid_thw=grid_thw
            )

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    # Bind the patched method to the model instance
    model._process_image_input = types.MethodType(
        _patched_process_image_input, model
    )

    logger.info(
        "[ViT CUDAGraph] Patched _process_image_input on %s",
        type(model).__name__,
    )


def disable_vit_cudagraph(model) -> bool:
    """Restore the original _process_image_input (undo monkey-patch).

    Returns True if successfully restored, False if nothing to restore.
    """
    original = getattr(model, "_original_process_image_input", None)
    if original is None:
        return False

    model._process_image_input = original
    if hasattr(model, "_vit_cg_wrapper"):
        del model._vit_cg_wrapper
    if hasattr(model, "_original_process_image_input"):
        del model._original_process_image_input

    logger.info("[ViT CUDAGraph] Disabled and restored original method.")
    return True
