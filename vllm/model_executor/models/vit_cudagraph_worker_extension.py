"""
Worker extension for ViT CUDA Graph Bucket support.

This module defines a WorkerExtension class that gets injected into the
vLLM worker via the `worker_extension_cls` mechanism. It provides a
`enable_vit_cudagraph` method callable via `collective_rpc`.

Usage in AsyncEngineArgs:
    engine_args = AsyncEngineArgs(
        ...
        worker_extension_cls="vllm.model_executor.models.vit_cudagraph_worker_extension.ViTCUDAGraphWorkerExtension",
    )

Then after engine creation:
    await engine.collective_rpc("enable_vit_cudagraph")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ViTCUDAGraphWorkerExtension:
    """Worker extension that enables ViT CUDA Graph on the worker process.

    When injected into the worker via `worker_extension_cls`, the worker
    gains the `enable_vit_cudagraph` method which can be called via
    `collective_rpc("enable_vit_cudagraph")`.

    Inside the worker, `self` is the Worker instance, which has:
    - self.model_runner          (GPUModelRunner)
    - self.model_runner.model    (the loaded nn.Module)
    - self.model_runner.vllm_config
    """

    def enable_vit_cudagraph(
        self,
        patch_buckets: list[int] | None = None,
        segment_buckets: list[int] | None = None,
        max_patches_per_image: int = 384,
        min_patches_per_image: int = 100,
    ) -> bool:
        """Enable ViT CUDA Graph acceleration on this worker's model.

        Called via: await engine.collective_rpc("enable_vit_cudagraph")
        """
        from vllm.model_executor.models.vit_cudagraph_wrapper import (
            ViTCUDAGraphBucketWrapper,
        )
        from vllm.model_executor.models.vit_cudagraph_integration import (
            _patch_process_image_input,
        )

        try:
            model = self.model_runner.get_model()
            vllm_config = self.model_runner.vllm_config

            visual = getattr(model, "visual", None)
            if visual is None:
                logger.warning(
                    "[ViT CUDAGraph] Model has no 'visual' attribute."
                )
                return False

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

            # Pre-capture all CUDA Graphs
            wrapper.warmup()

            # Monkey-patch _process_image_input
            _patch_process_image_input(model, wrapper)

            logger.info(
                "[ViT CUDAGraph] Successfully enabled on worker (rank=%s).",
                getattr(self, "rank", "?"),
            )
            return True

        except Exception as e:
            logger.error(
                "[ViT CUDAGraph] Failed to enable: %s", e, exc_info=True
            )
            return False

    def disable_vit_cudagraph(self) -> bool:
        """Disable ViT CUDA Graph and restore original _process_image_input."""
        from vllm.model_executor.models.vit_cudagraph_integration import (
            disable_vit_cudagraph,
        )

        try:
            model = self.model_runner.get_model()
            return disable_vit_cudagraph(model)
        except Exception as e:
            logger.error(
                "[ViT CUDAGraph] Failed to disable: %s", e, exc_info=True
            )
            return False
