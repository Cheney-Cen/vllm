# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3VL embedding model.

This implements an embedding-only variant of Qwen3-VL that:

- Reuses the existing Qwen3VL vision tower and language model.
- Adds vision-to-text connector MLPs (``vision_projector`` and
  ``deepstack_projector_list``) whose weights are stored in
  ``model.vision_projector.*`` and ``model.deepstack_projector_list.*``.
- Adds a final projection head ``projector_256`` (weights under
  ``projector_256.*``) that maps the last-token hidden state to a
  256‑dimensional embedding.

The model implements the pooling interface so it can be used with the
``pooling`` runner and the ``LLM.embed(...)`` API.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import (
    get_seq_pooling_method,
)
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler

from .interfaces import MultiModalEmbeddings
from .interfaces_base import default_pooling_type
from .qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)
from .utils import WeightsMapper, _merge_multimodal_embeddings

logger = init_logger(__name__)


class Qwen3VLEmbeddingProjector(nn.Module):
    """Patch-merging style projector used in Qwen3VL embedding variants.

    This is structurally equivalent to the ``Qwen3VLVisionPatchMergerLego``
    module used in the original Qwen3-VL-Embedding implementation:

    - Optional LayerNorm on the input features.
    - Linear(hidden) -> GELU -> Linear(out_hidden_size).
    - Internal hidden width is ``hidden_size * spatial_merge_size**2``.

    The exact structure is important to match the checkpoint weights:

    - ``*.norm.*``
    - ``*.linear_fc1.*``
    - ``*.linear_fc2.*``
    """

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        vit_out_size: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
    ) -> None:
        super().__init__()
        # This matches the original implementation's internal expansion size.
        self.hidden_size = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.vit_out_size = vit_out_size

        norm_dim = self.vit_out_size if use_postshuffle_norm else vit_out_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.vit_out_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Align input dtype with module parameters (e.g. when model runs in fp16
        # on V100 but projector was created in float32).
        target_dtype = self.norm.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # The original code normalizes on a flattened view when
        # ``use_postshuffle_norm`` is enabled. We preserve that behavior here.
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.vit_out_size))
        else:
            x = self.norm(x)
        x = x.view(-1, self.vit_out_size)
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x


@default_pooling_type(seq_pooling_type="LAST", tok_pooling_type="ALL")
class Qwen3VLForEmbeddingModel(Qwen3VLForConditionalGeneration):
    """Qwen3-VL embedding model with 256-dim sequence embeddings.

    This class:
    - Reuses the multimodal parsing / vision tower / language model from
      :class:`Qwen3VLForConditionalGeneration`.
    - Adds vision connectors matching the checkpoint:
        * ``vision_projector`` (main visual stream)
        * ``deepstack_projector_list`` (multi-scale deepstack streams)
    - Adds ``projector_256`` and constructs a sequence pooler that:
        * takes the LAST token hidden state,
        * applies ``projector_256``,
        * L2-normalizes the output.
    """

    # Mark as a pooling model so vLLM routes it through the pooling runner.
    is_pooling_model = True

    # Extend the base Qwen3-VL weight prefix mapping with embedding-specific
    # projectors present in the checkpoint.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            # Embedding-specific connectors:
            "model.vision_projector.": "vision_projector.",
            "model.deepstack_projector_list.": "deepstack_projector_list.",
            # ``projector_256.*`` lives at top-level and does not need a prefix
            # remap; AutoWeightsLoader will match it directly.
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config: Qwen3VLConfig = vllm_config.model_config.hf_config
        text_hidden = config.text_config.hidden_size
        vit_out_size = config.vision_config.out_hidden_size
        spatial_merge_size = config.vision_config.spatial_merge_size
        num_deepstack_levels = len(
            getattr(config.vision_config, "deepstack_visual_indexes", [])
        )

        # Vision main-path projector: [vit_out_size] -> [text_hidden]
        self.vision_projector = Qwen3VLEmbeddingProjector(
            hidden_size=config.vision_config.hidden_size,
            out_hidden_size=text_hidden,
            vit_out_size=vit_out_size,
            spatial_merge_size=spatial_merge_size,
            use_postshuffle_norm=False,
        )

        # Deepstack projectors: one per deepstack level.
        self.deepstack_projector_list = nn.ModuleList(
            [
                Qwen3VLEmbeddingProjector(
                    hidden_size=config.vision_config.hidden_size,
                    out_hidden_size=text_hidden,
                    vit_out_size=vit_out_size,
                    spatial_merge_size=spatial_merge_size,
                    use_postshuffle_norm=True,
                )
                for _ in range(num_deepstack_levels)
            ]
        )

        # Final projection to 256-d embedding from the last token hidden state.
        self.projector_256 = Qwen3VLEmbeddingProjector(
            hidden_size=text_hidden,
            out_hidden_size=256,
            vit_out_size=text_hidden,
            spatial_merge_size=1,
            use_postshuffle_norm=True,
        )

        # Align projector dtypes with model (e.g. fp16 when V100 falls back from bf16).
        model_dtype = vllm_config.model_config.dtype
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype)
        self.vision_projector = self.vision_projector.to(dtype=model_dtype)
        self.deepstack_projector_list = self.deepstack_projector_list.to(
            dtype=model_dtype
        )
        self.projector_256 = self.projector_256.to(dtype=model_dtype)

        # Build the sequence pooler:
        # - pooling method: LAST token
        # - head: EmbeddingPoolerHead(projector_256 + L2 normalization)
        pooler_config = vllm_config.model_config.pooler_config
        head_dtype = vllm_config.model_config.head_dtype

        seq_pooling_type = (
            pooler_config.get_seq_pooling_type()
            if pooler_config is not None
            else "LAST"
        )

        self.pooler = SequencePooler(
            pooling=get_seq_pooling_method(seq_pooling_type),
            head=EmbeddingPoolerHead(
                projector=self.projector_256,
                head_dtype=head_dtype,
                activation=PoolerNormalize(),
            ),
        )

    # ---------------------------------------------------------------------
    # Deepstack handling
    # ---------------------------------------------------------------------

    def _compute_deepstack_embeds(
        self,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        is_multimodal: torch.Tensor,
    ) -> tuple[torch.Tensor, MultiModalEmbeddings]:
        """Project visual features to text space and build deepstack inputs.

        Compared to the base implementation, this version:
        - Starts from vision tower outputs of shape
          ``[num_tokens, vit_out * (1 + num_deepstack_levels)]``.
        - Splits into main and multi-scale parts.
        - Applies the vision_projector / deepstack_projector_list to obtain
          text_hidden-sized representations.
        - Uses ``_merge_multimodal_embeddings`` to scatter multi-scale features
          into per-position deepstack inputs for the language model.
        """

        if len(multimodal_embeddings) == 0:
            return (
                inputs_embeds.new_zeros(
                    0, self.deepstack_num_level, inputs_embeds.size(-1)
                ),
                multimodal_embeddings,
            )

        visual_lens = [len(x) for x in multimodal_embeddings]
        multimodal_embeddings_cat = torch.cat(multimodal_embeddings, dim=0)

        vit_out_size = self.config.vision_config.out_hidden_size
        num_levels = self.deepstack_num_level
        expected_dim = vit_out_size * (1 + num_levels)
        if multimodal_embeddings_cat.size(-1) != expected_dim:
            raise ValueError(
                "Unexpected vision embedding dimension: "
                f"got {multimodal_embeddings_cat.size(-1)}, "
                f"expected {expected_dim} "
                f"(vit_out_size={vit_out_size}, levels={num_levels})."
            )

        # Split main and multiscale chunks.
        main_raw, multiscale_raw = torch.split(
            multimodal_embeddings_cat,
            [vit_out_size, vit_out_size * num_levels],
            dim=-1,
        )

        # Project main visual features to text hidden size.
        main_proj = self.vision_projector(main_raw)  # [N, text_hidden]

        # Project multiscale features: reshape to [N, L, vit_out] then project
        # each level independently to [N, text_hidden].
        multiscale_raw = multiscale_raw.view(
            -1, num_levels, vit_out_size
        )  # [N, L, vit_out]

        level_projs = []
        for level_idx, projector in enumerate(self.deepstack_projector_list):
            level_feats = multiscale_raw[:, level_idx, :]  # [N, vit_out]
            level_proj = projector(level_feats)  # [N, text_hidden]
            level_projs.append(level_proj)

        if level_projs:
            # [N, L, text_hidden]
            levels_stack = torch.stack(level_projs, dim=1)
        else:
            # No deepstack configured.
            levels_stack = inputs_embeds.new_zeros(
                multimodal_embeddings_cat.size(0), 0, inputs_embeds.size(-1)
            )

        # Re-split main projections per multimodal item.
        multimodal_main = torch.split(main_proj, visual_lens, dim=0)

        # Flatten multiscale projections into [N, L * text_hidden] so that
        # we can use the existing `_merge_multimodal_embeddings` utility.
        text_hidden = inputs_embeds.size(-1)
        multiscale_flat = levels_stack.view(
            -1, num_levels * text_hidden
        )  # [N, L * H]
        multimodal_multiscale = torch.split(multiscale_flat, visual_lens, dim=0)

        # Build deepstack_input_embeds over the full sequence length and then
        # reshape to [L, seq_len, text_hidden] for consumption by Qwen3LLMModel.
        seq_len = inputs_embeds.size(0)
        deepstack_input_embeds = inputs_embeds.new_zeros(
            seq_len,
            num_levels * text_hidden,
        )

        deepstack_input_embeds = _merge_multimodal_embeddings(
            inputs_embeds=deepstack_input_embeds,
            multimodal_embeddings=multimodal_multiscale,
            is_multimodal=is_multimodal,
        )

        deepstack_input_embeds = deepstack_input_embeds.view(
            seq_len, num_levels, text_hidden
        )
        deepstack_input_embeds = deepstack_input_embeds.permute(1, 0, 2)

        return deepstack_input_embeds, multimodal_main

