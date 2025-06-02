from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
import matplotlib.pyplot as plt


def plot_attention_grid(
    attentions: List[torch.FloatTensor],
    *,
    max_layers: int | None = None,
    max_heads: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Visualize attentions as a ``layer × head`` grid and optionally save as PDF.

    Parameters
    ----------
    attentions
        Tuple/list length == ``n_layers``; each tensor shape =
        ``(batch, heads, seq_len, seq_len)``.
    max_layers, max_heads
        Slice the visualization if given.
    save_path
        Path to a *.pdf* file to save the figure. Directory is created if needed.
    show
        Whether to display the figure interactively. Use ``False`` inside a
        headless training job.
    """

    n_layers_all = len(attentions)
    n_heads_all = attentions[0].size(1)
    n_layers = n_layers_all if max_layers is None else min(n_layers_all, max_layers)
    n_heads = n_heads_all if max_heads is None else min(n_heads_all, max_heads)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 2, n_layers * 2), squeeze=False)

    for layer_idx in range(n_layers):
        layer_att = attentions[layer_idx][0]  # first example in batch
        for head_idx in range(n_heads):
            ax = axes[layer_idx][head_idx]
            attn = layer_att[head_idx].detach().cpu().numpy()
            im = ax.imshow(attn, cmap="Blues", interpolation="nearest", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"L{layer_idx}H{head_idx}", fontsize=6)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


class AttentionPlotCallback(TrainerCallback):
    """🤗 Trainer callback that plots attentions on each evaluation step."""

    def __init__(
        self,
        example_texts: Iterable[str],
        *,
        output_dir: str | Path = "attention_plots",
        max_layers: int | None = None,
        max_heads: int | None = None,
        show: bool = False,
    ):
        self.example_texts = list(example_texts)
        self.output_dir = Path(output_dir)
        self.max_layers = max_layers
        self.max_heads = max_heads
        self.show = show

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:  # noqa: D401, N802 – HF callback signature
        model = kwargs.get("model")
        if model is None:
            return

        model_device = next(model.parameters()).device
        model.eval()

        for idx, sample in enumerate(self.example_texts):
            batch = {k: v.to(model_device) for k, v in sample.items()}
            with torch.no_grad():
                outputs = model(**batch, output_attentions=True)
            attentions = outputs.attentions
            pdf_name = f"step{state.global_step}_ex{idx}.pdf"
            save_path = self.output_dir / pdf_name
            plot_attention_grid(
                attentions,
                max_layers=self.max_layers,
                max_heads=self.max_heads,
                save_path=save_path,
                show=self.show,
            )