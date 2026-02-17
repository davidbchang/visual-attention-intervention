import os
from enum import StrEnum
from pathlib import Path
from dataclasses import asdict, dataclass

import torch
from transformers.cache_utils import Cache
from transformers.utils.generic import ModelOutput


MODELS_BASE_DIR = "modeling"
PRETRAINED_MODELS_BASE_DIR = "models"


class Method(StrEnum):
    STANDARD = "standard"
    ADAPT_VIS = "adaptvis"
    VEA = "vea"
    CLVS = "clvs"


@dataclass(kw_only=True)
class ExperimentConfig:
    method: Method = Method.STANDARD

    @property
    def experiment_name(self) -> str:
        return "_".join(f"{field_name.replace("_", "-")}-{value}" for field_name, value in asdict(self).items())


@dataclass(kw_only=True)
class AdaptVisConfig(ExperimentConfig):
    threshold: float
    sharpen_weight: float
    smoothen_weight: float
    method: Method = Method.ADAPT_VIS

    @property
    def smoothen_description(self) -> str:
        return f"smoothen-with-weight={self.smoothen_weight}".replace(".", "")

    @property
    def sharpen_description(self) -> str:
        return f"sharpen-with-weight={self.sharpen_weight}".replace(".", "")


@dataclass(kw_only=True)
class VEAConfig(ExperimentConfig):
    smooth_strength: float
    highlight_strength: float
    method: Method = Method.VEA


@dataclass(kw_only=True)
class CLVSConfig(ExperimentConfig):
    smoothing: float
    window_memory_size: float
    uncertainty_threshold: float
    method: Method = Method.CLVS


@dataclass
class ExperimentOutputPaths:
    root: Path
    dataset_type: str
    split: str
    experiment_name: str

    results_dir_name: str = "results"
    visualizations_dir_name: str = "visualizations"
    attention_plots_dir_name: str = "attention_plots"
    attention_weights_dir_name: str = "attention_weights"
    heatmaps_dir_name: str = "heatmaps"

    predictions_file_name: str = "predictions.json"

    def __post_init__(self):
        self.results_dir_path = self.root / self.results_dir_name
        self.dataset_type_path = self.results_dir_path / self.dataset_type
        self.split_dir_path = self.dataset_type_path / self.split
        self.experiment_dir_path = self.split_dir_path / self.experiment_name
        self.visualizations_dir_path = self.experiment_dir_path / self.visualizations_dir_name
        self.attention_plots_dir_path = self.visualizations_dir_path / self.attention_plots_dir_name
        self.attention_weights_dir_path = self.visualizations_dir_path / self.attention_weights_dir_name
        self.heatmaps_dir_path = self.visualizations_dir_path / self.heatmaps_dir_name


def get_experiment_output_paths(model_name: str, dataset_type: str, split: str, experiment_name: str) -> ExperimentOutputPaths:
    model_output_base_dir = Path(MODELS_BASE_DIR) / model_name
    model_output_paths = ExperimentOutputPaths(model_output_base_dir, dataset_type, split, experiment_name)

    os.makedirs(model_output_paths.experiment_dir_path, exist_ok=True)
    os.makedirs(model_output_paths.attention_weights_dir_path, exist_ok=True)
    os.makedirs(model_output_paths.attention_plots_dir_path, exist_ok=True)
    os.makedirs(model_output_paths.heatmaps_dir_path, exist_ok=True)

    return model_output_paths


@dataclass
class EagerAttentionOutput(ModelOutput):
    """
    Base class for eager attention outputs.

    Args:
        attention_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_attention_heads, head_dim)`):
            Attention output after multiplying the attention weights with the values.

        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, sequence_length, sequence_length)`):
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        image_attention_scores (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_image_tokens)`):
            Attention scores from the first generated token allocated to the image tokens.

        text_attention_scores (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_text_tokens)`):
            Attention scores from the first generated token allocated to the text tokens.

        image_attention_weights (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_image_tokens)`):
            Attention weights from the first generated token allocated to the image tokens.

        text_attention_weights (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_text_tokens)`):
            Attention weights from the first generated token allocated to the text tokens.
    """
    attention_output: torch.FloatTensor | None = None
    attention_weights: torch.FloatTensor | None = None
    image_attention_scores: torch.FloatTensor | None = None
    text_attention_scores: torch.FloatTensor | None = None
    image_attention_weights: torch.FloatTensor | None = None
    text_attention_weights: torch.FloatTensor | None = None


@dataclass
class TextAttentionOutput(ModelOutput):
    """
    Base class for Gemma2 attention outputs.

    Args:
        attention_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Attention output after multiplying the attention weights with the values and applying the output projection layer.

        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, sequence_length, sequence_length)`):
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        image_attention_scores (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_image_tokens)`):
            Attention scores from the first generated token allocated to the image tokens.

        text_attention_scores (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_text_tokens)`):
            Attention scores from the first generated token allocated to the text tokens.

        image_attention_weights (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_image_tokens)`):
            Attention weights from the first generated token allocated to the image tokens.

        text_attention_weights (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_text_tokens)`):
            Attention weights from the first generated token allocated to the text tokens.
    """
    attention_output: torch.FloatTensor | None = None
    attention_weights: torch.FloatTensor | None = None
    image_attention_scores: torch.FloatTensor | None = None
    text_attention_scores: torch.FloatTensor | None = None
    image_attention_weights: torch.FloatTensor | None = None
    text_attention_weights: torch.FloatTensor | None = None


@dataclass
class TextDecoderLayerOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Hidden-states of the model at the output of the layer

        self_attention_weights (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, sequence_length, sequence_length)`):
            Output self-attentions weights for the layer

        image_attention_scores (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_image_tokens)`):
            Attention scores from the first generated token allocated to the image tokens for the layer.

        text_attention_scores (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_text_tokens)`):
            Attention scores from the first generated token allocated to the text tokens for the layer.

        image_attention_weights (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_image_tokens)`):
            Attention weights from the first generated token allocated to the image tokens for the layer.

        text_attention_weights (`torch.FloatTensor` of shape `(batch_size, num_attention_heads, num_text_tokens)`):
            Attention weights from the first generated token allocated to the text tokens for the layer.
    """
    hidden_states: torch.FloatTensor | None = None
    self_attention_weights: torch.FloatTensor | None = None
    image_attention_scores: torch.FloatTensor | None = None
    text_attention_scores: torch.FloatTensor | None = None
    image_attention_weights: torch.FloatTensor | None = None
    text_attention_weights: torch.FloatTensor | None = None


@dataclass
class TextModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        image_attention_scores (`tuple(torch.FloatTensor)`:
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attention_heads, num_image_tokens)`.

            Attention scores from the first generated token allocated to the image tokens for each layer.

        text_attention_scores (`tuple(torch.FloatTensor)`:
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attention_heads, num_text_tokens)`.

            Attention scores from the first generated token allocated to the text tokens for each layer.

        image_attention_weights (`tuple(torch.FloatTensor)`:
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attention_heads, num_image_tokens)`.

            Attention weights from the first generated token allocated to the image tokens for each layer.

        text_attention_weights (`tuple(torch.FloatTensor)`:
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attention_heads, num_text_tokens)`.

            Attention weights from the first generated token allocated to the text tokens for each layer.
    """
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    image_attention_scores: tuple[torch.FloatTensor, ...] | None = None
    text_attention_scores: tuple[torch.FloatTensor, ...] | None = None
    image_attention_weights: tuple[torch.FloatTensor, ...] | None = None
    text_attention_weights: tuple[torch.FloatTensor, ...] | None = None
