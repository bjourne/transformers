# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "3.0.2"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging

# Configurations
from .configuration_auto import CONFIG_MAPPING, AutoConfig
from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from .configuration_utils import PretrainedConfig
from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
from .data import (
    DataProcessor,
    InputExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    SquadExample,
    SquadFeatures,
    SquadV1Processor,
    SquadV2Processor,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    is_sklearn_available,
    squad_convert_examples_to_features,
    xnli_output_modes,
    xnli_processors,
    xnli_tasks_num_labels,
)

# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_apex_available,
    is_psutil_available,
    is_py3nvml_available,
    is_tf_available,
    is_torch_available,
    is_torch_tpu_available,
)
from .hf_argparser import HfArgumentParser

# TF 2.0 <=> PyTorch conversion utilities
from .modeling_tf_pytorch_utils import (
    convert_tf_weight_name_to_pt_weight_name,
    load_pytorch_checkpoint_in_tf2_model,
    load_pytorch_model_in_tf2_model,
    load_pytorch_weights_in_tf2_model,
    load_tf2_checkpoint_in_pytorch_model,
    load_tf2_model_in_pytorch_model,
    load_tf2_weights_in_pytorch_model,
)

# Tokenizers
from .tokenization_gpt2 import GPT2Tokenizer, GPT2TokenizerFast
from .tokenization_longformer import LongformerTokenizer, LongformerTokenizerFast
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TensorType,
    TokenSpan,
)
from .tokenization_utils_fast import PreTrainedTokenizerFast

# Trainer
from .trainer_utils import EvalPrediction, set_seed
from .training_args import TrainingArguments
from .training_args_tf import TFTrainingArguments


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


if is_sklearn_available():
    from .data import glue_compute_metrics, xnli_compute_metrics


# Modeling
if is_torch_available():
    from .generation_utils import top_k_top_p_filtering
    from .modeling_utils import PreTrainedModel, prune_layer, Conv1D, apply_chunking_to_forward
    from .modeling_auto import (
        AutoModel,
        AutoModelForPreTraining,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelWithLMHead,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoModelForTokenClassification,
        AutoModelForMultipleChoice,
        MODEL_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_WITH_LM_HEAD_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    )

    from .modeling_gpt2 import (
        GPT2PreTrainedModel,
        GPT2Model,
        GPT2LMHeadModel,
        load_tf_weights_in_gpt2,
    )
    from .modeling_encoder_decoder import EncoderDecoderModel

    # Optimization
    from .optimization import (
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    # Trainer
    from .trainer import Trainer, set_seed, torch_distributed_zero_first, EvalPrediction
    from .data.data_collator import (
        default_data_collator,
        DataCollator,
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
    )
    from .data.datasets import (
        GlueDataset,
        TextDataset,
        LineByLineTextDataset,
        GlueDataTrainingArguments,
        SquadDataset,
        SquadDataTrainingArguments,
    )

# TensorFlow
if is_tf_available():
    from .generation_tf_utils import tf_top_k_top_p_filtering
    from .modeling_tf_utils import (
        shape_list,
        TFPreTrainedModel,
        TFSequenceSummary,
        TFSharedEmbeddings,
    )
    from .modeling_tf_gpt2 import (
        TFGPT2LMHeadModel,
        TFGPT2MainLayer,
        TFGPT2Model,
        TFGPT2PreTrainedModel,
    )

    # Optimization
    from .optimization_tf import (
        AdamWeightDecay,
        create_optimizer,
        GradientAccumulator,
        WarmUp,
    )

    # Trainer
    from .trainer_tf import TFTrainer
