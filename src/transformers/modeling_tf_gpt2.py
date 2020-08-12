from tensorflow.keras.layers import (Dropout, Embedding,
                                     Layer, LayerNormalization)


import logging

import numpy as np
import tensorflow as tf

from .configuration_gpt2 import GPT2Config
from .file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable
from .modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFConv1D,
    TFPreTrainedModel,
    TFSequenceSummary,
    TFSharedEmbeddings,
    cast_bool_to_primitive,
    get_initializer,
    keras_serializable,
    shape_list,
)
from .tokenization_utils import BatchEncoding

logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]

def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi)
                                * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

class TFAttention(Layer):
    def __init__(self, nx, n_ctx, config, scale = False, **kwargs):
        super().__init__(**kwargs)

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep
        # identical to TF implem]
        assert n_state % config.n_head == 0
        self.n_ctx = n_ctx
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = TFConv1D(
            n_state * 3, nx,
            initializer_range=config.initializer_range, name="c_attn")
        self.c_proj = TFConv1D(
            n_state, nx,
            initializer_range=config.initializer_range, name="c_proj")
        self.attn_dropout = Dropout(config.attn_pdrop)
        self.resid_dropout = Dropout(config.resid_pdrop)

    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """1's in the lower triangle, counting from the lower right
        corner. Same as tf.matrix_band_part(tf.ones([nd, ns]), -1,
        ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def _attn(self, q, k, v, training):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b = True)
        if self.scale:
            # scale attention_scores
            dk = tf.cast(shape_list(k)[-1], tf.float32)
            w = w / tf.math.sqrt(dk)

        # w has shape [batch, heads, dst_sequence, src_sequence],
        # where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = self.causal_attention_mask(nd, ns, dtype = w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])

        # Is it necessary to multiply w with b?
        w = w * b - 1e4 * (1 - b)

        w = tf.nn.softmax(w, axis = -1)
        w = self.attn_dropout(w, training = training)

        return tf.matmul(w, v)

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] \
            + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        # (batch, head, seq_length, head_features)
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, x, layer_past, training):

        x = self.c_attn(x)
        q, k, value = tf.split(x, 3, axis=2)
        q = self.split_heads(q)
        k = self.split_heads(k)
        value = self.split_heads(value)
        if layer_past is not None:
            past_k, past_value = tf.unstack(layer_past, axis = 0)
            k = tf.concat([past_k, k], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        present = tf.stack([k, value], axis = 0)

        a = self._attn(q, k, value, training)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training = training)

        return a, present


class TFMLP(Layer):
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.c_fc = TFConv1D(
            n_state, nx,
            initializer_range = config.initializer_range,
            name = "c_fc")
        self.c_proj = TFConv1D(
            nx, n_state,
            initializer_range = config.initializer_range,
            name = "c_proj")
        self.act = gelu
        self.dropout = Dropout(config.resid_pdrop)

    def call(self, x, training):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training = training)
        return h2


class TFBlock(Layer):
    def __init__(self, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.ln_1 = LayerNormalization(
            epsilon=config.layer_norm_epsilon,
            name="ln_1")
        self.attn = TFAttention(nx, n_ctx, config, scale, name = "attn")
        self.ln_2 = LayerNormalization(
            epsilon=config.layer_norm_epsilon,
            name="ln_2")
        self.mlp = TFMLP(4 * nx, config, name="mlp")

    def call(self, x, layer_past, training):
        '''
        0. Input
        1. LayerNormalization
        2. MultiheadAttention
        3. 0 + 2
        4. LayerNormalization
        5. MLP
        6. 3 + 5
        '''
        a = self.ln_1(x)

        a, present = self.attn(a, layer_past, training)
        x = x + a

        m = self.ln_2(x)
        m = self.mlp(m, training)
        x = x + m
        return x, present

class TFGPT2MainLayer(Layer):
    config_class = GPT2Config

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache

        self.num_hidden_layers = config.n_layer
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd

        initializer_range = config.initializer_range
        self.wte = TFSharedEmbeddings(
            config.vocab_size,
            config.hidden_size,
            initializer_range = initializer_range,
            name = "wte"
        )
        embeddings_initializer = get_initializer(initializer_range)
        self.wpe = Embedding(
            config.n_positions,
            config.n_embd,
            embeddings_initializer = embeddings_initializer,
            name = "wpe",
        )
        self.drop = Dropout(config.embd_pdrop)
        self.h = [TFBlock(config.n_ctx,
                          config,
                          scale = True, name = "h_._{}".format(i))
                  for i in range(config.n_layer)]
        self.ln_f = LayerNormalization(
            epsilon = config.layer_norm_epsilon,
            name = "ln_f")

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte.weight = value
        self.wte.vocab_size = self.wte.weight.shape[0]

    def call(
        self,
        inputs,
        past = None,
        position_ids = None,
        inputs_embeds = None,
        training = False):
        if isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            past = inputs.get("past", past)
            position_ids = inputs.get("position_ids", position_ids)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            assert len(inputs) <= 10, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and "
                "inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = shape_list(past[0][0])[-2]
        if position_ids is None:
            position_ids = tf.range(past_length,
                                    input_shape[-1] + past_length,
                                    dtype = tf.int32)[tf.newaxis, :]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        position_ids = tf.reshape(position_ids,
                                  [-1, shape_list(position_ids)[-1]])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids, mode = "embedding")
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states, training=training)

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        presents = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            hidden_states, present = block(hidden_states,
                                           layer_past,
                                           training)
            presents = presents + (present,)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = tf.reshape(hidden_states, output_shape)

        return hidden_states, presents

class TFGPT2PreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"


class TFGPT2Model(TFGPT2PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPT2MainLayer(config, name = "transformer")

    def call(self, inputs, **kwargs):
        return self.transformer(inputs, **kwargs)

class TFGPT2LMHeadModel(TFGPT2PreTrainedModel,
                        TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPT2MainLayer(config, name = "transformer")

    def get_output_embeddings(self):
        return self.transformer.wte

    def prepare_inputs_for_generation(self, inputs, past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            inputs = tf.expand_dims(inputs[:, -1], -1)

        return {"inputs": inputs,
                "past": past,
                "use_cache": kwargs["use_cache"]}

    def call(
        self,
        inputs,
        past = None,
        position_ids = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        training=False):
        use_cache = True
        if isinstance(inputs, (tuple, list)):
            labels = inputs[10] if len(inputs) > 10 else labels
            if len(inputs) > 10:
                inputs = inputs[:10]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        print('use cache here', use_cache)
        transformer_outputs = self.transformer(
            inputs,
            past = past,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            training = training,
        )

        hidden_states = transformer_outputs[0]

        logits = self.transformer.wte(hidden_states, mode = "linear")

        outputs = (logits,) + transformer_outputs[1:]
        if labels is not None:
            # shift labels to the left and cut last logit token
            logits = logits[:, :-1]
            labels = labels[:, 1:]
            loss = self.compute_loss(labels, logits)
            outputs = (loss,) + outputs

        return outputs
