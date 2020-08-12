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
        self.pruned_heads = set()

    def prune_heads(self, heads):
        pass

    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """1's in the lower triangle, counting from the lower right
        corner. Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd),
        but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def _attn(self, q, k, v, attention_mask, training):
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

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = tf.nn.softmax(w, axis=-1)
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

    def call(self, x, layer_past, attention_mask, use_cache, training):

        x = self.c_attn(x)
        q, k, value = tf.split(x, 3, axis=2)
        q = self.split_heads(q)
        k = self.split_heads(k)
        value = self.split_heads(value)
        if layer_past is not None:
            past_k, past_value = tf.unstack(layer_past, axis = 0)
            k = tf.concat([past_k, k], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        # to cope with keras serialization
        if cast_bool_to_primitive(use_cache, True) is True:
            present = tf.stack([k, value], axis = 0)
        else:
            present = (None,)

        a = self._attn(q, k, value, attention_mask, training)

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

    def call(self, x, training = False):
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

    def call(self, x, layer_past,
             attention_mask, head_mask, use_cache,
             training):
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

        a, present = self.attn(a, layer_past, attention_mask,
                                use_cache, training)
        x = x + a

        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m

        return [x, present]

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
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        inputs_embeds = None,
        use_cache = None,
        # output_attentions = None,
        # output_hidden_states = None,
        training = False):
        if isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            past = inputs.get("past", past)
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            use_cache = inputs.get("use_cache", use_cache)
            # output_attentions = inputs.get("output_attentions", output_attentions)
            # output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            assert len(inputs) <= 10, "Too many inputs."
        else:
            input_ids = inputs

        # if output_attentions is None:
        #     output_attentions = self.output_attentions

        # if output_hidden_states is None:
        #     output_hidden_states = self.output_hidden_states

        if use_cache is None:
            use_cache = self.use_cache

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

        # print('output', output_attentions)

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = shape_list(past[0][0])[-2]
        if position_ids is None:
            position_ids = tf.range(past_length, input_shape[-1] + past_length, dtype=tf.int32)[tf.newaxis, :]

        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads,
            # from_seq_length, to_seq_length] this attention mask is
            # more simple than the triangular masking of causal
            # attention used in OpenAI GPT, we just need to prepare
            # the broadcast dimension here.
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

            # Since attention_mask is 1.0 for positions we want to
            # attend and 0.0 for masked positions, this operation will
            # create a tensor which is 0.0 for positions we want to
            # attend and -10000.0 for masked positions.  Since we are
            # adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.

            attention_mask = tf.cast(attention_mask, tf.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        print('use cache', use_cache)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # if head_mask is not None:
        #     raise NotImplementedError
        # else:
        head_mask = [None] * self.num_hidden_layers
        # head_mask = tf.constant([0] * self.num_hidden_layers)

        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids, mode = "embedding")
        position_embeds = self.wpe(position_ids)

        token_type_embeds = 0
        if token_type_ids is not None:
            shape = [-1, shape_list(token_type_ids)[-1]]
            token_type_ids = tf.reshape(token_type_ids, shape)
            token_type_embeds = self.wte(token_type_ids, mode="embedding")

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=training)

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        presents = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            outputs = block(hidden_states,
                            layer_past,
                            attention_mask,
                            head_mask[i],
                            use_cache,
                            training)

            hidden_states, present = outputs[:2]
            presents = presents + (present,)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = tf.reshape(hidden_states, output_shape)
        outputs = (hidden_states,)

        if use_cache is True:
            outputs = outputs + (presents,)
        return outputs


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
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        training=False):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.

    Return:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        prediction_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
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
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            # output_attentions = output_attentions,
            # output_hidden_states = output_hidden_states,
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
