"""An implementation of a decoder which uses hierarchical memory attention to retrieve information from memory."""

import tensorflow as tf
from tensorflow.keras import layers
from transformer_modules import DecoderLayer, FeedForward
from hierarchical_attention import MultiHeadHierarchicalAttention

class HierMemoryDecoder(tf.keras.layers.Layer):
    """Hierarchical Memory Decoder layer.

    At each decoder layer, the decoder performs self-attention,
    cross-attends to the context, then cross-attends to the memory,
    using hierarchical attention.
    """
    def __init__(self, num_layers, num_heads, dff, hier_attn_kwargs, layernorm_first=True, dropout_rate=0.1, name="hier_mem_decoder"):
        """
        create a HierMemoryDecoder layer.

        Parameters
        ----------
        num_layers : int
            number of decoder layers
        num_heads : int
            number of heads in self-attention and cross-attention to context.
        dff : int
            hidden dense layer size in feedforward network.
        hier_attn_kwargs : dict
            kwargs of hierarchical attention module.
        layernorm_first : bool, optional
            whether to apply layer norm first, by default True
        dropout_rate : float, optional
            dropout rate, by default 0.1
        name : str, optional
            name of layer, by default "hier_mem_decoder"
        """
        super(HierMemoryDecoder, self).__init__(name=name)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.hier_attn_kwargs = hier_attn_kwargs
        self.layernorm_first = layernorm_first
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        _, self.sequence_length, self.d_model = input_shape

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.dec_layers = [
            DecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                layernorm_first=self.layernorm_first,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]

        self.mem_dec_layers = [
            HierAttnDecoderLayer(
                d_model=self.d_model,
                dff=self.dff,
                hier_attn_kwargs=self.hier_attn_kwargs,
                layernorm_first=self.layernorm_first)
            for _ in range(self.num_layers)]

        self.last_attn_scores = None

    def call(self, x, context, memory):
        x = self.dropout(x)

        for i in range(self.num_layers):
            # causal self-attn + cross-attend to context + ffn
            x = self.dec_layers[i](x, context)
            # hierarchical attention to memory + ffn
            x = self.mem_dec_layers[i](x, memory)

        return x

class HierAttnDecoderLayer(tf.keras.layers.Layer):
    """
    Hierarchical Attention Decoder Layer.

    Hierarchical attention to memory, residual connection, then feedforward.

    This is combined with a standard decoder layer, which performs self-attention,
    and cross-attention to context.
    """

    def __init__(self, d_model, dff, hier_attn_kwargs, layernorm_first=True, **kwargs):
        super(HierAttnDecoderLayer, self).__init__(**kwargs)

        self.layernorm_first = layernorm_first
        self.hierarchical_attention = MultiHeadHierarchicalAttention(**hier_attn_kwargs)

        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()
        self.ffn = FeedForward(d_model, dff, layernorm_first=layernorm_first)

    def call(self, x, memory):
        # hierarchical memory attention then feedforward
        # note: no causal-self-attention or cross-attention to context
        # (this would be done in preceeding standard decoder layer)

        if self.layernorm_first:
            x_ = self.layernorm(x)
            retrieved_mem, *attn_scores = self.hierarchical_attention(
                    [x_, memory, memory], return_attention_scores=True)
            x = self.add([x, retrieved_mem])
        else:
            retrieved_mem, *attn_scores = self.hierarchical_attention(
                [x_, memory, memory], return_attention_scores=True)
            x = self.add([x, retrieved_mem])
            x = self.layernorm(x)

        self.last_attn_scores = attn_scores

        x = self.ffn(x)

        return x