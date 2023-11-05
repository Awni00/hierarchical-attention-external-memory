"""Implements cross-attention mechanisms for transformers and Abstractors"""

import tensorflow as tf

from tensorflow.keras.layers import MultiHeadAttention

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self,
        use_resiudal=True,
        layernorm_first=True,
        **kwargs):

        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm_first = layernorm_first
        self.use_residual = use_resiudal

        self.layernorm = tf.keras.layers.LayerNormalization()
        if self.use_residual:
            self.add = tf.keras.layers.Add()

    def add_residual(self, x, attn_output):
        if self.use_residual:
            x = self.add([x, attn_output])
        else:
            x = attn_output
        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        if self.layernorm_first:
            x_ = self.layernorm(x)
            attn_output = self.mha(query=x_, value=x_, key=x_)
            x = self.add_residual(x, attn_output)
        else:
            attn_output = self.mha(query=x, value=x, key=x)
            x = self.add_residual(x, attn_output)
            x = self.layernorm(x)

        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        if self.layernorm_first:
            x_ = self.layernorm(x)
            attn_output = self.mha(query=x_, value=x_, key=x_, use_causal_mask=True)
            x = self.add_residual(x, attn_output)
        else:
            attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
            x = self.add_residual(x, attn_output)
            x = self.layernorm(x)

        return x


class CrossAttention(BaseAttention):
    def call(self, x, context):

        if self.layernorm_first:
            x_ = self.layernorm(x)
            attn_output, attn_scores = self.mha(
                    query=x_, key=context, value=context, return_attention_scores=True)
            x = self.add_residual(x, attn_output)
        else:
            attn_output, attn_scores = self.mha(
                query=x, key=context, value=context, return_attention_scores=True)
            x = self.add_residual(x, attn_output)
            x = self.layernorm(x)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        return x

class RelationalAttention(BaseAttention):
  def call(self, symbols, input_objects):
    attn_output, attn_scores = self.mha(
        query=input_objects,
        key=input_objects,
        value=symbols ,
        return_attention_scores=True)

    if self.layernorm_first:
        input_objects = self.layernorm(input_objects)
        attn_output, attn_scores = self.mha(
            query=input_objects, key=input_objects, value=symbols, return_attention_scores=True)
        x = self.add_residual(symbols, attn_output)
    else:
        attn_output, attn_scores = self.mha(
            query=input_objects, key=input_objects, value=symbols, return_attention_scores=True)
        x = self.add_residual(symbols, attn_output)
        symbols = self.layernorm(symbols)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    return symbols
