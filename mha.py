import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Attention(tf.keras.layers.Layer):
    def __init__(self, key_dim, value_dim, beta=None, symmetric_attention=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.beta = beta
        self.symmetric_attention = symmetric_attention

    def build(self, input_shape):
        _, seq_len, dim = input_shape

        if self.beta is None:
            self.beta = 1 / np.sqrt(dim)

        self.query_transform = layers.Dense(self.key_dim)
        if self.symmetric_attention:
            self.key_transform = self.query_transform
        else:
            self.key_transform = layers.Dense(self.key_dim)
        self.value_transform = layers.Dense(self.value_dim)

        self.softmax = layers.Softmax(axis=-1)

    def _compute_causal_mask(self, query, value=None):
        q_seq_length = tf.shape(query)[1]
        v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
        causal_mask = tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0
        )
        return causal_mask


    def call(self, query, key, value, use_causal_mask=False, return_attention_scores=False):
        if use_causal_mask:
            mask = self._compute_causal_mask(query, value)
        else:
            mask=None

        query = self.query_transform(query)
        key = self.key_transform(key)
        value = self.value_transform(value)
        A = tf.matmul(query, key, transpose_b=True)
        A = self.softmax(self.beta * A, mask=mask)
        attention_seq = tf.einsum('bij,bjk->bik', A, value)
        if return_attention_scores:
            return attention_seq, A
        else:
            return attention_seq

class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, key_dim, num_heads, value_dim=None, beta=None, symmetric_attention=False, **kwargs):
        super(CustomMultiHeadAttention, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.beta = beta
        self.num_heads = num_heads
        self.symmetric_attention = symmetric_attention

    def build(self, input_shape):
        _, seq_len, dim = input_shape

        if self.beta is None:
            self.beta = 1 / np.sqrt(dim)

        if self.key_dim is None:
            self.key_dim = dim // self.num_heads
            if dim % self.num_heads != 0:
                print(f'WARNING dim {dim} is not a multiple of n_heads {self.num_heads}')

        if self.value_dim is None:
            self.value_dim = dim // self.num_heads
            if dim % self.num_heads != 0:
                print(f'WARNING dim {dim} is not a multiple of n_heads {self.num_heads}')

        self.attention_heads = [
            Attention(key_dim=self.key_dim, value_dim=self.value_dim, beta=self.beta, symmetric_attention=self.symmetric_attention)
            for _ in range(self.num_heads)]

    def call(self, query, key, value, use_causal_mask=False, return_attention_scores=False):

        attention_head_results = [
            attention_head(query, key, value, use_causal_mask=use_causal_mask, return_attention_scores=return_attention_scores)
            for attention_head in self.attention_heads]

        if return_attention_scores:
            attention_results, attention_scores = zip(*attention_head_results)
            return tf.concat(attention_results, axis=-1), tf.stack(attention_scores, axis=-1)
        else:
            return tf.concat(attention_head_results, axis=-1)