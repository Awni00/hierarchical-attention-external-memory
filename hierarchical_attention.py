import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class HierarchicalAttention(tf.keras.layers.Layer):

    def __init__(self, key_dim, value_dim=None,
        attn_scale_factor_per_seq=None, attn_scale_factor_over_seqs=None,
        dense_kwargs=None, **kwargs):
        """create HierarchicalAttention layer.

        HierarchicalAttention attention retrieves information from a hierarchical memory structure.
        The hierarchical memory contains several independent pairs of input-output sequences.
        The hierarchical attention mechanism attends between the input sequence and each memory sequence.
        This is the first level of the "hierarchy". Then, the hierarchical attention mechanism attends
        between the input sequence and the memory *sequences*. This is the second level of the "hierarchy".
        The two levels of attention are combined to retrieve a single memory vector for each position in
        the input sequence.

        Parameters
        ----------
        key_dim : int
            dimension of keys and queries.
        value_dim : int, optional
            dimension of value projection if different from embedding dimension, by default None
        attn_scale_factor_per_seq : float, optional
            the scale factor for the within-sequence attention, by default None
        attn_scale_factor_over_seqs : float, optional
            the scale factor for the over-sequences attention, by default None
        dense_kwargs : dict, optional
            kwargs for the query/key dense layers, by default None
        """

        super(HierarchicalAttention, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.attn_scale_factor_per_seq = attn_scale_factor_per_seq
        if self.attn_scale_factor_per_seq is None:
            self.attn_scale_factor_per_seq = 1 / np.sqrt(self.key_dim)
        self.attn_scale_factor_over_seqs = attn_scale_factor_over_seqs
        if self.attn_scale_factor_over_seqs is None:
            self.attn_scale_factor_over_seqs = 1 / np.sqrt(self.key_dim)
        self.dense_kwargs = dense_kwargs
        if self.dense_kwargs is None:
            self.dense_kwargs = {}

    def build(self, input_shape):

        input_seq_shape, input_mem_x_shape, input_mem_y_shape = input_shape

        self.query_dense = layers.Dense(self.key_dim, **self.dense_kwargs)
        self.key_dense = layers.Dense(self.key_dim, **self.dense_kwargs)
        if self.value_dim is None:
            self.value_dim = input_mem_y_shape[-1]
        self.value_dense = layers.Dense(self.value_dim, **self.dense_kwargs)

    def call(self, inputs):
        input_seq, memory_x, memory_y = inputs

        queries = self.query_dense(input_seq)
        keys = self.key_dense(memory_x)
        values = self.value_dense(memory_y)

        # compute full pairwise inner products between all objects in input sequence and memory sequences
        attn_mat = tf.einsum('bik,btjk->btij', queries, keys)
        # shape: [batch_size, mem_size, seq_len_batch, seq_len_mem]
        self.last_attn_mat = attn_mat

        # attend *within* each memory sequence independently (i.e., select relevant parts of each memory sequence)
        # softmax along memory sequence length axis
        per_seq_attn_mat = tf.nn.softmax(self.attn_scale_factor_per_seq * attn_mat, axis=-1)
        self.last_per_seq_attn_mat = per_seq_attn_mat

        # retrieved memory vector for each position in input sequence for each sequence in memory
        per_seq_retrieved_mems = tf.einsum('btij,btjk->btik', per_seq_attn_mat, values)
        # shape: [batch_size, mem_size, seq_len_batch, value_dim]

        # attend *over* memory sequences (i.e., select relevant memory sequences)
        mem_seq_attn_mat = tf.reduce_max(attn_mat, axis=-1)
        # softmax along memory sequences (mem_size) axis
        mem_seq_attn_mat = tf.nn.softmax(self.attn_scale_factor_over_seqs * mem_seq_attn_mat, axis=1)
        # shape: [batch_size, mem_size, seq_len_batch]
        self.last_mem_seq_attn_mat = mem_seq_attn_mat
        # TODO: add option for scaling constant

        # retrieve memory vector for each position in input sequence
        retrieved_mems = tf.einsum('bti,btik->bik', mem_seq_attn_mat, per_seq_retrieved_mems)
        # shape: [batch_size, seq_len_batch, value_dim]

        return retrieved_mems