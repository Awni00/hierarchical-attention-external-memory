import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class HierarchicalAttention(tf.keras.layers.Layer):

    def __init__(self, key_dim, value_dim=None, symmetric_kernel=False,
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
        symmetric_kernel: boolean, optional
            whether to restrict query_dense and key_dense to be the same, by default False
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
        self.symmetric_kernel = symmetric_kernel
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

        if self.symmetric_kernel:
            self.key_dense = self.query_dense
        else:
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

        # retrieve memory vector for each position in input sequence
        retrieved_mems = tf.einsum('bti,btik->bik', mem_seq_attn_mat, per_seq_retrieved_mems)
        # shape: [batch_size, seq_len_batch, value_dim]

        return retrieved_mems

class MultiHeadHierarchicalAttention(tf.keras.layers.Layer):

    def __init__(self, key_dim, value_dim=None, n_heads=1, symmetric_kernel=False,
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
        symmetric_kernel: boolean, optional
            whether to restrict query_dense and key_dense to be the same, by default False
        n_heads : int, optional
            number of attention heads, by default 1
        attn_scale_factor_per_seq : float, optional
            the scale factor for the within-sequence attention, by default None
        attn_scale_factor_over_seqs : float, optional
            the scale factor for the over-sequences attention, by default None
        dense_kwargs : dict, optional
            kwargs for the query/key dense layers, by default None
        """

        super(MultiHeadHierarchicalAttention, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.symmetric_kernel = symmetric_kernel
        self.n_heads = n_heads
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
        _, self.input_seq_len, d_in = input_seq_shape
        _, self.mem_size, self.mem_seq_len, d_mx = input_mem_x_shape
        _, _, _, d_my = input_mem_y_shape

        self.query_maps = [layers.Dense(self.key_dim, **self.dense_kwargs) for _ in range(self.n_heads)]

        if self.symmetric_kernel:
            self.key_maps = self.query_maps
        else:
            self.key_maps = [layers.Dense(self.key_dim, **self.dense_kwargs) for _ in range(self.n_heads)]

        if self.value_dim is None:
            self.value_dim = d_my // self.n_heads
        self.value_maps = [layers.Dense(self.value_dim, **self.dense_kwargs) for _ in range(self.n_heads)]

        self.head_concatenator = tf.keras.layers.Reshape((self.input_seq_len, self.value_dim * self.n_heads))

    def call(self, inputs, return_attention_scores=False):
        input_seq, memory_x, memory_y = inputs

        queries = tf.stack([query_map(input_seq) for query_map in self.query_maps], axis=-1)
        keys = tf.stack([key_map(memory_x) for key_map in self.key_maps], axis=-1)
        values = tf.stack([value_map(memory_y) for value_map in self.value_maps], axis=-1)

        # compute full pairwise inner products between all objects in input sequence and memory sequences
        attn_mat = tf.einsum('bikh,btjkh->btijh', queries, keys) # shape [batch_size, mem_size, seq_len_batch, seq_len_mem, n_heads]
        # shape: [batch_size, mem_size, seq_len_batch, seq_len_mem]
        self.last_attn_mat = attn_mat

        # attend *within* each memory sequence independently (i.e., select relevant parts of each memory sequence)
        # softmax along memory sequence length axis
        per_seq_attn_mat = tf.nn.softmax(self.attn_scale_factor_per_seq * attn_mat, axis=-2) # softmax along seq_len_mem axis
        self.last_per_seq_attn_mat = per_seq_attn_mat

        # retrieved memory vector for each position in input sequence for each sequence in memory
        per_seq_retrieved_mems = tf.einsum('btijh,btjkh->btikh', per_seq_attn_mat, values)
        # [batch_size, mem_size, seq_len_batch, embedding_dim, n_heads]

        # attend *over* memory sequences (i.e., select relevant memory sequences)
        mem_seq_attn_mat = tf.reduce_max(attn_mat, axis=-2)
        # softmax along mem_size axis
        mem_seq_attn_mat = tf.nn.softmax(self.attn_scale_factor_over_seqs * mem_seq_attn_mat, axis=1) 
        # shape: [batch_size, mem_size, seq_len_batch]
        self.last_mem_seq_attn_mat = mem_seq_attn_mat

        # retrieve memory vector for each position in input sequence
        retrieved_mems = tf.einsum('btih,btikh->bikh', mem_seq_attn_mat, per_seq_retrieved_mems)
        # shape: [batch_size, input_seq_len, embedding_dim, n_heads]

        # concatenate heads
        retrieved_mems = self.head_concatenator(retrieved_mems)
        # shape: [batch_size, input_seq_len, embedding_dim * n_heads]

        if return_attention_scores:
            return retrieved_mems, mem_seq_attn_mat, per_seq_attn_mat, attn_mat
        else:
            return retrieved_mems
