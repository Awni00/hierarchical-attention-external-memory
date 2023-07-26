import tensorflow as tf

class SimpleHierarchicalAttentionGate(tf.keras.layers.Layer):
    def __init__(self, componentwise=True, mem_dim=None, **kwargs):
        super(SimpleHierarchicalAttentionGate, self).__init__(**kwargs)
        self.componentwise = componentwise
        self.mem_dim = mem_dim
        if self.componentwise and self.mem_dim is None:
            raise ValueError('If componentwise, mem_dim must be specified.')
    
    def build(self, input_shape):
        if self.componentwise:
            self.gate_dense = tf.keras.layers.Dense(self.mem_dim, activation='sigmoid')
        else:
            self.gate_dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        mem_seq_attn_mat = inputs
        n_m = tf.shape(mem_seq_attn_mat)[1]
        unif = tf.ones_like(mem_seq_attn_mat) / tf.cast(n_m, tf.float32)
        tv_dist = tf.norm(mem_seq_attn_mat - unif, axis=1, ord=1)
        gate = self.gate_dense(tv_dist)
        return gate

