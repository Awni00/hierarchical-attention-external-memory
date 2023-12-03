import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def create_vectorizers(source, target, save_path=None):
    command_vectorizer = tf.keras.layers.TextVectorization(split='whitespace', output_mode='int', standardize=None)
    action_vectorizer = tf.keras.layers.TextVectorization(split='whitespace', output_mode='int', standardize=None)

    command_vectorizer.adapt(source)
    action_vectorizer.adapt(target)

    if save_path is not None:
        save_vocab(command_vectorizer, save_path + '/commands')
        save_vocab(action_vectorizer, save_path + '/actions')

    return command_vectorizer, action_vectorizer

def load_vectorizers(path):
    command_vectorizer = tf.keras.layers.TextVectorization(split='whitespace', output_mode='int', standardize=None)
    action_vectorizer = tf.keras.layers.TextVectorization(split='whitespace', output_mode='int', standardize=None)

    command_vectorizer.load_assets(f'{path}/commands')
    action_vectorizer.load_assets(f'{path}/actions')

    return command_vectorizer, action_vectorizer

def invert_seq_vector(sequence, vectorizer, join=True):
    vocab = np.array(vectorizer.get_vocabulary())
    seq = vocab[sequence]
    if join:
        seq = list(seq)
        seq = ' '.join(seq)
    return seq

def save_vocab(vectorizer, path):
    if not os.path.exists(path):
        os.mkdir(path)
    vectorizer.save_assets(path)

def load_scan_ds(split):

    # load data
    data = tfds.load(f'scan/{split}', as_supervised=True)
    train_ds = data['train']
    test_ds = data['test']

    # add start and end tokens to target sequence
    def add_start_eos_token(text):
        return "<START> " + text + " <END>"

    train_ds = train_ds.map(lambda x, y: (x, add_start_eos_token(y)))
    test_ds = test_ds.map(lambda x, y: (x, add_start_eos_token(y)))

    # load vectorizers
    command_vectorizer, action_vectorizer = load_vectorizers(f'text_vectorizer_vocabs/{split}')

    # unravel dataset into numpy arrays
    source_train, target_train = unravel_ds(train_ds)
    source_test, target_test = unravel_ds(test_ds)

    # tokenize and create label
    tokenized_source_train = command_vectorizer(source_train)
    tokenized_target_train = action_vectorizer(target_train)
    tokenized_label_train = tokenized_target_train[:, 1:]
    tokenized_target_train = tokenized_target_train[:, :-1]

    tokenized_source_test = command_vectorizer(source_test)
    tokenized_target_test = action_vectorizer(target_test)
    tokenized_label_test = tokenized_target_test[:, 1:]
    tokenized_target_test = tokenized_target_test[:, :-1]

    # create tf dataset
    train_ds = tf.data.Dataset.from_tensor_slices(((tokenized_source_train, tokenized_target_train), tokenized_label_train))
    test_ds = tf.data.Dataset.from_tensor_slices(((tokenized_source_test, tokenized_target_test), tokenized_label_test))

    return train_ds, test_ds, command_vectorizer, action_vectorizer

def sample_memory_buffer(input_seqs, target_seqs, n_mems, n=None):
    '''given input, target seqs, sample n_mems for each training example'''

    if n is None:
        n = len(input_seqs)
        assert n == len(target_seqs)
    sample = np.array([np.random.choice(n, n_mems) for _ in range(n)])
    input_mem_seqs = tf.gather(input_seqs, sample, axis=0)
    target_mem_seqs = tf.gather(target_seqs, sample, axis=0)

    return input_mem_seqs, target_mem_seqs

def create_memory_ds(base_ds, n_mems, memory_bank=None):
    '''
    given input, target, label seqs in tf dataset, sample memory buffer
    for each training example and create new dataset including memories
    '''
    if memory_bank is None:
        memory_bank = base_ds

    input_seqs, target_seqs, label_seqs = unravel_ds(base_ds, format='s,t,l')

    mem_input_bank, mem_target_bank, _ = unravel_ds(memory_bank, format='s,t,l')
    input_mem_seqs, target_mem_seqs = sample_memory_buffer(mem_input_bank, mem_target_bank, n_mems=n_mems, n=len(input_seqs))
    mem_ds = tf.data.Dataset.from_tensor_slices((((input_seqs, target_seqs), (input_mem_seqs, target_mem_seqs)), label_seqs))
    return mem_ds

def unravel_ds(ds, format='s,t'):
    if format == 's,t':
        source = np.array([x.numpy() for x, y in ds])
        target = np.array([y.numpy() for x, y in ds])

        return source, target

    elif format =='s,t,l':
        source = np.array([x.numpy() for (x, y), z in ds])
        target = np.array([y.numpy() for (x, y), z in ds])
        label = np.array([z.numpy() for (x, y), z in ds])
        return source, target, label

def unravel_mem_ds(mem_ds):
    source = np.array([s for (((s, t), (sm, tm)), l) in mem_ds])
    target = np.array([t for (((s, t), (sm, tm)), l) in mem_ds])
    mem_source = np.array([sm for (((s, t), (sm, tm)), l) in mem_ds])
    mem_target = np.array([tm for (((s, t), (sm, tm)), l) in mem_ds])
    label = np.array([l for (((s, t), (sm, tm)), l) in mem_ds])
    return source, target, mem_source, mem_target, label