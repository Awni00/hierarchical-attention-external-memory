import numpy as np
from tqdm import trange

def autoregressive_predict(model, source, target_length, start_token):
    n = len(source)

    output = np.zeros(shape=(n, target_length+1), dtype=int)
    output[:,0] = start_token

    for i in range(target_length):
        predictions = model((source, output[:, :-1]), training=False)
        predictions = predictions[:, i, :]
        predicted_id = np.argmax(predictions, axis=-1)
        output[:,i+1] = predicted_id

    return output[:, 1:]

def autoregressive_predict_batch(model, source, target_length, start_token, batch_size):
    n = len(source)

    outputs = []

    for i in trange(0, n, batch_size):
        output = autoregressive_predict(model, source[i:i+batch_size], target_length, start_token)
        outputs.append(output)

    return np.concatenate(outputs, axis=0)

def mem_autoregressive_predict(model, source, mem_in, mem_out, target_length, start_token):
    n = len(source)

    output = np.zeros(shape=(n, target_length+1), dtype=int)
    output[:,0] = start_token

    for i in range(target_length):
        predictions = model(((source, output[:, :-1]), (mem_in, mem_out)), training=False)
        predictions = predictions[:, i, :]
        predicted_id = np.argmax(predictions, axis=-1)
        output[:,i+1] = predicted_id

    return output[:, 1:]

def mem_autoregressive_predict_batch(model, source, mem_in, mem_out, target_length, start_token, batch_size):
    n = len(source)

    outputs = []

    for i in trange(0, n, batch_size, leave=False):
        output = mem_autoregressive_predict(
            model, source[i:i+batch_size], mem_in[i:i+batch_size], mem_out[i:i+batch_size],
            target_length, start_token)
        outputs.append(output)

    return np.concatenate(outputs, axis=0)