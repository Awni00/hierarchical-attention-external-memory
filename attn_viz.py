# utilities for analyzing attention patterns in hierarchical attention model
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def plot_per_seq_attention(attn_, mem_seqs_x, train_seqs_x, trim=None, inch_per_token=0.5):

    # find end of input and trim
    train_seqs_x = list(train_seqs_x)
    end_in = train_seqs_x.index(trim) if trim is not None else -1
    train_seqs_x = train_seqs_x[:end_in]
    attn_ = attn_[:, :end_in, :]

    # create gridspec
    num_mem_seqs = attn_.shape[0]
    cmap = 'gray' if np.shape(attn_)[-1] == 1 else None
    # different memories have different length; scale axes accordingly
    width_ratios = [list(mem_seq).index(trim) if trim is not None else len(mem_seq) for mem_seq in mem_seqs_x] + [1]
    fig = plt.figure(figsize=(inch_per_token * sum(width_ratios), inch_per_token * len(train_seqs_x))) # 1-inch per token?
    gs = gridspec.GridSpec(1, num_mem_seqs+1, width_ratios=width_ratios) # color-bar 1-token length
    axs = [fig.add_subplot(gs[i]) for i in range(num_mem_seqs)] # create ax for each mem seq
    cax = fig.add_subplot(gs[-1]) # colorbar ax
    im = None

    for t, ax in enumerate(axs):
        attn_t = attn_[t] # attn for t-th mem seq
        # find length of t-th memory sequence and trim
        mem_seq_t = list(mem_seqs_x[t])
        mem_end = mem_seq_t.index(trim) if trim is not None else -1
        mem_seq_t = mem_seq_t[:mem_end]
        attn_t = attn_t[:, :mem_end]

        # plot attn
        im = ax.imshow(attn_t, cmap=cmap, vmin=0, vmax=1)
        # label axes
        ax.set_xticks(range(len(mem_seq_t)))
        ax.set_yticks(range(len(train_seqs_x)))
        ax.set_xticklabels(mem_seq_t, rotation=90);
        ax.set_yticklabels(train_seqs_x);

    fig.colorbar(im, cax=cax)
    axs[0].set_ylabel('input sequence')
    fig.supxlabel('memory sequences', y=0.)
    fig.suptitle('attention within each memory sequence')
    return fig


def plot_seq_attention(attn_, mem_seqs_x, train_seqs_x, trim=None, inch_per_token=0.5, **figkwargs):

    train_seqs_x = list(train_seqs_x)
    end_in = train_seqs_x.index(trim) if trim is not None else -1
    train_seqs_x = train_seqs_x[:end_in]
    attn_ = attn_[:, :end_in, :]

    fig, ax = plt.subplots(figsize=(inch_per_token*len(train_seqs_x), inch_per_token*len(mem_seqs_x)))
    cmap = 'gray' if attn_.shape[-1] == 1 else None
    im = ax.imshow(attn_, cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(mem_seqs_x)))
    ax.set_xticks(range(len(train_seqs_x)))
    yticklabels = []
    for t in range(np.shape(attn_)[0]):
        mem_seq_t = list(mem_seqs_x[t])
        mem_end = mem_seq_t.index(trim) if trim is not None else -1
        mem_seq_t = mem_seq_t[:mem_end]
        yticklabels.append(' '.join(mem_seq_t))
    ax.set_yticklabels(yticklabels);
    ax.set_xticklabels(train_seqs_x, rotation=90);

    ax.set_xlabel('input sequence')
    ax.set_ylabel('memory sequences')
    ax.set_title('attention over sequences')

    if cmap:
        fig.colorbar(im, ax=ax);
    return fig
