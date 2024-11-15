import os
import torch
import numpy as np
import matplotlib.pyplot as plt

sorted_attention = np.load("out/fig/attn/sorted_attention.npy", allow_pickle=True)

os.makedirs("out/fig/attn", exist_ok=True)

for layer, attention in enumerate(sorted_attention):
    fig, ax = plt.subplots(figsize=(20, 10))
    total_tokens = attention.shape[0]

    specific_tokens = [(total_tokens//4)-1 , (total_tokens//2)-1, (3*total_tokens//4)-1]
    specific_colors = ['red', 'green', 'blue']  # Colors for the specific lines

    for t, token_attention in enumerate(attention):
        token_attention = token_attention[:t + 1]
        color = plt.cm.cool(t / total_tokens)
        ax.plot(token_attention, color=color, linewidth=0.2)
    
    for t in specific_tokens:
        token_attention = attention[t, :t + 1]
        color_index = specific_tokens.index(t)
        ax.plot(token_attention, color=specific_colors[color_index], linewidth=1.5, label=f'Token {t+1}')  # t+1 for 1-based indexing in label

            
    # specifically, draw 512, 1024 
    smappable = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=0, vmax=total_tokens))
    cbar = fig.colorbar(smappable, ax=ax, label="Token Index")
    
    indices_to_show = [0, total_tokens//4, total_tokens//2, 3*total_tokens//4, total_tokens-1]
    cbar.set_ticks(indices_to_show)
    cbar.set_ticklabels([f"Token {i}" for i in indices_to_show])

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(linestyle="--", linewidth=0.5)
    plt.legend(loc='upper right', fontsize=17)
    ax.set_title(f"Layer_{layer}_Sorted_Attention", fontsize=20)
    
    # for desired_x_lim in [63, 127, 255, 511, total_tokens-1]:
    for desired_x_lim in [total_tokens-1]:
        os.makedirs(f"out/fig/attn/x_log_y_log/{desired_x_lim+1}", exist_ok=True)
        os.makedirs(f"out/fig/attn/x_linear_y_log/{desired_x_lim+1}", exist_ok=True)
        os.makedirs(f"out/fig/attn/x_linear_y_linear/{desired_x_lim+1}", exist_ok=True)
        ax.set_xscale("symlog")
        ax.set_yscale("log")
        ax.set_xlim(0, desired_x_lim)
        y_min = np.min([attention[i, :min(i+1, desired_x_lim)].min() for i in range(total_tokens)])
        ax.set_ylim(y_min, 1)
        plt.savefig(f"out/fig/attn/x_log_y_log/{desired_x_lim+1}/Layer_{layer}.png", bbox_inches="tight")
        ax.set_xscale("linear")
        plt.savefig(f"out/fig/attn/x_linear_y_log/{desired_x_lim+1}/Layer_{layer}.png", bbox_inches="tight")
        ax.set_yscale("linear")
        plt.savefig(f"out/fig/attn/x_linear_y_linear/{desired_x_lim+1}/Layer_{layer}.png", bbox_inches="tight")


plt.close()