# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# alphas = torch.logspace(-7, 0, 6)
# betas = torch.linspace(0., 3., 10)
# result = np.load("out/fig/alpha_beta.npy", allow_pickle=True)
# print(result.shape)
# result = np.exp(result)
# # plot a heatmap
# fig, ax = plt.subplots()
# # Use imshow and manually specify the color limits with vmin and vmax to make the central range clearer and ignore outliers.
# cax = ax.imshow(result, cmap='viridis', aspect='auto', 
#                 extent=[betas[0], betas[-1], alphas[-1], alphas[0]],
#                 vmin=70, vmax=73)  # Clipping outliers by adjusting color scale limits
# fig.colorbar(cax)

# # Adjusting the tick positions and labels using alphas' actual values for the y-axis to reflect a log scale visually.
# alphas_indices = np.arange(len(alphas)) # Indices of alphas to be used as tick positions
# ax.set_yticks(alphas_indices) # Set y-tick positions at indices to avoid squeezing

# # Label formatting
# ax.set_yticklabels([f"{alpha:.1e}" for alpha in alphas])

# plt.xlabel("Beta")
# plt.ylabel("Alpha (Log Scale)")
# plt.title("alpha_beta")

# plt.savefig("out/fig/alpha_beta.png")
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Assuming alphas and betas are defined as before and the result is loaded and processed
alphas = torch.linspace(0.85, 0.95, 6)
betas = torch.linspace(0., 0.1, 10)
result = np.load("out/fig/alpha_beta.npy", allow_pickle=True)
result = np.exp(result)

best_alpha, best_beta = np.unravel_index(result.argmin(), result.shape)
print(f"Best alpha: {alphas[best_alpha]}, Best beta: {betas[best_beta]}")
# As before, setting up the plot
fig, ax = plt.subplots()
cax = ax.imshow(result, cmap='viridis', aspect='auto', interpolation='nearest',
                extent=[betas[0].item(), betas[-1].item(), np.log10(alphas[-1].item()), np.log10(alphas[0].item())],
                vmin=68, vmax=73)

fig.colorbar(cax)

# Adjusting to display the axis correctly. Given the adjustment above, we now correctly set ticks
# We convert alphas into their log10 values for y-ticks
ticks_loc = ax.get_yticks().tolist()
ax.yaxis.set_major_locator(plt.FixedLocator(ticks_loc))
ax.set_yticklabels([f"{10**val:.1e}" for val in ticks_loc])

plt.xlabel("Beta")
plt.ylabel("Alpha (Log Scale)")
plt.title("alpha_beta")

plt.savefig("out/fig/alpha_beta.png")