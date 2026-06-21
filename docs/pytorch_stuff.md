# PyTorch stuff — running reference

A log of PyTorch functions and classes used in this codebase, in the order
encountered. Each entry: what it does, the math behind it, where we used it,
and a placeholder for a fuller explanation to come back to later.

Notation follows the theory thread convention throughout:
- $\mathbf{x}, \mathbf{z}$ — boldface lowercase: column vectors
- $X, Z, W$ — capital non-boldface: matrices
- $\mathbf{X}, \mathbf{Z}$ — boldface capital: random vectors
- $D_{\mathbf{x}}f$ — derivative matrix, shape $m \times n$ for $f: \mathbb{R}^n \to \mathbb{R}^m$
- $\nabla_{\mathbf{x}}f = (D_{\mathbf{x}}f)^\top$ — gradient, transpose of the derivative matrix

---

## `tensor.unfold(dimension, size, step)`

**What it does:** slides a window of length `size` along `dimension`,
stepping by `step` each time, and appends a new trailing dimension
containing the window contents.

**The math:** for a 1D tensor with index $i \in \{0,\dots,L-1\}$, calling
`unfold(0, P, P)` rewrites each index as $i = kP + p$ (the division
algorithm, $k = \lfloor i/P \rfloor$, $p = i \bmod P$), producing a
$(L/P, P)$ tensor where entry $[k, p]$ equals the original entry $[kP+p]$.
For a 2D spatial image, two calls are needed — one along $H$, one along $W$
— since each call decomposes only one axis at a time.

**Where used:** considered for `patchify()` but replaced by the cleaner
`reshape` + `permute` approach (see below). Kept here for reference.

**Full explanation:** deferred.

---

## `tensor.reshape(*shape)` and `tensor.permute(*dims)`

**What reshape does:** returns a view of the tensor with a new shape,
reinterpreting the flat memory layout without moving data. Correct when
splitting one axis into two nested axes (e.g. $H \to (H_p, P)$ where
$H = H_p \cdot P$). Incorrect for regrouping non-adjacent memory
locations — this is why a naive `reshape` on a raw image gives rows,
not spatial patches.

**What permute does:** reorders axes without moving any data in memory —
produces a new view with axes in the specified order. No computation, pure
relabeling.

**The math for patchify:** starting from $X \in \mathbb{R}^{B \times C \times H \times W}$:

$$X \xrightarrow{\text{reshape}} \mathbb{R}^{B \times C \times H_p \times P \times W_p \times P}
\xrightarrow{\text{permute}(0,2,4,1,3,5)} \mathbb{R}^{B \times H_p \times W_p \times C \times P \times P}
\xrightarrow{\text{reshape}} \mathbb{R}^{B \times H_p W_p \times CP^2}$$

For STL-10 with $P=8$, $C=3$, $H=W=96$: $(B,3,96,96) \to (B,144,192)$.

**The math for split/merge heads:** for $Q \in \mathbb{R}^{B \times L \times d}$
with $h$ heads and $d_h = d/h$:

$$Q \xrightarrow{\text{reshape}} \mathbb{R}^{B \times L \times h \times d_h}
\xrightarrow{\text{permute}(0,2,1,3)} \mathbb{R}^{B \times h \times L \times d_h}$$

Each slice $Q[:,i,:,:]$ is head $i$'s query matrix. The inverse
(`_merge_heads`) applies `permute(0,2,1,3)` then `reshape` back to
$(B, L, d)$.

**Where used:**
- `patchify()` in `patch_embedding.py`
- `_split_heads()` and `_merge_heads()` in `attention.py`

**Full explanation:** deferred.

---

## `torch.nn.Linear(in_features, out_features, bias=True)`

**What it does:** applies the affine map $\mathbf{y} = W\mathbf{x} + \mathbf{b}$
to a single vector, or equivalently $Y = XW^\top + \mathbf{b}$ when $X$
is a batch of row vectors (shape $B \times d_\text{in}$).

**Critical convention:** PyTorch stores the weight as
$W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ (output rows $\times$
input columns). The $W^\top$ in $Y = XW^\top$ is baked into the
implementation — you never need to transpose manually. So
`nn.Linear(d_in, d_out)` applied to $X \in \mathbb{R}^{B \times d_\text{in}}$
gives $Y \in \mathbb{R}^{B \times d_\text{out}}$, matching the row-stacked
convention $Y = XW_Q$ used throughout the theory thread.

**Stored weight shape vs. math convention:**

| Math notation | Meaning | `weight.shape` in PyTorch |
|---|---|---|
| $W_Q \in \mathbb{R}^{d \times d_k}$ (row-stacked: $Q = XW_Q$) | batched form | $(d_k, d)$ — transposed |
| $\hat{W}_Q \in \mathbb{R}^{d_k \times d}$ (column-vector: $\mathbf{q} = \hat{W}_Q \mathbf{x}$) | single-vector form | $(d_k, d)$ — same |

Both refer to the same linear map. PyTorch stores the weight in the
single-vector form shape.

**Where used:**
- `PatchEmbedding`: $E = XW_e + \mathbf{b}_e$, projects $\mathbb{R}^{192} \to \mathbb{R}^d$
- `MultiHeadAttention`: $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$, no bias
- `MLP`: $\text{fc1}: d \to 4d$, $\text{fc2}: 4d \to d$
- `Projector`: three layers, $192 \to 512$

**Full explanation:** deferred.

---

## `torch.nn.Parameter(tensor)`

**What it does:** wraps a tensor and registers it as a learnable
parameter with PyTorch's autograd engine and the optimizer. Without this
wrapper, a tensor stored as a module attribute is invisible to the
optimizer — no gradient is computed, and the tensor is never updated.

**The math:** the positional embedding table $P \in \mathbb{R}^{L \times d}$
(one row per patch position) is a learned parameter with no input. It is
added to the patch embeddings, broadcast over the batch dimension:

$$X_\text{pos} = X_\text{embed} + P \in \mathbb{R}^{B \times L \times d}$$

Gradient descent updates $P$ directly via $\nabla_P \mathcal{L}_\text{VICReg}$.
The $\times 0.02$ initialization keeps the positional signal small
relative to patch content at the start of training.

**Where used:** `PositionalEmbedding`:
```python
self.pos_embed = nn.Parameter(torch.randn(num_patches, embed_dim) * 0.02)
```

**Full explanation:** deferred.

---

## `torch.nn.ModuleList(modules)`

**What it does:** a list-like container for `nn.Module` objects that
properly registers their parameters with PyTorch. A plain Python `list`
of modules does NOT register parameters — the optimizer never sees them,
no gradients flow, weights are never updated. This is a silent failure
with no error message.

**Where used:** `ViTEncoder` stores $N$ stacked `TransformerBlock`s:

```python
self.blocks = nn.ModuleList([
    TransformerBlock(embed_dim, num_heads, mlp_ratio)
    for _ in range(depth)
])
```

Forward pass iterates: `for block in self.blocks: x = block(x)`.

**Full explanation:** deferred.

---

## `torch.softmax(input, dim)`

**What it does:** applies the softmax function along `dim`, converting
raw scores into a probability distribution summing to 1 along that axis.

**The math:** for a score vector $\mathbf{s} \in \mathbb{R}^L$:

$$\text{softmax}(\mathbf{s})_j = \frac{e^{s_j}}{\sum_{k=1}^{L} e^{s_k}}$$

In attention, the score matrix $S \in \mathbb{R}^{B \times h \times L \times L}$
is produced by:

$$S = \frac{QK^\top}{\sqrt{d_k}}, \qquad Q, K \in \mathbb{R}^{B \times h \times L \times d_k}$$

Softmax is applied along `dim=-1`, normalizing each **row** of $S$:
row $i$ becomes patch $i$'s attention distribution over all $L$ keys.
The output $A = \text{softmax}(S)$ satisfies $\sum_j A_{ij} = 1$ for
every query $i$, making $AV$ a proper weighted average of value vectors:

$$\text{Attention}(X) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Critical detail — wrong axis means wrong semantics:**

| `dim` | What sums to 1 | Meaning |
|---|---|---|
| `dim=-1` (correct) | each row | each query's distribution over all keys |
| `dim=-2` (wrong) | each column | each key's distribution over all queries |

**Where used:** `SingleHeadAttention` and `MultiHeadAttention` forward passes.

**Full explanation:** deferred.

---

## `torch.nn.LayerNorm(normalized_shape)`

**What it does:** normalizes the input along the last `normalized_shape`
dimensions, then applies a learned affine transform.

**The math:** for a single token vector $\mathbf{x} \in \mathbb{R}^d$:

$$\text{LayerNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot
\frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol{\beta}$$

where:

$$\mu = \frac{1}{d}\sum_{j=1}^d x_j, \qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2$$

and $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ are
learned scale and shift. Statistics are computed per token, per sample
— no dependence on the batch dimension.

**Why not BatchNorm:** BatchNorm normalizes across the batch dimension,
computing statistics over all $B$ samples at a given position. This
couples unrelated images and makes statistics depend on batch size.
LayerNorm normalizes each token independently — correct for sequence
models.

**Pre-norm convention:** LayerNorm is applied *before* each sublayer:

$$\mathbf{x}' = \mathbf{x} + \text{MHA}(\text{LN}_1(\mathbf{x}))$$
$$\mathbf{y} = \mathbf{x}' + \text{MLP}(\text{LN}_2(\mathbf{x}'))$$

Pre-norm (LN before sublayer) is more stable than post-norm (LN after)
for deep networks. Modern standard since ~2020.

**Where used:** `TransformerBlock` (LN$_1$ before attention, LN$_2$
before MLP), `ViTEncoder` (one final LN before mean pooling).

**Full explanation:** deferred.

---

## `torch.nn.GELU()`

**What it does:** applies the Gaussian Error Linear Unit activation —
a smooth, differentiable approximation to ReLU.

**The math:**

$$\text{GELU}(x) = x \cdot \Phi(x), \qquad
\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

Asymptotic behavior:
- $x \to +\infty$: $\Phi(x) \to 1 \Rightarrow \text{GELU}(x) \approx x$ (like ReLU)
- $x \to -\infty$: $\Phi(x) \to 0 \Rightarrow \text{GELU}(x) \approx 0$ (like ReLU)
- $x = 0$: $\text{GELU}(0) = 0$, $\text{GELU}'(0) = 0.5$ (smooth, unlike ReLU's kink at 0)

**Why not ReLU:** the hard zero at $x=0$ in ReLU ($\text{ReLU}'(0)$
undefined) creates a non-smooth gradient landscape less favorable for
deep transformers. GELU's smooth transition helps optimization.
Empirical finding, not a theorem.

**Where used:** `MLP` in `transformer_block.py`, between `fc1` and `fc2`.

**Full explanation:** deferred.

---

## `torch.nn.BatchNorm1d(num_features)`

**What it does:** normalizes a batch of 1D feature vectors across the
batch dimension, then applies a learned affine transform.

**The math:** for a batch $Z \in \mathbb{R}^{B \times d}$, normalizing
column $j$ across the batch:

$$\mu_j = \frac{1}{B}\sum_{i=1}^B z_{ij}, \qquad
\sigma_j^2 = \frac{1}{B}\sum_{i=1}^B (z_{ij} - \mu_j)^2$$

$$\text{BN}(z_{ij}) = \gamma_j \cdot \frac{z_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} + \beta_j$$

where $\gamma_j, \beta_j$ are learned per-feature scale and shift.

**Why BatchNorm in the projector (not LayerNorm):** the projector sees
one vector per image (shape $(B, d)$, not a sequence $(B, L, d)$), so
batch statistics are well-defined and stable. LayerNorm would normalize
over the $d$ feature dimensions of a single vector — less meaningful for
a fully-connected layer. The original VICReg paper used BatchNorm
specifically.

**`bias=False` on linear layers before BatchNorm:** a bias term before
BatchNorm is immediately cancelled out by the mean subtraction step
($z_{ij} - \mu_j$ absorbs any constant offset). It wastes parameters.
Our projector uses `bias=False` on the first two linear layers and
`bias=True` on the final layer (which has no BatchNorm after it).

**Important constraint:** BatchNorm1d requires $B \geq 2$ in train mode
— $B=1$ raises a `ValueError` since $\sigma_j^2$ is undefined for a
single sample. This is why `VICRegConfig` enforces `batch_size >= 2`.

**Train vs eval mode:**
- Train mode: uses current batch statistics $(\mu_j, \sigma_j^2)$ and
  accumulates running estimates via exponential moving average.
- Eval mode: uses accumulated running estimates, not batch statistics.

Output differs between modes for the same input — expected and correct.

**Where used:** `Projector` in `projector.py` (two `BatchNorm1d` layers,
after layers 1 and 2).

**Full explanation:** deferred.

---

## `torch.utils.data.Dataset` and `DataLoader`

**`Dataset`:** abstract base class. Subclass it and implement:
- `__len__()` — number of samples
- `__getitem__(idx)` — sample at index `idx`

**`DataLoader`:** wraps a `Dataset` and handles batching, shuffling, and
parallel data loading. Each iteration yields one batch.

**Two-view output contract:** our `STL10Unlabeled.__getitem__` returns
$(x', x'')$ — two independently augmented views of the same image.
The `DataLoader` collates these into batched tensors:

$$(X', X'') \in \mathbb{R}^{B \times 3 \times 96 \times 96} \times \mathbb{R}^{B \times 3 \times 96 \times 96}$$

The VICReg training loop then computes:

$$\mathbf{z}_i' = g_\phi(f_\theta(X'_i)), \quad
\mathbf{z}_i'' = g_\phi(f_\theta(X''_i)), \quad
\mathcal{L} = \mathcal{L}_\text{VICReg}(Z', Z'')$$

where $f_\theta$ is the encoder and $g_\phi$ is the projector.

**Windows note:** `num_workers > 0` can cause issues with CUDA
multiprocessing on Windows. Default: `num_workers=0`.

**Where used:** `data/stl10.py` (`STL10Unlabeled`), `vic_reg_loss/train.py`.

**Full explanation:** deferred.

---

## Profiling — GPU memory and timing

**Four things compete for GPU memory during training:**

| Component | Size | Notes |
|---|---|---|
| Parameters $\boldsymbol{\theta}$ | $\vert\boldsymbol{\theta}\vert \times 4$ bytes | fixed once model is built |
| Gradients $\nabla_{\boldsymbol{\theta}}\mathcal{L}$ | same as params | computed during backward |
| Adam state $(m_t, v_t)$ | $2 \times \vert\boldsymbol{\theta}\vert \times 4$ bytes | two buffers per parameter |
| Activations | varies with $B, L, N$ | **dominant term at training time** |

For a small ViT, the first three are tens of MiB — irrelevant against
8 GB. Activations dominate: every intermediate tensor must be kept from
the forward pass until the backward pass consumes it for the chain rule.

### `torch.cuda.memory_allocated()`

Returns bytes *currently* allocated on the GPU at the moment of the
call. Useful for point-in-time snapshots (e.g. right after the forward
pass, before backward).

### `torch.cuda.max_memory_allocated()`

Returns the peak allocation since the last reset — the high-water mark.
This is what determines OOM errors: the peak (during backward, when
activations + gradients coexist) is higher than the steady-state usage.

### `torch.cuda.reset_peak_memory_stats()`

Resets the peak counter to zero. Must be called before each run in a
sweep so `max_memory_allocated()` reports the peak for *that* run only.

### `torch.cuda.synchronize()`

Blocks the CPU until all queued GPU operations finish. Required for
accurate timing: GPU ops are dispatched asynchronously, so `time.time()`
without `synchronize()` measures queue time, not execution time.

**Where used:** profiling script (planned).

---

## Back-of-envelope (BOTE) GPU memory calculation

**Setup:** STL-10, $L = 144$ (fixed: $(96/8)^2 = 144$ patches per image),
float32 (4 bytes/value), Adam optimizer. Variables: $B$ = batch size,
$d$ = embedding dimension, $h$ = heads, $N$ = depth.

### Parameters + gradients + Adam state

Per block, dominant matrices are $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$
and the MLP ($d \to 4d \to d$):

$$\text{params per block} = 4d^2 + 2(4d^2) = 12d^2$$

$$\text{total params} = 12Nd^2$$

Gradients match params; Adam adds two extra buffers per parameter:

$$\text{params} + \text{grads} + \text{Adam} = 12Nd^2 \times 4 \times 4\text{ bytes}
= 192Nd^2\text{ bytes}$$

### Activations (dominant term)

Per block, three tensors kept for backward:

$$\text{Attention score matrix: } S \in \mathbb{R}^{B \times h \times L \times L}
\quad \Rightarrow \quad BhL^2 \times 4\text{ bytes}$$

$$\text{Q, K, V tensors: } \in \mathbb{R}^{B \times L \times d} \text{ each}
\quad \Rightarrow \quad 3BLd \times 4\text{ bytes}$$

$$\text{MLP hidden activation: } \in \mathbb{R}^{B \times L \times 4d}
\quad \Rightarrow \quad 4BLd \times 4\text{ bytes}$$

Total activations across $N$ blocks:

$$\text{activations} = N \cdot (BhL^2 + 7BLd) \cdot 4\text{ bytes}$$

### Grand total

$$\boxed{\text{total} = 192Nd^2 + 4N(BhL^2 + 7BLd)\text{ bytes}}$$

**Caveats:** ignores LayerNorm activations and allocator overhead.
VICReg requires two views per image $\Rightarrow$ multiply by $\approx 2$.
Expect actual usage at $2$–$4\times$ this estimate.

### Computed results ($N=6$, $h=6$, $L=144$)

| $d$ | $B$ | params+opt (MiB) | $S$ (MiB) | Q,K,V (MiB) | MLP (MiB) | Total (MiB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 |  64 | 18.0 | 182.2 |  81.0 | 108.0 |  389.2 |
| 128 | 128 | 18.0 | 364.5 | 162.0 | 216.0 |  760.5 |
| 128 | 256 | 18.0 | 729.0 | 324.0 | 432.0 | 1503.0 |
| 192 |  64 | 40.5 | 182.2 | 121.5 | 162.0 |  506.2 |
| 192 | 128 | 40.5 | 364.5 | 243.0 | 324.0 |  972.0 |
| 192 | 256 | 40.5 | 729.0 | 486.0 | 648.0 | 1903.5 |
| 256 |  64 | 72.0 | 182.2 | 162.0 | 216.0 |  632.2 |
| 256 | 128 | 72.0 | 364.5 | 324.0 | 432.0 | 1192.5 |
| 256 | 256 | 72.0 | 729.0 | 648.0 | 864.0 | 2313.0 |

**Key takeaways:**
- Params + optimizer state: 18–72 MiB — negligible vs 8151 MiB (RTX 5060 Laptop).
- At $L=144$, MLP hidden and Q,K,V (linear in $B,d$) outweigh the attention
  score matrix $S$ (quadratic in $L$, but $L$ is small here). $S$ dominates
  only at much longer sequences — relevant for V-JEPA tubelets.
- Largest config ($d=256, B=256$): $\approx 2.3$ GiB — comfortable headroom
  even after doubling for VICReg's two views.

---

## Actual parameter count (measured, not estimated)

`sum(p.numel() for p in encoder.parameters())` on the default `ViTEncoder`
($d=192$, $N=6$, $h=6$, $P=8$, STL-10):

$$\text{Total: } 2{,}729{,}664 \approx 2.7\text{M parameters}$$

| Component | Formula | Count |
|---|---|---:|
| `PatchEmbedding` | $d^2 + d$ | 37,056 |
| `PositionalEmbedding` | $Ld$ | 27,648 |
| `MultiHeadAttention` per block | $4d^2$ | 147,456 |
| `MLP` per block | $8d^2 + 5d$ | 295,872 |
| `LayerNorm` ×2 per block | $2(d+d)$ | 768 |
| 6 blocks subtotal | $444{,}096 \times 6$ | 2,664,576 |
| Final `LayerNorm` | $d + d$ | 384 |
| **Grand total** | | **2,729,664** |

---

## Data augmentation transforms (`torchvision.transforms`)

Used in `contrastive_learning/augmentations.py`. Each transform is applied
independently to the same PIL image twice to produce views $X'$ and $X''$.

**Order constraint (critical):** `ColorJitter`, `RandomGrayscale`,
`GaussianBlur` must come **before** `ToTensor` — they operate on PIL
Images. `Normalize` must come **after** `ToTensor` — it operates on
Tensors. Applying PIL transforms to Tensors raises a cryptic error.

### `T.RandomResizedCrop(size, scale, ratio)`

Randomly crops a region (area $\in$ `scale` $\times$ total area, aspect
ratio $\in$ `ratio`), then resizes to `size`.

**Why it is the most important SSL augmentation:** forces scale and
position invariance — two crops of the same image at different zoom
levels and positions must share a representation:

$$f_{\boldsymbol{\theta}}(\text{crop}_1(\mathbf{x})) \approx
f_{\boldsymbol{\theta}}(\text{crop}_2(\mathbf{x}))$$

Our settings: `scale=(0.08, 1.0)`, `ratio=(3/4, 4/3)`,
`interpolation=BICUBIC`.

### `T.RandomHorizontalFlip(p=0.5)`

Flips the image left-right with probability $p=0.5$. Forces left-right
symmetry invariance. Not applied vertically — natural images are not
vertically symmetric.

### `T.ColorJitter(brightness, contrast, saturation, hue)`

Randomly perturbs brightness, contrast, saturation, and hue. Wrapped in
`T.RandomApply(..., p=0.8)` — fires 80% of the time. Forces invariance
to lighting and color shift.

Our settings: brightness=$0.4$, contrast=$0.4$, saturation=$0.2$,
hue=$0.1$. Hue perturbation is subtle — large hue shifts are semantically
significant (green apple vs red apple).

### `T.RandomGrayscale(p=0.2)`

Converts to grayscale with probability $p=0.2$, outputting 3 channels
(each channel is a copy of the grayscale value). Forces the encoder to
rely on shape and texture rather than color.

### `T.GaussianBlur(kernel_size, sigma)`

Applies a Gaussian blur with $\sigma$ sampled from `sigma=(0.1, 2.0)`.
Wrapped in `T.RandomApply(..., p=0.5)`.

**The math:** convolution with a Gaussian kernel $G_\sigma$:

$$\tilde{I} = I * G_\sigma, \qquad
G_\sigma(x,y) = \frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$

Forces invariance to high-frequency detail — the encoder should care
about semantic structure, not pixel-level sharpness.

### `T.ToTensor()`

Converts a PIL Image (dtype `uint8`, values $[0, 255]$, shape $H \times W \times C$)
to a float32 Tensor (values $[0, 1]$, shape $C \times H \times W$):

$$t_{chw} = \frac{\text{uint8}_{hwc}}{255.0}$$

### `T.Normalize(mean, std)`

Normalizes each channel $c$ independently:

$$t_{chw} \leftarrow \frac{t_{chw} - \mu_c}{\sigma_c}$$

We use ImageNet per-channel statistics:
$\boldsymbol{\mu} = (0.485,\, 0.456,\, 0.406)$,
$\boldsymbol{\sigma} = (0.229,\, 0.224,\, 0.225)$.

After normalization, values can be outside $[0,1]$ — expected and
correct. A pixel of 0 becomes $-\mu_c/\sigma_c \approx -2.1$ for the
red channel. **Do not apply this step before saving images to disk.**
Use `vicreg_augmentation_unnormalized` for visualization.

---

## VICReg loss hyperparameters

From Bardes, Ponce, LeCun (2022), used in `vic_reg_loss/config.py`:

$$\mathcal{L}_\text{VICReg} = \lambda\,\mathcal{L}_\text{inv}(Z', Z'')
+ \mu\,\mathcal{L}_\text{var}(Z', Z'')
+ \nu\,\mathcal{L}_\text{cov}(Z', Z'')$$

| Symbol | Value | Role |
|:---:|:---:|---|
| $\lambda$ | 25 | weight on invariance term |
| $\mu$ | 25 | weight on variance term |
| $\nu$ | 1 | weight on covariance term |
| $\gamma$ | 1 | target std per dimension in $\mathcal{L}_\text{var}$ |
| $\epsilon$ | $10^{-4}$ | numerical stability inside $\sqrt{\cdot}$ in $\mathcal{L}_\text{var}$ |

**CRITICAL — the invariance term's normalization (corrected after a real
collapse bug, see case study below):**

The paper's PROSE, Eq. (5), states:

$$s(Z,Z') = \frac{1}{n}\sum_i \|\mathbf{z}_i - \mathbf{z}_i'\|_2^2
\qquad\text{(sum over $d$ features, then mean over $n$ samples)}$$

But the paper's own RELEASED PSEUDOCODE (Algorithm 1, Appendix A) computes
it differently:

$$\texttt{sim\_loss = mse\_loss(z\_a, z\_b)}
\qquad\text{i.e. } \frac{1}{nd}\sum_i\sum_j(z_{ij}-z'_{ij})^2$$

These disagree by a factor of $d$ (the embedding/projector dimension).
**Our code matches the pseudocode** (`mse_loss`), since that is what was
actually run to produce every published result and what $\lambda=25,
\mu=25,\nu=1$ were calibrated against. Implementing the prose formula
literally makes $\mathcal{L}_\text{inv}$'s gradient $\approx d\times$ too
large relative to $\mathcal{L}_\text{var}$ and $\mathcal{L}_\text{cov}$,
which empirically causes representational collapse (confirmed by direct
experiment — see case study below).

**At collapse** ($f_{\boldsymbol{\theta}}(\mathbf{x}) = \mathbf{c}$ for all $\mathbf{x}$):

$$\mathcal{L}_\text{inv} = 0, \quad \mathcal{L}_\text{var} \approx \gamma = 1,
\quad \mathcal{L}_\text{cov} = 0$$

$$\Rightarrow \mathcal{L}_\text{VICReg} \approx \mu\gamma = 25 > 0$$

Collapse is a **saddle point** of the full objective, not a minimum.
Both $\mathcal{L}_\text{var}$ and $\mathcal{L}_\text{cov}$ jointly push
$S \to \gamma^2 I$ (the sample covariance toward a scaled identity),
while $\mathcal{L}_\text{inv}$ alone would be satisfied at collapse
with zero loss.

---

## Case study: diagnosing representational collapse during real training

This section documents an actual debugging investigation, kept here as a
template for the diagnostic process, not just the bug. The mathematical
signature of collapse, the systematic elimination method used to find the
real cause, and the eventual root cause are all worth internalizing —
this kind of bug (a subtle but consequential mismatch between a paper's
written math and its actual released code) is common in ML research code,
and the *method* used to catch it generalizes far beyond this one case.

### The mathematical signature of collapse (what to watch for)

At collapse, $\mathbf{f}_{\boldsymbol\theta}(\mathbf{x})=\mathbf{c}$ for
all $\mathbf{x}$, so $\tilde Z = Z-\bar Z = 0$, hence
$S=\frac{1}{N-1}\tilde Z^\top\tilde Z = 0$ entirely (every entry, diagonal
and off-diagonal). Tracing this through each loss term:

$$\mathcal{L}_\text{var} = \frac{1}{d}\sum_j\max(0,\gamma-\sqrt{S_{jj}+\epsilon})
\xrightarrow{S_{jj}\to 0} \gamma \quad\text{(rises to its CEILING)}$$

$$\mathcal{L}_\text{cov} = \frac{1}{d}\sum_{j\neq k}S_{jk}^2
\xrightarrow{S_{jk}\to 0} 0 \quad\text{(crashes to exactly zero)}$$

$$\mathcal{L}_\text{inv} = \text{mse}(Z,Z')
\xrightarrow{Z=Z'=\mathbf{c}} 0 \quad\text{(also goes to zero)}$$

**The diagnostic signature is all three happening together**, in this
specific direction: $\mathcal{L}_\text{var}\to\gamma$ (not $\to 0$),
$\mathcal{L}_\text{cov}\to 0$, and $\mathcal{L}_\text{inv}\to 0$
simultaneously. This is genuinely deceptive if you only watch
$\mathcal{L}_\text{total}$ or $\mathcal{L}_\text{inv}$ — both *look* like
training is succeeding (loss going down) right up until you check
$\mathcal{L}_\text{var}$ and $\mathcal{L}_\text{cov}$ specifically and see
they are moving toward their degenerate values instead of away from them.
This is precisely why VICReg logs three separate components rather than
just a scalar total — $\mathcal{L}_\text{total}$ alone is not a reliable
training-health signal.

**Practical rule:** plot $\mathcal{L}_\text{var}$ with a reference line at
$\gamma$, and watch the trend. Healthy training: flat or decreasing.
Collapsing: rising toward the $\gamma$ line. Cross-check with
$\mathcal{L}_\text{cov}$: healthy training keeps it meaningfully nonzero;
collapse drives it toward exactly $0$.

### The debugging method (systematic elimination, in order)

When training showed $\mathcal{L}_\text{var}$ climbing toward $\gamma$ and
$\mathcal{L}_\text{cov}\to 0$ within the first epoch, each of the
following hypotheses was tested as an isolated, falsifiable experiment —
not argued about abstractly:

1. **Mixed precision (AMP) numerical breakdown** — tested by computing the
   same covariance matrix in float16 vs float32 directly; found no
   meaningful precision loss, no inf/NaN. Ruled out.
2. **Vanishing variance through network depth at initialization** — tested
   by measuring per-dimension output std of a naive deep ReLU stack
   (found severe vanishing, std shrinking ~50% per layer) vs the REAL
   `ViTEncoder` (found healthy std ≈0.15, no vanishing — the residual
   connections and LayerNorm were doing their job). Ruled out for the
   real architecture.
3. **Input scale mismatch (ImageNet-normalized vs unit-variance inputs)**
   — tested by comparing encoder output std for both input distributions;
   nearly identical. Ruled out.
4. **Crop augmentation too aggressive** (`RandomResizedCrop(scale=(0.08,1.0))`)
   — tested by softening to `scale=(0.5,1.0)`; collapse trajectory was
   essentially unchanged. Ruled out (or at least, not sufficient alone).
5. **Learning rate too high** — tested across a 20x range
   ($3\times10^{-4}$ down to $10^{-5}$) on a deep toy network; ALL learning
   rates converged to the same collapsed fixed point eventually (just
   slower at lower lr). This was the key clue that ruled out "just needs
   tuning" and pointed at something structural in the loss/gradient
   balance, not the optimizer step size.
6. **BatchNorm separate-pass vs joint-pass statistics** — tested by
   concatenating both views into one batch before the projector (so
   BatchNorm sees joint statistics) vs two independent forward passes;
   measurably different outputs, but the collapse trajectory was
   unchanged either way. Ruled out as the primary cause (though the
   joint-pass version is still architecturally more correct and was kept).

None of these explained it. The actual breakthrough came from comparison
against an external, verified-working reference implementation
(a from-scratch CIFAR-10 VICReg blog post with public, runnable code),
substituting pieces of OUR implementation into THEIR training loop one at
a time:

- **Our `ViTEncoder` + their loss + their projector** → trained correctly
  ($\mathcal{L}_\text{var}$ decreasing). Encoder ruled out.
- **Our `ViTEncoder` + our `Projector` + their loss** → also trained
  correctly. Our projector ruled out.
- **Our `ViTEncoder` + our `Projector` + OUR `VICRegLoss`** → collapsed,
  reproduced in complete isolation, same random seed, same data. **The
  loss class was the cause.**

Comparing our individual loss functions against the reference's
function-by-function on identical input pinpointed it exactly:
`invariance_loss` differed by a factor of exactly $d$ (our projector
dimension, 512) — `variance_loss` and `covariance_loss` matched the
reference exactly.

### The actual root cause

Checking the original paper's PDF directly (not a secondary source)
confirmed the prose equation (Eq. 5) and the released pseudocode
(Algorithm 1) use different normalizations for the invariance term,
differing by exactly the factor of $d$ found in the comparison above. Our
original implementation (built by literally implementing the prose
equation, and at the time even "fixing" a unit test that disagreed with
that interpretation) was internally consistent but matched the wrong
artifact — the formula's literal text, not the actual code that produced
the calibrated $\lambda=25,\mu=25,\nu=1$ hyperparameters.

**Fix:** `invariance_loss` now calls `torch.nn.functional.mse_loss(z_a, z_b)`
directly (averages over all $N\times d$ elements via PyTorch's default
`reduction='mean'`, implemented in a compiled C++/ATen kernel —
`torch._C._nn.mse_loss` — not visible as explicit Python arithmetic, but
verified numerically equivalent to `((z_a-z_b)**2).sum(dim=1).mean() / d`).

**Lesson for the lab notebook:** when a paper's prose and its released
code disagree, the code is ground truth for reproducing published
results — prose can contain transcription errors, simplifications, or
inconsistencies that never get caught because reviewers read the
equations, not the pseudocode line by line. When implementing a method
from a paper, cross-check the prose against any released pseudocode or
official repository before trusting either alone.

---

## Bug note: resuming past `cfg.epochs` crashes with `UnboundLocalError`

**Symptom:** resuming training from a checkpoint via `--resume` crashed
immediately, before running a single epoch, with
`UnboundLocalError: cannot access local variable 'avg' where it is not
associated with a value`.

**Cause:** `train()`'s main loop is `for epoch in range(start_epoch,
cfg.epochs):`, and the variable `avg` (the per-epoch loss summary dict)
is only ever assigned *inside* that loop body. The final "always save a
checkpoint" call at the end of `train()` references `avg["total"]`
unconditionally. If `start_epoch >= cfg.epochs` -- e.g. resuming a
checkpoint saved at epoch 129 while `config.py` had been left at its
default `epochs=100` (a value that had been overridden to 200
in-memory for the original run, but never saved back to the file) --
`range(129, 100)` is empty, the loop body never executes, and `avg` is
never created. Python then raises `UnboundLocalError` rather than
`NameError`, specifically because `avg` IS assigned somewhere in the
function's body (inside the loop), so Python treats it as a local
variable throughout the function -- it's "unbound" rather than
"undefined", a subtlety of how Python scoping works (a variable
assigned anywhere in a function is local to that function for its
entire body, even before the assignment line executes).

**Fix:** guard the final save with `if "avg" not in locals():` and print
a clear warning instead of crashing, covering the case where the
training loop legitimately has nothing left to do.

**Practical lesson:** when overriding a config value for a single run
(e.g. setting `epochs=200` by editing the dataclass instance in code,
or intending to but not actually persisting it), remember the override
does not survive a process restart -- if you need to resume an
interrupted run, the config file on disk is what determines the new
process's behavior, not whatever was true of the previous process's
in-memory state. Persist intentional overrides to the actual config
file (or pass them via a working `--epochs` CLI flag) rather than
relying on a one-off in-memory change.