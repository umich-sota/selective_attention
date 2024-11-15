# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch


class GMRNorm(torch.nn.Module):
    """Global Mean Reduce Layer Normalization.
    For a sequence of vectors :math:`x = (x_1, ..., x_m)`, GMRNorm subtracts the mean
    of all previous vectors for each vector. This is similar to Layer Normalization.
    GMRNorm is defined as:
    .. math::
        GMRNorm(x_i) = \frac{x_i - \mu(x_i..x_0)}{\sigma(x_i)} * \gamma + \beta
    where :math:`\mu(x_i..x_0)` is the mean of all previous vectors.
    Optimally we want to compute mean of all vectors, but to keep causal property
    we can only compute mean of previous vectors.
    To do so, we need to construct a causal mask matrix to mask out all future vectors
    and by multiplying this mask matrix with the input vector we can get the mean of
    all previous vectors. It should be a upper triangular where each element in each
    column is 1/(i+1) where i is the column index. Thus, So by multiplying this mask
    matrix with the input vector we can get the mean of all previous vectors.
    Example of causal mask matrix:
    .. code-block:: text
        [[1, 1/2, 1/3, 1/4, 1/5],
        [0,   1/2, 1/3, 1/4, 1/5],
        [0,    0,  1/3, 1/4, 1/5],
        [0,    0,   0,  1/4, 1/5],
        [0,    0,   0,   0,  1/5]]
    """

    def __init__(self,
                 size: int,
                 dim: int = -1,
                 eps: float = 1e-5,
                 block_size: int = 4096,
                 elementwise: bool = True) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.size = size
        self.block_size = block_size
        self.elementwise = elementwise
        self.register_buffer('causal_matrix', self._get_causal_matrix())
        # self.causal_matrix = torch.nn.Parameter(self._get_causal_matrix())
        self.causal_mask = torch.triu(
            torch.ones((self.block_size, self.block_size),
                       dtype=torch.float32))
        print("Use GMRNorm: init causal mean matrix")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # Get mean of all vectors
        # (B, T, C) -> (B, T)
        B, T, C = x.shape
        if not self.elementwise:
            mean = torch.mean(x, dim=self.dim)
            # do causal mean, on each example, multiply by causal matrix
            mean = torch.matmul(
                mean, self.causal_matrix[:T, :T].masked_fill(
                    self.causal_mask[:T, :T] == 0, 0)).unsqueeze(-1)
        else:
            # mean = torch.matmul(
            #     x.transpose(-1, -2), self.causal_matrix[:T, :T].masked_fill(
            #         self.causal_mask[:T, :T] == 0, 0)).transpose(-1, -2)
            mean = torch.matmul(
                x.transpose(-1, -2), self.causal_matrix[:T, :T]).transpose(-1, -2)
        
        x = x - mean

        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.weight * x_normed).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _get_causal_matrix(self) -> torch.Tensor:
        """Construct a causal mask matrix.
        Returns:
            torch.Tensor: causal mask matrix
        """

        causal_matrix = torch.triu(
            torch.ones((self.block_size, self.block_size),
                       dtype=torch.float32))
        causal_matrix /= torch.arange(1,
                                      self.block_size + 1,
                                      dtype=torch.float32)
        if self.elementwise:
            # causal_matrix /= 2
            causal_matrix[0, 0] = 0

        return causal_matrix
