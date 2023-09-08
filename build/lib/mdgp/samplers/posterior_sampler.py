import torch 
import gpytorch 
from typing import Tuple


class PosteriorSampler(torch.nn.Module): 
    def __init__(self, rff_sampler, vi_sampler, inducing_points, inv_jitter=10e-6, whitened_variational_strategy=True):
        super().__init__()
        self.rff_sampler = rff_sampler
        self.vi_sampler = vi_sampler
        self.inducing_points = inducing_points
        self.inv_jitter = inv_jitter
        self.whitened_variational_strategy = whitened_variational_strategy

    def sample_prior(self, x, z, sample_shape: torch.Size = torch.Size([]), normalize_kernel=True) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi_w_x = self.rff_sampler(x, sample_shape=sample_shape, normalize_kernel=normalize_kernel)
        Phi_w_z = self.rff_sampler(z, sample_shape=sample_shape, normalize_kernel=normalize_kernel, resample_weights=False)


        # TODO Check if the concatenation version is faster. It might not be, since we have to broadcast the z to the shape of x and that's redundant computation
        # Phi_w_x_cat_z = self.rff_sampler(torch.cat([x, z]), sample_shape=sample_shape, normalize_kernel=normalize_kernel) # [S, O, N + M] or [S, N + M]
        # Phi_w_x = Phi_w_x_cat_z[..., :x.shape[0]] # [S, O, N] or [S, N]
        # Phi_w_z = Phi_w_x_cat_z[..., x.shape[0]:] # [S, O, M] or [S, M]
        return Phi_w_x, Phi_w_z

    def sample_variational(self, sample_shape: torch.Size = torch.Size([])) -> torch.Tensor:
        return self.vi_sampler(sample_shape=sample_shape) # [S, O, M] or [S, M]

    def compute_posterior_update(self, x, z, u, Phi_w_z, normalize=True): 
        k_x_z = self.rff_sampler.covar_module(x, z, normalize=normalize) # [O, N, M] or [N, M]
        K_z_z = self.rff_sampler.covar_module(z, z, normalize=normalize) # [O, M, M] or [M, M]
        # FIXME temporary fix. Move this to VISampler
        if self.whitened_variational_strategy:
            u = torch.einsum('...mn, ...n -> ...m', K_z_z.cholesky().evaluate(), u) + self.rff_sampler.mean_module(z)
        delta = u - Phi_w_z # [S, O, M] or [S, M]

        K_z_z = K_z_z.expand(*delta.shape[:-1], *K_z_z.shape[-2:]) # [S, O] + [M, M] or [S] + [M, M]
        k_x_z = k_x_z.expand(*delta.shape[:-1], *k_x_z.shape[-2:]).evaluate() # [S, O] + [N, M] or [S] + [N, M]
        delta = delta.unsqueeze(-1) # [S, O, M, 1] or [S, M, 1]

        return gpytorch.solve((K_z_z.add_jitter(self.inv_jitter)), delta, k_x_z).squeeze(-1)

    def forward(self, x, sample_shape: torch.Size = torch.Size([]), normalize_kernel=True) -> torch.Tensor:
        z = self.inducing_points # [M, D]

        # Step 1. Get prior samples from RFF
        Phi_w_x, Phi_w_z = self.sample_prior(x=x, z=z, sample_shape=sample_shape, normalize_kernel=normalize_kernel) # [S, O, N], [S, O, M] or [S, N], [S, M]

        # Step 2. Get prior sample from VI
        u = self.sample_variational(sample_shape=sample_shape) # [S, O, M] or [S, M]

        # Step 3. Update prior 
        update = self.compute_posterior_update(x=x, z=z, u=u, Phi_w_z=Phi_w_z) # [S, O, N] or [S, N]
        return Phi_w_x + update 
