import torch
import gpytorch


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood):
        super().__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
        self.cov_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2]), ard_num_dims=3),
            batch_shape=torch.Size([2]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.cov_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
