from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior


class BotorchGP(Model):
    num_outputs = 1
    def __init__(self, base_model, posterior_sample_method='pathwise'):
        super().__init__()
        self.base_model = base_model
        self.likelihood = self.base_model.likelihood
        self.posterior_sample_method = posterior_sample_method

    def forward(self, X, sample_hidden='naive'):
        # Temporary fix resulting from the shape with optimize_acqf and q=1 
        if X.shape[-2] == 1: 
            X = X.squeeze(-2)
        return self.base_model(X, sample_hidden=sample_hidden)

    def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None, **kwargs):
        was_training = self.training

        self.eval()
        # Applying input transforms
        X = self.transform_inputs(X)
        
        # Get the predictive prior
        mvn = self.forward(X, sample_hidden=self.posterior_sample_method)
        
        # If we need to consider observation noise, apply likelihood
        if observation_noise:
            mvn = self.likelihood(mvn)
            
        posterior = GPyTorchPosterior(mvn)

        if posterior_transform is not None: 
            posterior = posterior_transform(posterior)

        if was_training:
            self.train()
        return posterior
