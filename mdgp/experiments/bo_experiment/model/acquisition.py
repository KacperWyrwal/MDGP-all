from torch import Tensor 


import warnings 
from botorch.acquisition import AnalyticAcquisitionFunction


class DeepAnalyticAcquisitionFunction(AnalyticAcquisitionFunction):

    def __init__(self, base_acquisition: AnalyticAcquisitionFunction):
        """
        Initialize the DeepAnalyticAcquisitionFunction.
        
        Args:
        - base_acquisition (AnalyticAcquisitionFunction): An instance of an AnalyticAcquisitionFunction.
        """
        super().__init__(base_acquisition.model)  # Pass the model from the base acquisition function
        self.base_acquisition = base_acquisition
        
    def forward(self, X: Tensor) -> Tensor:
        """
        Overload the forward method to apply mean to its output.

        Args:
        - X (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The mean of the output from the base acquisition function.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = self.base_acquisition(X)
        return output.mean(dim=0)
