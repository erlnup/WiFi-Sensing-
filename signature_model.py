import torch
import torch.nn as nn


class SignatureModel:

    def __init__(self,x, signature_dimensionality = 10):
        self.input = x
        self.batch_size = x.shape[0]
        self.d_model = x.shape[1]
        self.signature_dimensionality = signature_dimensionality

        print(f"Initialized SignatureModel with batch_size = {self.batch_size}, d_model = {self.d_model}, signature_dimensionality = {self.signature_dimensionality}")



    def linear_layer(self, x):
        """
        Input x is the output of the transformer encoder: x shape (batch_size, features) Currently: (342,16) -> 342 rows with 16 featurs 
        One sample is 1 row
        We have 342 rows 

        Split the 342 samples into two list, one query list and one gallery list 
        """

        linear_layer = nn.Linear(self.d_model, self.s)
        output_linear_layer = linear_layer(x)


        return output_linear_layer





    def l2_normalization(self,x):
        output_l2_normalization = nn.functional.normalize(x, p=2, dim=1)  # normalize each row

        return output_l2_normalization

        

