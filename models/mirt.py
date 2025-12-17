"""
Multidimensional Item Response Theory (MIRT) model integration.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel


def irt2pl(theta, a, b, *, F_backend=torch):
    """
    2-parameter logistic function used by MIRT.
    """
    return 1 / (1 + F_backend.exp(-(theta * a).sum(dim=-1) + b))


class MIRTNet(nn.Module):
    """
    Neural network that parameterises the MIRT model.
    """

    def __init__(self, llm_input_dim, item_input_dim, latent_dim, a_range, theta_range, irf_kwargs=None):
        super().__init__()
        self.llm_input_dim = llm_input_dim
        self.item_input_dim = item_input_dim
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        self.theta = nn.Linear(llm_input_dim, latent_dim, bias=False)
        self.a = nn.Linear(item_input_dim, latent_dim, bias=False)
        self.b = nn.Linear(item_input_dim, 1, bias=False)

        self.a_range = a_range
        self.theta_range = theta_range

    def forward(self, llm_features, item_features):
        theta = torch.squeeze(self.theta(llm_features), dim=-1)
        a = torch.squeeze(self.a(item_features), dim=-1)

        if self.theta_range is not None:
            theta = self.theta_range * torch.sigmoid(theta)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)

        b = torch.squeeze(self.b(item_features), dim=-1)

        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):
            raise ValueError("theta, a, or b contains NaNs. Consider reducing the a_range or theta_range.")

        pred = self.irf(theta, a, b, **self.irf_kwargs)
        return pred, theta, a, b

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F_backend=torch)


class MIRTModel(BaseModel):
    """
    Wrapper module that exposes MIRT as a standard PyTorch model.
    """

    def __init__(
        self,
        num_llms,
        llm_embedding_dim,
        item_input_dim,
        latent_dim,
        a_range=None,
        theta_range=None,
        irf_kwargs=None,
    ):
        super().__init__(
            input_dim=item_input_dim,
            hidden_dims=[],
            output_dim=1,
            dropout=0.0,
            activation="relu",
        )

        self.num_llms = num_llms
        self.llm_embedding_dim = llm_embedding_dim
        self.latent_dim = latent_dim

        self.mirt_net = MIRTNet(
            llm_input_dim=llm_embedding_dim,
            item_input_dim=item_input_dim,
            latent_dim=latent_dim,
            a_range=a_range,
            theta_range=theta_range,
            irf_kwargs=irf_kwargs,
        )

        self._latest_theta = None
        self._latest_a = None
        self._latest_b = None

    def forward(self, llm_features, item_features):
        if llm_features.shape[-1] != self.llm_embedding_dim:
            raise ValueError(
                f"llm_features must have last dimension {self.llm_embedding_dim}, "
                f"got {llm_features.shape[-1]}"
            )

        pred, theta, a, b = self.mirt_net(llm_features, item_features)
        pred = pred.clamp(min=1e-6, max=1 - 1e-6)

        self._latest_theta = theta
        self._latest_a = a
        self._latest_b = b

        return pred, theta, a, b

    def get_latest_parameters(self):
        return {
            "theta": self._latest_theta,
            "a": self._latest_a,
            "b": self._latest_b,
        }

    def get_model_info(self):
        info = super().get_model_info()
        info.update(
            {
                "num_llms": self.num_llms,
                "llm_embedding_dim": self.llm_embedding_dim,
                "latent_dim": self.latent_dim,
                "mirt": True,
            }
        )
        return info

