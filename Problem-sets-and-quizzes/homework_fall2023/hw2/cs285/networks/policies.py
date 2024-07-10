import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            )
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            )
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action

        if(len(obs.shape)>1):
            obs = obs
        else:
            obs = obs[None] 
        
        ob = ptu.from_numpy(obs)
        action_distribution = self(ob)
        action = action_distribution.sample()
        action = ptu.to_numpy(action)

        

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits = logits)
            
        else:
            # TODO: define the forward pass for a policy with a continuous action space.

            batch_mean = self.mean_net(obs)
            logstd_clamped = torch.clamp(self.logstd, min=-20, max=2)
            scale_tril = torch.diag_embed(torch.exp(logstd_clamped).expand(batch_mean.shape[0], -1))
            action_distribution = distributions.MultivariateNormal(batch_mean, scale_tril=scale_tril)
            

        return action_distribution

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""

        # TODO: implement the policy gradient actor update.

        print("advantages:",advantages)
        advantages = ptu.from_numpy(advantages)
        ob = ptu.from_numpy(obs)
        acs = ptu.from_numpy(actions)
         
        action_distribution = self(ob)
        if (self.discrete):
            # log_probs = action_distribution.log_prob(acs)
            entropy = action_distribution.entropy().mean()
            print("updating discrete")
        else:
            # log_probs = action_distribution.log_prob(acs).sum(dim = -1)
            # print("log probs", log_pi)
            entropy = action_distribution.entropy().sum(dim=-1).mean()
            
            

            print("updating cont")

        log_pi = self.forward(ob).log_prob(acs)
        loss = - torch.mean(torch.mul(log_pi, advantages)) - 0.01* entropy


        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()


        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
