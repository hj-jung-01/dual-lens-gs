from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: Float[Tensor, "batch view 3 height width"],
        target: Float[Tensor, "batch view 3 height width"],
        global_step: int,
    ) -> Float[Tensor, ""]:
        return self.cfg.weight * ((prediction - target) ** 2).mean()
