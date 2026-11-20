from .encoder import Encoder
from .encoder_dual_lens import EncoderDualLens, EncoderDualLensCfg

ENCODERS = {
    "dual_lens": EncoderDualLens,
}

EncoderCfg = EncoderDualLensCfg


def get_encoder(cfg: EncoderCfg) -> Encoder:
    return ENCODERS[cfg.name](cfg)
