from .builder import build_data_loader
from .detect import KKD_CTW, KKD_IC15, KKD_MSRA, KKD_TT, KKD_Synth

__all__ = [
    'KKD_IC15', 'KKD_TT', 'KKD_CTW', 'KKD_MSRA', 'KKD_Synth'
]
