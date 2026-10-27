"""ALS with an actual continuous preference target, not changed confidence."""
import numpy as np
from .ials import ImplicitALS


class CustomTargetALS(ImplicitALS):
    def targets(self, data):
        target = data.counts.copy().astype(np.float64)
        # Identical sparsity and confidence to baseline ALS. Only p_ui changes.
        repeats = 0.6 + 0.4 * (1.0 - np.exp(-(target.data - 1.0) / 2.0))
        recency = 0.5 + 0.5 * np.exp(-data.age_days.data / 90.0)
        target.data = repeats * recency
        return target
