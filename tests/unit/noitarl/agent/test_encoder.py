from __future__ import annotations

import torch

from noitarl.agent.encoder import ImpalaEncoder


class TestImpalaEncoder:
    def test_output_shape(self) -> None:
        encoder = ImpalaEncoder((3, 64, 64), [16, 32, 32], 64, 1)
        batch = torch.zeros(2, 3, 64, 64)
        output = encoder(batch)
        assert output.shape == (2, 64)
