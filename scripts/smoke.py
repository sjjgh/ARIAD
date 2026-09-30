"""Small CPU forward/backward demo using the released experiment models."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['single', 'qk'], default='single')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = 'ariad_single_space' if args.variant == 'single' else 'ariad_qk_space'
    sys.path.insert(0, str(root / folder))
    torch.manual_seed(0)
    torch.set_num_threads(2)
    if args.variant == 'single':
        from ariad_train_single import ARIAD_CONFIG, AriadSingleConfig, OneLayerSingleSpaceAttention
        model = OneLayerSingleSpaceAttention(128, 64, 64, AriadSingleConfig(**ARIAD_CONFIG))
    else:
        from ariad_train_qk import (QQ_CONFIG, KK_CONFIG, QK_CONFIG,
                                    AriadSingleConfig, AriadQKConfig, OneLayerQKAttention)
        model = OneLayerQKAttention(128, 64, 64, AriadSingleConfig(**QQ_CONFIG),
                                   AriadSingleConfig(**KK_CONFIG), AriadQKConfig(**QK_CONFIG))
        # Match the fixed standardized-value projection of the QK sweep.
        with torch.no_grad():
            model.w_v.zero_()
            model.w_v[64:, :] = torch.eye(64)
        model.w_v.requires_grad_(False)
    x = torch.randn(256, 128)
    target = torch.randn(256, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        model.train()
        optimizer.zero_grad()
        output, neighbors, *_ = model(x)
        loss = (output - target).square().mean()
        assert torch.isfinite(loss), 'nonfinite loss'
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        optimizer.step()
    assert neighbors.shape == (256, 32)
    assert ((neighbors >= 0) & (neighbors < 256)).all()
    assert not (neighbors == torch.arange(256)[:, None]).any()
    ordered = neighbors.sort(dim=1).values
    assert not (ordered[:, 1:] == ordered[:, :-1]).any()
    previous = neighbors.clone()
    model.eval()
    with torch.no_grad():
        evaluated, reused, *_ = model(x)
    assert torch.isfinite(evaluated).all()
    assert torch.equal(previous, reused), 'eval must reuse the training graph'
    print(f'{args.variant}: forward/backward OK; graph={tuple(neighbors.shape)}; '
          f'eval graph reused; final demo loss={loss.item():.6f}')


if __name__ == '__main__':
    main()
