"""Count actual trainable CNN parameters; never add unused padding tensors."""

from torch import nn


def matched_plain_cnn(cin, cout, target, *, fixed_parameters=0, tolerance=0.01):
    best = None
    for blocks in range(1, 9):
        for width in range(1, 1025):
            for first_kernel in (3, 5, 7, 9):
                for hidden_kernel in (3, 5, 7):
                    count = (
                        first_kernel * cin * width
                        + 3 * width
                        + blocks * (hidden_kernel * width * width + 3 * width)
                        + width * cout
                        + cout
                        + fixed_parameters
                    )
                    candidate = (
                        abs(count - target),
                        width,
                        blocks,
                        first_kernel,
                        hidden_kernel,
                        count,
                    )
                    if best is None or candidate < best:
                        best = candidate
    error, width, blocks, first_kernel, hidden_kernel, count = best
    if error / target > tolerance:
        raise ValueError(
            f"Plain CNN cannot match {target} parameters within {tolerance:.1%}; closest is {count}."
        )
    layers = [
        nn.Conv1d(cin, width, first_kernel, padding=first_kernel // 2),
        nn.BatchNorm1d(width),
        nn.GELU(),
    ]
    for _ in range(blocks):
        layers.extend(
            [
                nn.Conv1d(width, width, hidden_kernel, padding=hidden_kernel // 2),
                nn.BatchNorm1d(width),
                nn.GELU(),
            ]
        )
    layers.extend([nn.Conv1d(width, cout, 1), nn.GELU()])
    module = nn.Sequential(*layers)
    actual = sum(p.numel() for p in module.parameters()) + fixed_parameters
    if actual != count:
        raise RuntimeError("CNN parameter accounting disagreement.")
    return module, {
        "target_parameters": target,
        "actual_parameters": actual,
        "relative_difference": error / target,
        "width": width,
        "hidden_blocks": blocks,
        "first_kernel": first_kernel,
        "hidden_kernel": hidden_kernel,
        "tolerance": tolerance,
    }
