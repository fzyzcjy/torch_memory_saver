# Repository Guidelines

## Scope

- Keep pull requests atomic. If feature work uncovers a pre-existing or cross-backend issue, fix it in a separate pull request with a regression test and land that first.
- When a backend skips or changes an existing check or behavior, determine whether the difference is truly backend-specific. Prefer fixing the common path when it is not.

## Implementation

- Use backend-neutral PyTorch APIs such as `torch.get_device_module()` when behavior is shared; do not duplicate equivalent CUDA, ROCm, and XPU branches.
- Put shared helpers in `utils.py` instead of repeating them.
- Guard backend-specific functions and keep platform-specific setup code grouped together.

## Tests

- Write shared scenarios so they run on every backend that supports the behavior; do not make a test XPU-only when CUDA or ROCm can exercise the same path.
- Do not weaken existing coverage while adapting tests. Preserve preload and torch modes wherever each mode is supported.

## Comments and documentation

- Keep only comments that explain non-obvious constraints, and make them concise.
- Keep README text brief and do not use bold text.
