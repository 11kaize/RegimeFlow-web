# CPU validation sources

These two files preserve cached upstream sources from state-spaces/mamba v2.2.6:
- https://github.com/state-spaces/mamba/blob/v2.2.6/mamba_ssm/modules/mamba_simple.py
- https://github.com/state-spaces/mamba/blob/v2.2.6/mamba_ssm/ops/selective_scan_interface.py

`upstream_cpu.py` executes only the upstream `Mamba` class and
`selective_scan_ref` function through AST selection. CUDA, Triton and fused paths
are disabled; no scan or projection math is replaced. This is an independent
CPU semantic reference, not a CUDA fused-kernel measurement. The original
copyright headers are preserved. Source hashes are recorded in validation output.

The upstream files are distributed under Apache-2.0; see `LICENSE`.
The original whitespace is preserved for source-hash verification;
`.gitattributes` exempts only these two snapshots from trailing-space checks.
