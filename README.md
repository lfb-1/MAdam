# MAdam

This repository contains the active MAdam implementation across two benchmark
integrations:

- `madam_core/`: shared Fisher preconditioning core.
- `ParetoMTL/palora/`: Pareto multi-task learning experiments.
- `PINN/PINNacle/`: physics-informed neural network experiments.

## Key Commands

ParetoMTL / PaLoRA:

```bash
cd ParetoMTL/palora
python _multimnist.py method.fim_method=madam seed=0
```

PINNacle:

```bash
cd PINN/PINNacle
python benchmark.py --case Burgers1D --optimizer adam --preconditioner madam --seed 0 --iter 10
```

## Method Names

- Full MAdam: `madam`
- Diagonal MAdam: `diag_madam`
- Baseline without MAdam: `none`

PaLoRA also supports `method=nashmtl`, `method=fairgrad`, and `method=excessmtl`
with each FIM mode. See [adaptive methods usage](ParetoMTL/palora/docs/methods.md).
