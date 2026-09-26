# Adaptive methods

Run commands from `ParetoMTL/palora` in an environment with the project
`requirements.txt` installed. NashMTL needs CVXPY and ECOS (the default solver);
FairGrad uses SciPy's `least_squares`; ExcessMTL uses PyTorch. The existing pinned
requirements target the project's older PyTorch environment, not every current
Python release. See the repository's `THIRD_PARTY_NOTICES.md` for provenance and
licenses.

```bash
python _sarcos.py method=nashmtl method.fim_method=none
python _sarcos.py method=fairgrad method.fim_method=madam
python _sarcos.py method=excessmtl method.fim_method=diag_madam
```

All three methods accept `none`, `madam`, and `diag_madam`; they are registered
as non-Pareto baselines. General method configs are available to the other PaLoRA
entrypoints as well. Dataset/training defaults are unchanged.

NashMTL weights both shared and task-head gradients. FairGrad preserves the
official task-count scaling of the shared direction and **unweighted** head
gradients. ExcessMTL preserves its all-history squared-gradient accumulator,
first-batch excess-risk reference, and weights summing to the number of tasks;
these weights apply to both shared and head gradients. MAdam acts on the declared
shared parameter block only. NashMTL and FairGrad default to shared `max_norm=1`
clipping **before** MAdam, with AMP gradients unscaled before clipping.

Self-contained composition controls are explicit in the method configs:

- `fim_lambda_policy=aggregation_raw` uses aggregation coefficients as MAdam
  preference; `aggregation_simplex` normalizes them, and `uniform` uses equal
  preference. Neither alternative changes the aggregation coefficients themselves.
- `fim_preserve_shared_grad_norm=false` leaves MAdam's output scale intact.
  Setting it true restores the incoming shared norm before optional FIM clipping.
- `fim_solver_geometry=raw` uses raw task gradients. NashMTL/FairGrad optionally
  accept `lagged_preconditioned`, using an existing MAdam denominator (raw on the
  first step). It requires identical solver/FIM parameter order; BatchNorm
  exclusion that changes the partition is rejected. ExcessMTL supports raw only.
- `max_norm_clip_order=after_madam` is an optional alternative for NashMTL/FairGrad,
  not the default official-method composition. `fim_clip_grad_norm` is separate,
  optional post-preconditioning clipping.
- NashMTL reports solver fallback counts and can opt into a failure-rate guard;
  FairGrad rejects solves that do not satisfy `solver_rtol`.

Existing result JSONs include the new methods' resolved `method_config` and
`method_diagnostics` when saved through the entrypoints.
