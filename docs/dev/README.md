# Developer notes

## Pipeline map

[`pipeline_map.toml`](pipeline_map.toml) lists every input format, matrix
type, windowed engine and public function in pg_gpu, with the parallel
implementations of each statistic and the axes that a change must cover.
Read it before you change a statistic or a rule; its header says how to use
it.

### The drift test

`tests/test_pipeline_map.py` fails when the map and the code disagree. It
needs no GPU, so CI runs it on every pull request (the "Pipeline map" job).
Run it locally with:

```bash
pixi run check-map
```

It checks the modules, the public functions and their axis keywords, the
loaders and their parameters, the matrix methods, the engine of each
windowed statistic, alias rows, and every dotted code name in the map. It
does not check the free-text fields (`span`, `undefined`, `streaming`,
`notes`); keep those true by hand.

It also checks how each statistic is reached by name, because a name that
one dispatcher knows and another does not fails only when a user mixes
them:

- **Engine gaps.** A `[windowed]` row lists, under `not_in`, each engine
  family (scatter, fused, fallback) that does not compute the statistic,
  with the issue that tracks the gap or `"untracked"`. No request runs
  unless one family computes every name in it, so a statistic in one family
  only cannot be mixed with a name outside that family.
- **Batch dispatchers.** `[dispatch]` lists the names that
  `diversity_stats`, `divergence_stats` and `compute_ld_statistics` accept,
  from the tables in the code. Its `not_named` lists every other public
  function of the module, with the issue that tracks its absence,
  `"untracked"`, or `"n/a"` when it is not a statistic of that kind.
- **Aliases.** `[[alias]]` lists the statistics that have more than one
  name, and which names each dispatcher accepts.

### When the test fails

The message names each entry to add, remove or correct. Typical causes:

- **A new public function:** add a `[functions."module.name"]` row. Copy a
  row of a similar function and fill in each field.
- **A changed signature:** update `axis_kwargs`, or `params` for a loader.
- **A renamed or removed name:** fix every map field that names it.
- **A windowed statistic that moved to another engine:** update the
  `engines` list of its `[windowed]` row. The engine sets are the module
  constants in `windowed_analysis.py` that each `[engines]` row names.

- **A statistic added to or removed from an engine or a batch dispatcher:**
  update `not_in`, `[dispatch]` and `[[alias]]` as the message says. To close a gap, delete its `not_in`
  entry in the pull request that adds the statistic to the engine.

Make the map change in the same pull request as the code change.
