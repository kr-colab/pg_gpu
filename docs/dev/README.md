# Developer notes

## Pipeline map

[`pipeline_map.toml`](pipeline_map.toml) lists every input format, matrix
type, windowed engine and public function in pg_gpu, with the parallel
implementations of each statistic and the axes that a change must cover.
Read it before you change a statistic or a rule; its header says how to use
it. `tests/test_pipeline_map.py` fails when the map and the code disagree,
and names the entry to correct.
