# Multi-phase solar-sail GEO disposal (benchmark)

The first problem of Kelly and Bevilacqua, *"Geostationary debris mitigation
using minimum time solar sail trajectories with eclipse constraints"* (2021):
move a sail-equipped spacecraft from circular GEO to a graveyard orbit with
eccentricity at most `0.003` and perigee at least `r_GEO + 250 km`, in minimum
time, using solar radiation pressure alone. Each sunlit and eclipsed arc is a
separate trajopt phase. With `A/m = 10 m^2/kg` the transfer takes 21
revolutions (41 phases).

This folder uses the problem as a benchmark for the solvers and methods in
trajopt. The development of the initial solutions is archived in
[`__old/`](__old/): the original README, `FINDINGS.md`, the `config.yaml` /
`config-ps.yaml` aliases and the `results/` / `results-ps/` figures.

## Running it

```bash
python main.py algorithms/ms.yaml --no-search        # RK4 multiple shooting
python main.py algorithms/p.yaml --no-search         # Radau collocation (default)
python main.py algorithms/p_to_ms_to_hp.yaml         # flow, with the revolution search
python compare.py algorithms/ms.yaml algorithms/p.yaml --csv compare.csv
```

Results go to `results/<algorithm file name>/` unless the file or `--output`
says otherwise. Every run is checked by `solution.validation_report`, which
re-propagates the trajectory end to end; a solution counts only if it is
`NUMERICALLY FEASIBLE`.

## Reference solutions (21 revolutions, current trajopt)

| | ms (`algorithms/ms.yaml`) | p (`algorithms/p.yaml`) |
| --- | --- | --- |
| SCP iterations | 30 | 72 |
| elapsed time | 20.1007 d | 20.1123 d |
| true r_p - r_GEO / e | +250.32 km / 0.003000 | +250.09 km / 0.002994 |
| accumulated defect | -0.005 km | -0.004 km |
| validated | yes | yes |

## Files

| file | role |
| --- | --- |
| `problem.yaml` | the problem: physical and structural settings, no solver settings |
| `algorithms/` | ways of solving it, single methods and flows; see `algorithms/README.md` |
| `solar_sail.py` | dynamics, sail steering law, shadow geometry, terminal cones |
| `problem.py` | expands the settings into the alternating trajopt phases |
| `algorithm.py` | loads an algorithm file and runs its stages, handing guesses between them |
| `search.py` | outer loop over the revolution count (the paper's Algorithm 1) |
| `solution.py` | extraction, independent validation, save/load |
| `visualization.py` | summary dashboard, 3D still, shadow animation |
| `main.py` | one algorithm: solve (or search), validate, save, draw |
| `compare.py` | several algorithms on one problem: per-stage table and CSV |
| `__old/` | the initial-solution work, kept for reference |
