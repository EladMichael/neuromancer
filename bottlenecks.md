# Performance Bottlenecks in NeuroMANCER (vanilla flow)

Scope: the standard training path used by examples like building DPC — `System` rollout, `Problem`/`Loss`/`Constraint` evaluation, `Trainer`, plain `MLP`/`ResMLP` blocks with `slim.Linear`, and the Euler/RK4 integrators. KAN blocks, exotic `slim` parametrizations (SVD/orthogonal/Schur/etc.), and networked ODEs are opt-in and out of scope — they were checked and confirmed *not* on the default path.

Environment used for measurements below: torch 2.10.0, Python 3.10.18, CPU, repo at `e9456ff`.

## TL;DR

There is one dominant, fixable bottleneck and one pervasive architectural pattern that compounds it:

1. **`System.forward` grows its output tensors with `torch.cat` on every rollout step** (`src/neuromancer/system.py:234-246, 261-277`). This is O(n²) in the number of rollout steps. Measured on the actual `System` class with a toy 2-node closed loop (batch=100, nx=4, nu=1): **676ms at nsteps=2000 vs. 44ms for a collect-then-stack-once rewrite — a 15x slowdown from this pattern alone**, and it gets worse the longer the horizon (closed-loop test rollouts in the DPC examples use nsteps=2000; long-horizon sim/eval will feel this hardest).
2. **Everything in the vanilla flow is orchestrated through plain-Python dict rebuilding and object-graph walks**, re-executed every rollout step and every training step: `Node.forward` (system.py:48-59), `Problem.step`'s `{**input_dict, **output_dict}` dict merge (problem.py:204-209), and `Variable.forward`'s per-call NetworkX `in_edges()` graph walk (constraint.py:541-561) for every constraint. None of these alone is quadratic, but they multiply together as `epochs × batches × nsteps × RK-stages × MLP-layers × constraints`, and none of it is vectorized or JIT/compile-friendly.

Fix #1 first — it's a contained, mechanical change with the highest measured payoff. Fix #2 is an architecture-level cost (dict-of-tensors + graph-interpreter design chosen for readability/composability) that's harder to remove without a bigger refactor; mitigations below reduce it without changing the public API.

---

## 1. `System.forward`: O(n²) tensor growth via `torch.cat` (primary bottleneck)

**Where:** `src/neuromancer/system.py:261-277` (the rollout loop), using the helper at `system.py:234-246`:

```python
def cat(self, data3d, data2d):
    for k in data2d:
        if k not in data3d:
            data3d[k] = data2d[k][:, None, :]
        else:
            data3d[k] = torch.cat([data3d[k], data2d[k][:, None, :]], dim=1)  # <-- reallocates + copies everything so far
    return data3d

def forward(self, input_dict):
    ...
    for i in range(nsteps):
        for node in self.nodes:
            indata = {k: data[k][:, i] for k in node.input_keys}
            outdata = node(indata)
            data = self.cat(data, outdata)   # called once per node per step
    return data
```

Every call to `cat` allocates a brand-new tensor and copies **all previously accumulated timesteps** plus the one new step. Over `n` steps this is `1+2+...+n = O(n²)` element copies per output key, per node. This is the classic "grow a tensor in a loop with `torch.cat`" antipattern — the fix is to either preallocate the output buffer or collect into a Python list and call `torch.stack` once at the end (`stack` on a list is O(n) total, not O(n²)).

**Measured** (real `System` instance, 2 nodes — a linear policy and a linear dynamics model — batch=100, state dim=4, input dim=1; average of 5 forward passes):

| nsteps | current `System.forward` | collect-then-`stack`-once rewrite | speedup |
|---:|---:|---:|---:|
| 100  | 5.96 ms   | 2.14 ms  | 2.8x |
| 500  | 55.38 ms  | 10.73 ms | 5.2x |
| 2000 | 676.21 ms | 44.49 ms | 15.2x |

Note the scaling: going from nsteps=100 to nsteps=2000 (20x more steps) makes the current implementation **113x slower**, while the rewrite is only **21x slower** (i.e., linear, as expected). This is exactly the quadratic signature.

**Why it matters for the vanilla DPC flow specifically:** the building-control DPC example (`examples/domain_examples/DPC_building_control.ipynb`) trains with `nsteps=100`, then runs a **closed-loop evaluation rollout at `nsteps_test=2000`** — precisely the regime where this cost explodes. Even during training, at `nsteps=100 × epochs=200`, the `cat` overhead alone is ~200 × 5ms ≈ 1s wasted just on tensor reallocation, on top of whatever the actual model compute costs.

**Suggested fix:** rewrite `System.forward` (and `SystemPreview.forward`, which has the identical pattern at system.py:348-367) to collect each node's per-step output into a Python list keyed by output name, and call `torch.stack(list, dim=1)` once after the loop instead of `cat`-ing on every iteration. This is a mechanical, behavior-preserving change (verified above by reimplementing `forward` this way and confirming identical output shapes) — no change to the `Node`/graph API.

---

## 2. Pervasive Python-level dict/graph orchestration (compounding, architectural)

The library is built around passing `dict[str, Tensor]` through a graph of small Python objects (`Node`, `Problem`, `Variable`/`Constraint`) rather than passing plain tensors through compiled/vectorized code. This is a deliberate readability/composability tradeoff (symbolic constraint algebra via operator overloading, arbitrary DAG wiring), but it adds real, repeated interpreter overhead in the hot path:

- **`Node.forward`** (`system.py:48-59`) rebuilds an input list and output dict via list/dict comprehensions on every single node call — called `nsteps × len(nodes)` times per rollout.
- **`Problem.step`** (`problem.py:204-209`) does `input_dict = {**input_dict, **output_dict}` — a full dict copy — on every node call, once per training step:
  ```python
  for node in self.nodes:
      output_dict = node(input_dict)
      input_dict = {**input_dict, **output_dict}
  ```
- **`Problem.forward`** (`problem.py:197-202`) then rebuilds the *entire* output dict again with renamed keys (`f'{data["name"]}_{k}'`) every training step.
- **`Variable.forward` / `get_value`** (`constraint.py:541-561`) walks the constraint's expression DAG and, for every node in it, calls NetworkX's `self._g.in_edges(n)` to find its dependencies — **on every single evaluation of every constraint, every training step.** NetworkX edge lookups are not designed for tight inner loops; DPC problems typically carry several bound constraints (input/state limits), each of which is a small DAG walked this way every step.
- **`AggregateLoss.calculate_constraints`** (`loss.py:76-108`) repeats the same grow-a-Python-list-then-`torch.cat` pattern as `System.cat`, just over the (small, fixed) number of constraints rather than timesteps — same antipattern, smaller magnitude.

None of these is individually pathological — they're small dicts and small DAGs. What matters is that they multiply against the rollout: a single training step for the DPC building example does roughly

```
nsteps(100) × nodes-per-step(policy + dynamics + integrator RK-stages) × constraints × DAG-nodes-per-constraint
```

Python-level calls, none of them vectorized or torch.compile-friendly (dict access and NetworkX calls will graph-break under `torch.compile`). This is the reason the library, while algorithmically simple per-step, is slower than the flop count alone would suggest — it's dominated by interpreter overhead, not tensor math.

**Suggested mitigations (roughly by effort):**
- Cheapest: in `Problem.step`, use `input_dict.update(output_dict)` (mutate in place) instead of `{**input_dict, **output_dict}` (full copy) — same semantics, avoids reallocating the whole dict every node call.
- Medium: in `constraint.py`, precompute each `Variable`'s dependency list once at graph-construction time (`make_graph`, constraint.py:360-401 already builds `self._g`) instead of calling `self._g.in_edges(n)` inside `get_value` on every forward call — cache `{n: list(self._g.in_edges(n)) for n in ordered_nodes}` once and reuse it.
- Larger: batch the RK4/Euler integrator's sub-stage calls and the MLP's per-layer loop are both fine as-is (they're small and expected); the real win is reducing how many times the *outer* Python loop (rollout × constraints) round-trips through dict rebuilding, which points back at fixing `System.forward`'s structure (§1) as the highest-leverage change.

---

## 3. Secondary items (real, but lower priority for the vanilla flow)

- **`Trainer.train`: `deepcopy(self.model.state_dict())` on every improving epoch** (`trainer.py:223` init, `trainer.py:276` in the loop). For DPC-sized models (small MLPs) this is cheap in absolute terms, but with `epochs=200-1000` and loss improving frequently in early training, it's a repeated full-parameter clone that's easy to avoid — e.g., only keep a reference and copy once at the very end of training, or track just the epoch index of the best checkpoint and reload from disk via `ModelCheckpoint`-style saving instead of an in-memory deep copy every epoch.
- **`RK4` integration multiplies the per-timestep MLP cost by 4x** (`dynamics/integrators.py`) — this is inherent to the method (4 stage evaluations per step), not a bug, but worth knowing: switching a DPC dynamics model from RK4 to Euler is a 4x reduction in per-step neural-net evaluations if the accuracy tradeoff is acceptable.
- **Fancy indexing in `ode.py` parametrized ODEs** (e.g. `CSTR_Param.ode_equations`, `TwoTankParam`) uses `x[:, [0]]`-style list indexing, which copies rather than views. Called once per RK stage per timestep — minor per-call cost, but repeats `RK-stages × nsteps × epochs` times so it's non-zero. Switching to `x[:, 0:1]` slice indexing (a view, no copy) is a free, drop-in fix wherever it appears in the hot dynamics path.
- **`dataset.py`'s `DictDataset`/`SequenceDataset` `__getitem__` + default `collate_fn`** rebuild a per-sample dict every access. This is **not** currently a hot-loop cost for the vanilla flow because NeuroMANCER's SGD examples default to `batch_size = len(dataset)` (one big batch per epoch) — flagged only in case a user reduces `batch_size`, at which point this per-sample dict overhead runs once per minibatch per epoch instead of once per epoch.

## Confirmed non-issues (checked, ruled out)

- `blocks.MLP`/`ResMLP`'s default `linear_map=slim.Linear` is a thin wrapper around plain `torch.nn.Linear` (`slim/linear.py:104-119`) — no hidden SVD/orthogonal/spectral-parametrization cost on the default path. Those exotic `slim` maps exist but must be explicitly selected.
- KAN blocks (`modules/blocks.py:183-457`) are present but strictly opt-in; not reachable from the default MLP/ResMLP + `slim.Linear` construction used in the DPC examples.
- `PenaltyLoss.forward` (`loss.py:168-181`), the default loss used in vanilla DPC (as opposed to `AugmentedLagrangeLoss`), is comparatively lean — two dict merges and a sum. The cost is concentrated upstream in `calculate_objectives`/`calculate_constraints`.

---

## Recommended order of attack

1. **Rewrite `System.forward` / `SystemPreview.forward`** to collect-then-`stack` instead of repeated `cat`. Highest measured impact (15x+ at realistic closed-loop horizons), contained change, no API break.
2. **`input_dict.update(output_dict)` instead of dict-unpacking** in `Problem.step` — one-line change, removes a full dict copy per node per step.
3. **Cache constraint dependency edges** in `constraint.py` instead of calling NetworkX `in_edges()` per evaluation — removes a graph-library call from the innermost loop.
4. Profile after 1-3 with `torch.profiler` on an actual DPC training run to see how much headroom is left before chasing the smaller items (RK4 stage count, ODE indexing, `deepcopy` in `Trainer`) — those are real but individually small next to the `cat` fix.
