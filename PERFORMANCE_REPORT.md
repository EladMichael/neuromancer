# Performance work on the NeuroMANCER core

Branch: `perf/core-speedups`, forked from `features/preview_and_logging` @ `32699d1`.
Scope: the vanilla path used to build a policy/model and train it — `System` rollout,
`Problem`/`PenaltyLoss`/`Constraint` evaluation, and `Trainer`. KANs, neural operators and
other exotic blocks were left alone.

Everything below is measured, not estimated. Environment: macOS arm64, torch 2.6.0,
Python 3.10, CPU, single thread unless noted. Reproduce with `python benchmarks/bench_core.py`.

## Results

| benchmark (batch=100)              | before   | after    | speedup |
|------------------------------------|---------:|---------:|--------:|
| `System.forward`, nsteps=100       |   6.50ms |   4.80ms |   1.35x |
| `System.forward`, nsteps=500       |  37.90ms |  23.59ms |   1.61x |
| `System.forward`, nsteps=2000      | 235.87ms |  95.15ms |   2.48x |
| rollout + backward, nsteps=100     |  13.70ms |  10.19ms |   1.34x |
| rollout + backward, nsteps=500     |  92.52ms |  52.23ms |   1.77x |
| `SystemPreview.forward`, nsteps=500|  43.50ms |  29.29ms |   1.49x |
| loss/constraint evaluation         |   0.16ms |   0.14ms |   1.14x |
| **full DPC train step**            | **15.81ms** | **10.49ms** | **1.51x** |
| **5 epochs end-to-end**            | **228.91ms** | **163.78ms** | **1.40x** |

Peak memory of a rollout + backward (RSS growth, batch=100):

| nsteps | before   | after   |
|-------:|---------:|--------:|
|    100 |  13.6 MB | 13.1 MB |
|    500 |  39.6 MB | 36.6 MB |
|   2000 | 317.4 MB | 130.5 MB |
|   8000 | 1140.7 MB | 513.9 MB |

Plus a separate memory fix in `Trainer`: 60 epochs of DPC training retained **166 MB → 7.5 MB**.

## What changed

**1. The rollout is linear in the horizon instead of quadratic** (`src/neuromancer/system.py`).

This was the bottleneck report's item #1 and it was correct. `System.forward` grew its 3-d
output tensors with `torch.cat` once per node per step, so step *n* reallocated and copied
everything accumulated so far.

You were right to flag the "collect then append" concern — the naive version of that rewrite
does run into trouble, because reads and writes are interleaved *within* a step: at step `i` a
node reads `data[k][:, i]`, where index `i` may refer either to data the caller supplied or to
a value a node appended on step `i-1`. A rewrite that only collects outputs and stacks at the
end breaks feedback loops.

The version implemented keeps that aliasing intact: `init_buffers` unbinds every key the nodes
touch into a list of per-step tensors (views, no copy), the rollout appends to those lists, and
`torch.stack` runs once at the end. Reads stay index-based, so a node reading step `i` sees
exactly what it saw before, whether that value came from the caller or from an earlier step.

Autograd is fine with this — in fact it is strictly better. The old code built `n` intermediate
concatenated tensors and the graph held all of them, which is where the quadratic *memory*
came from; the new code holds `n` per-step tensors and one `stack` node.

`SystemPreview` needed slightly more care because its preview window clamps against the
*current* length of a key, which grows during the rollout for generated keys. The window index
math moved into `window_indices`, so the same padding logic serves both a 3-d tensor
(`get_mapped_data`, public API unchanged) and the per-step buffers (`get_mapped_steps`).

`System.cat` is kept — it is public API and tested — it is just no longer on the hot path.

**2. Dictionaries are no longer rebuilt per node and per loss term**
(`problem.py`, `loss.py`). `Problem.step` and `PenaltyLoss.forward` did
`{**input_dict, **output_dict}` on every node/term, copying the whole data dict each time.
They now copy the caller's dict once and accumulate in place, which preserves the "don't
mutate the caller's dict" contract for one copy instead of N.

**3. Constraint graphs are resolved once, not per evaluation** (`constraint.py`).
`Variable.get_value` called networkx `in_edges()` for every node of every constraint on every
forward pass. The expression graph is fixed once built, so arguments are resolved in
`make_graph`.

**Correction on the value of changes 2 and 3.** The commit message for `04705f5` credits these
with taking a train step from 11.39ms to 10.74ms. That was a single-shot before/after across
two process invocations, and it does not survive a proper measurement. Reverting each change
individually on the current code, round-robin, taking the min of 9 rounds:

| config                    | train step | cost of reverting |
|---------------------------|-----------:|------------------:|
| branch as committed       |   10.79ms  |                 — |
| revert rollout rewrite    |   15.72ms  |          +4.93ms  |
| revert changes 2 and 3    |   10.80ms  |          +0.01ms  |
| revert everything         |   15.71ms  |          +4.93ms  |

**Changes 2 and 3 are not measurable on this problem** — the whole train-step gain is change 1.
Change 3 is real but small where it lives: it is ~5% of constraint evaluation (and grows with
constraint count — measured at 2, 8 and 32 constraints), but constraint evaluation is only
~1.7% of a train step, so it vanishes into the noise. Change 2 saves a handful of dict copies
of an 8-key dict, i.e. sub-microsecond here; it would matter for a `Problem` with many nodes or
a data dict with many keys. Both are kept because they are strictly less work with identical
output, not because they showed up in a benchmark.

**4. `Trainer` no longer pins one autograd graph per epoch** (`trainer.py`). This one was not
in the bottleneck report and is a bug, not a slowdown: `loss_history` and `best_devloss` stored
the epoch's mean loss *with its `grad_fn` attached*, so every epoch's autograd graph stayed
reachable for the entire run — ~2.8 MB per epoch for a 100-step DPC rollout, so multiple GB
over a 1000-epoch run. History is now stored detached. Separately, the `deepcopy` of the state
dict flagged in the report is now a `detach().clone()` (1.8–3.4x faster, small in absolute terms).

## Correctness

- **Bit-identical output**, verified against a reference implementation of the original
  `cat` rollout: all four padding modes, several past/future windows, horizons 0/1/3/17,
  feedback loops, keys that are both supplied and overwritten, and `start_iter`.
- **Bit-identical gradients** (parameter grads after backward).
- **Bit-identical training**: a 5-epoch DPC run produces the same train and dev loss history
  to 8 decimals and the same final weight checksum on this branch and at `32699d1`.
- **`Problem.forward` output unchanged** under the dict-copy change: same 34 returned keys,
  every returned tensor bit-identical, and neither implementation leaks a key into the
  caller's dict (verified by evaluating one `Problem` object under both implementations).
- Test suite: **339 passed, 4 failed**. The 4 failures are device-placement tests that fail
  identically at the base commit (verified in a clean worktree) — unrelated to this work.
  23 new tests were added covering the rewrite. `tests/psl` is excluded from that count: it is
  a very slow hypothesis-driven emulator suite, and nothing under `src/neuromancer/psl` imports
  any of the five modules touched here.

## What I tried and rejected (with numbers)

I want to be clear about the JAX idea, because it is the one thing I did not build. After the
rollout fix, the framework is close to its floor: a hand-written PyTorch loop doing exactly the
same math with no `Node`/`System`/dict machinery takes **3.70ms** where `System.forward` takes
**4.63ms** (nsteps=100, batch=100). All of NeuroMANCER's abstraction now costs ~20%. The other
80% is PyTorch executing ~2,200 small tensor ops sequentially, and a rewrite in JAX would not
change that — it would change *where* the ops execute. A `lax.scan` rollout fuses the loop and
would remove per-op dispatch, which is the real ceiling here (~7 GFLOP/s achieved against a
CPU capable of several times that on this arithmetic). So the honest estimate is roughly 2–3x
on CPU for a complete parallel implementation of the core — much more on GPU, where dispatch
dominates far harder — in exchange for maintaining two implementations of every block, node,
constraint and integrator, and splitting the ecosystem. That is a bad trade at this ratio. The
same fusion argument is what makes `torch.compile` the cheaper thing to want, and:

- **`torch.compile` on the policy net**: 19.9s warmup, then **4.95ms → 11.38ms** (2.3x
  *slower*). The policy is called 100 times per rollout and guard evaluation per call swamps
  the tiny graph.
- **`torch.compile` on the whole `System`** (nsteps=20): **1.05ms → 3.36ms**, also slower.
- **`torch.jit.script` on `blocks.MLP`**: fails outright. `Block.forward(self, *inputs)` uses
  varargs, which TorchScript does not support. Worth knowing if scripting is ever wanted — it
  is an API-breaking change to fix, so I left it.
- **Flattening `slim.Linear` to call `F.linear` directly** (removing a nested `nn.Linear`
  dispatch per layer): **24.8µs → 25.2µs**, i.e. no gain. The saved dispatch is paid straight
  back in `nn.Module.__getattr__` lookups for `.weight`/`.bias`.
- **Making the pydot graph lazy.** `Problem` construction is 2.75ms, ~60% of it pydot. But
  `graph()` also assigns node names, runs the uniqueness assertions and computes
  `input_keys`/`output_keys`; deferring it changes when those fire. 2.75ms is not worth that
  risk unless you are building thousands of problems in a sweep.
- **Hoisting feed-forward nodes out of the loop** (evaluating nodes whose inputs are all static
  over all timesteps at once). Real win in principle, but it silently changes results for any
  callable that is not purely per-sample (`BatchNorm` being the obvious one), and the payoff on
  the DPC graph is one trivial node. Rejected as too clever for the return.

## The biggest lever left is not code

The core is op-dispatch bound, which means throughput improves sharply with batch size:

| batch | rollout time | per 1000 samples |
|------:|-------------:|-----------------:|
|    50 |      4.59ms  |          91.7ms  |
|   100 |      4.88ms  |          48.8ms  |
|   250 |      5.78ms  |          23.1ms  |
|  1000 |      8.05ms  |           8.1ms  |

10x the batch costs 1.65x the time. The DPC examples ship with `batch_size=100` against
`n_samples=1000`, which leaves about **6x of throughput on the table** — a full-batch epoch is
~8ms of rollout versus ~49ms as ten minibatches. This changes optimization dynamics (fewer,
larger steps per epoch), so it is your call, not something I changed in the examples. But if a
training run is too slow, this is the first knob to turn, ahead of anything in this report.

Thread count is *not* a knob: 1, 2, 4 and 8 threads all land within 4% of each other, because
the tensors are too small to parallelize.

## Where the time goes now (DPC train step, 10.49ms)

| | share |
|---|---:|
| backward pass | ~36% |
| forward tensor math | ~40% |
| `nn.Module.__call__` dispatch (~1,419 module calls per rollout) | ~16% |
| rollout bookkeeping (dicts, buffers, stacking) | ~5% |
| loss + constraints | ~3% |

The dispatch line is PyTorch's, not NeuroMANCER's: `blocks.MLP` costs 1.5–1.9x its
hand-written functional equivalent purely in `nn.Module` overhead, and every route out of that
(compile, script) measured worse or failed above.

## Two pre-existing issues found, not fixed

- **`SystemPreview(start_iter=N)` with feedback is broken**, and was before this work. Reads
  start at index `start_iter` but generated keys start filling at index 0, so any node reading
  another node's output raises `IndexError` unless every generated key is seeded with at least
  `start_iter+1` timesteps of data. Both old and new implementations fail identically; I
  verified this rather than silently changing it, since fixing it is a semantics decision.
- The 4 device-placement test failures noted above.

## Repo notes

`benchmarks/bench_core.py` is committed so these numbers can be re-measured:

```bash
python benchmarks/bench_core.py --json before.json
python benchmarks/bench_core.py --compare before.json
```

Getting the branch to run required installing `neuraloperator==1.0.2` (2.0.0 dropped `SFNO`,
which `modules/operators.py` imports), plus `ipython`, `pytest` and `hypothesis`, into `.venv`.
Nothing was upgraded or removed.
