"""
Benchmarks for the vanilla NeuroMANCER training path.

Covers the pieces exercised by a differentiable predictive control (DPC) problem:
the `System` rollout, the `SystemPreview` rollout, constraint/`Variable` evaluation,
a full `Problem` train step, and an end-to-end `Trainer` run.

Usage::

    python benchmarks/bench_core.py                     # run everything, print a table
    python benchmarks/bench_core.py --json base.json    # also dump results as json
    python benchmarks/bench_core.py --compare base.json # print speedup against a dump
"""
import argparse
import json
import time

import torch
import torch.nn as nn

from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.problem import Problem
from neuromancer.system import Node, System, SystemPreview
from neuromancer.trainer import Trainer

# Single-zone building model dimensions, matching examples/domain_examples/building_control.py
NX, NU, ND, NY = 4, 1, 3, 1


def timeit(func, repeats, warmup=1):
    """
    Run func repeats times and return the mean wall-clock seconds per call.
    """
    for _ in range(warmup):
        func()
    start = time.perf_counter()
    for _ in range(repeats):
        func()
    return (time.perf_counter() - start) / repeats


def build_cl_system(nsteps, system_class=System, input_map=None):
    """
    Closed-loop building system: disturbance observer -> neural policy -> linear SSM -> output map.

    :param nsteps: (int) rollout horizon
    :param system_class: (type) System or SystemPreview
    :param input_map: (dict) optional preview window config for the policy node
    :return: (System) closed-loop system
    """
    torch.manual_seed(0)
    A = torch.eye(NX) + 0.01 * torch.randn(NX, NX)
    B = 0.1 * torch.randn(NX, NU)
    C = torch.randn(NY, NX)
    E = 0.1 * torch.randn(NX, ND)

    state_model = Node(lambda x, u, d: x @ A.T + u @ B.T + d @ E.T,
                       ['x', 'u', 'd'], ['x'], name='SSM')
    output_model = Node(lambda x: x @ C.T, ['x'], ['y'], name='y=Cx')
    dist_obsv = Node(lambda d: d[:, :2], ['d'], ['d_obsv'], name='dist_obsv')

    window = 1 if input_map is None else input_map['ymin']['past'] + 1 + input_map['ymin']['future']
    net = blocks.MLP_bounds(insize=NY + 2 * NY * window + 2, outsize=NU,
                            hsizes=[32, 32], min=torch.tensor([0.]), max=torch.tensor([5000.]))
    policy = Node(net, ['y', 'ymin', 'ymax', 'd_obsv'], ['u'],
                  name='policy', input_map=input_map)

    return system_class([dist_obsv, policy, state_model, output_model],
                        nsteps=nsteps, name='cl_system')


def make_data(nsteps, batch, name='train'):
    """
    One batch of DPC training data: initial state, comfort bounds and disturbance trajectories.
    """
    torch.manual_seed(1)
    x0 = torch.randn(batch, 1, NX)
    ymin = 18.0 + torch.rand(batch, 1, NY) * torch.ones(batch, nsteps + 1, NY)
    return {'x': x0,
            'y': x0[:, :, [0]],
            'ymin': ymin,
            'ymax': ymin + 2.0,
            'd': torch.randn(batch, nsteps + 1, ND),
            'name': name}


def build_dpc_problem(nsteps):
    """
    The full DPC problem: closed-loop system, energy/smoothness objectives and comfort constraints.
    """
    cl_system = build_cl_system(nsteps)
    y, u = variable('y'), variable('u')
    ymin, ymax = variable('ymin'), variable('ymax')
    objectives = [0.01 * (u == 0.0), 0.1 * (u[:, :-1, :] - u[:, 1:, :] == 0.0)]
    constraints = [50.0 * (y > ymin), 50.0 * (y < ymax)]
    for term, name in zip(objectives + constraints, ['action', 'du', 'y_min', 'y_max']):
        term.update_name(name)
    return Problem([cl_system], PenaltyLoss(objectives, constraints))


def bench_system_forward(results, batch=100):
    """Rollout only, no autograd -- isolates the rollout bookkeeping from the model math."""
    for nsteps in [100, 500, 2000]:
        system = build_cl_system(nsteps)
        data = make_data(nsteps, batch)
        with torch.no_grad():
            results[f'system_forward_nsteps{nsteps}'] = timeit(
                lambda: system(data), repeats=3)


def bench_system_forward_backward(results, batch=100):
    """Rollout plus backward -- what a training step actually pays."""
    for nsteps in [100, 500]:
        system = build_cl_system(nsteps)
        data = make_data(nsteps, batch)

        def step():
            out = system(data)
            out['u'].square().mean().backward()

        results[f'system_fwd_bwd_nsteps{nsteps}'] = timeit(step, repeats=3)


def bench_preview_forward(results, batch=100, nsteps=500):
    """SystemPreview rollout where the policy sees a past/future window of the comfort bounds."""
    input_map = {'ymin': {'past': 2, 'future': 2}, 'ymax': {'past': 2, 'future': 2}}
    system = build_cl_system(nsteps, system_class=SystemPreview, input_map=input_map)
    data = make_data(nsteps, batch)
    with torch.no_grad():
        results[f'preview_forward_nsteps{nsteps}'] = timeit(lambda: system(data), repeats=3)


def bench_constraints(results, batch=100, nsteps=100):
    """Constraint + objective evaluation alone (the Variable graph interpreter)."""
    y, u = variable('y'), variable('u')
    ymin, ymax = variable('ymin'), variable('ymax')
    loss = PenaltyLoss([0.01 * (u == 0.0), 0.1 * (u[:, :-1, :] - u[:, 1:, :] == 0.0)],
                       [50.0 * (y > ymin), 50.0 * (y < ymax)])
    data = make_data(nsteps, batch)
    data['u'] = torch.randn(batch, nsteps, NU)
    data['y'] = torch.randn(batch, nsteps + 1, NY)
    results['loss_eval'] = timeit(lambda: loss(dict(data)), repeats=50)


def bench_train_step(results, batch=100, nsteps=100):
    """One full Problem forward + backward + optimizer step."""
    problem = build_dpc_problem(nsteps)
    data = make_data(nsteps, batch)
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)

    def step():
        out = problem(data)
        optimizer.zero_grad()
        out['train_loss'].backward()
        optimizer.step()

    results['train_step'] = timeit(step, repeats=5)


def bench_training(results, batch=100, nsteps=100, n_samples=200, epochs=5):
    """End-to-end Trainer loop over train and dev splits."""
    problem = build_dpc_problem(nsteps)
    train_data = DictDataset({k: v for k, v in make_data(nsteps, n_samples).items() if k != 'name'},
                             name='train')
    dev_data = DictDataset({k: v for k, v in make_data(nsteps, n_samples).items() if k != 'name'},
                           name='dev')
    loaders = [torch.utils.data.DataLoader(d, batch_size=batch, collate_fn=d.collate_fn,
                                           shuffle=False) for d in [train_data, dev_data]]

    def run():
        trainer = Trainer(problem, *loaders,
                          optimizer=torch.optim.AdamW(problem.parameters(), lr=0.001),
                          epochs=epochs, patience=epochs, warmup=epochs, epoch_verbose=epochs + 1)
        trainer.train()

    results[f'training_{epochs}epochs'] = timeit(run, repeats=1, warmup=0)


BENCHMARKS = {
    'system_forward': bench_system_forward,
    'system_fwd_bwd': bench_system_forward_backward,
    'preview_forward': bench_preview_forward,
    'constraints': bench_constraints,
    'train_step': bench_train_step,
    'training': bench_training,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', help='write results to this json file')
    parser.add_argument('--compare', help='json file of a previous run to compare against')
    parser.add_argument('--only', nargs='+', choices=list(BENCHMARKS),
                        help='run only these benchmark groups')
    args = parser.parse_args()

    torch.set_num_threads(1)
    results = {}
    for name in (args.only or BENCHMARKS):
        BENCHMARKS[name](results)

    baseline = json.load(open(args.compare)) if args.compare else {}
    header = f"{'benchmark':<32}{'ms':>10}" + (f"{'baseline ms':>14}{'speedup':>10}" if baseline else '')
    print(header)
    print('-' * len(header))
    for name, seconds in results.items():
        line = f'{name:<32}{seconds * 1e3:>10.2f}'
        if name in baseline:
            line += f'{baseline[name] * 1e3:>14.2f}{baseline[name] / seconds:>9.2f}x'
        print(line)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
