"""
Stage-2 fit: identify the StagedDXPlant from the Modelica RTU dataset.

Open-loop plant identification: we feed the *measured* cooling stage (converted
to an ordinal engagement vector), return air (= zone air), outdoor air, and
occupancy, and fit the plant's physical parameters to reproduce the measured
supply temperature, airflow, and fan/DX power. The staging controller is NOT
involved here — that closes the loop in Stage 3.

The plant is memoryless, so every timestep is independent: we stack the whole
split along the batch dimension and fit in one vectorized forward pass.

Run:
    python -m neuromancer.hvac.examples.fit_staged_rtu
"""
import torch

from neuromancer.hvac.building_components import StagedDXPlant, stage_to_engagement
from neuromancer.hvac.data.load import get_dataset

torch.manual_seed(0)

N_STAGES = 2
FIT_VARS = ["T_supply", "supply_airflow", "fan_power", "dx_power"]
LEARNABLE = {
    "dQ_stage", "dflow_stage", "dx_power_stage", "dfan_stage",
    "vent_flow", "fan_power_coeff", "dx_oat_coeff", "oa_fraction",
}


def _flatten(d):
    """[1, T, 1] -> [T, 1] (time as batch) for the memoryless plant."""
    return {k: v.reshape(-1, v.shape[-1]) for k, v in d.items() if torch.is_tensor(v)}


def _inputs(data):
    return dict(
        stage_engagement=stage_to_engagement(data["cooling_stage"], N_STAGES),
        T_return=data["T_zone"],     # single-zone RTU: return air == zone air
        T_outdoor=data["T_outdoor"],
        occupancy=data["occupancy"],
    )


def _r2(pred, meas, mask=None):
    if mask is not None:
        m = mask.bool().squeeze(-1)
        pred, meas = pred[m], meas[m]
    ss_res = ((pred - meas) ** 2).sum()
    ss_tot = ((meas - meas.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def main():
    train = _flatten(get_dataset("train", "5", "excited"))
    val = _flatten(get_dataset("val", "5", "excited"))

    plant = StagedDXPlant(n_stages=N_STAGES, learnable=LEARNABLE)

    # Per-variable scales (measured std on train) to balance the loss across units.
    scales = {v: train[v].std().clamp(min=1e-6) for v in FIT_VARS}

    # Supply temperature is only meaningful when the fan is on — with no airstream
    # the "supply temp" sensor is a memory/decay reading the memoryless plant
    # cannot (and shouldn't) reproduce. Mask T_supply to fan-on samples.
    fan_on = {"train": (train["supply_airflow"] > 1e-6).float(),
              "val": (val["supply_airflow"] > 1e-6).float()}
    print(f"fan-on fraction: train {fan_on['train'].mean():.2f}  val {fan_on['val'].mean():.2f}")

    def _term(out, data, v, mask):
        err = ((out[v] - data[v]) / scales[v]).pow(2)
        return (err * mask).sum() / mask.sum() if mask is not None else err.mean()

    opt = torch.optim.Adam(plant.parameters(), lr=0.05)
    x_train, x_val = _inputs(train), _inputs(val)

    plant.train()
    for step in range(2000):
        opt.zero_grad()
        out = plant.forward(**x_train)
        loss = sum(_term(out, train, v, fan_on["train"] if v == "T_supply" else None)
                   for v in FIT_VARS)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == 1999:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    plant.eval()
    with torch.no_grad():
        tr, va = plant.forward(**x_train), plant.forward(**x_val)

    print("\nFit quality (R^2):    train      val   (T_supply: fan-on only)")
    for v in FIT_VARS:
        m_tr = fan_on["train"] if v == "T_supply" else None
        m_va = fan_on["val"] if v == "T_supply" else None
        print(f"  {v:16s} {_r2(tr[v], train[v], m_tr):8.3f}  {_r2(va[v], val[v], m_va):8.3f}")

    print("\nLearned parameters:")
    print(f"  dQ_stage  [W]   incr: {plant.dQ_stage.tolist()}")
    print(f"  dflow     [kg/s] incr:{plant.dflow_stage.tolist()}")
    print(f"  dx_power  [W]   incr: {plant.dx_power_stage.tolist()}")
    print(f"  dfan      [W]   incr: {plant.dfan_stage.tolist()}")
    print(f"  vent_flow [kg/s]:     {plant.vent_flow.item():.3f}")
    print(f"  fan_coeff [W/(kg/s)^3]:{plant.fan_power_coeff.item():.1f}")
    print(f"  dx_oat_coeff [1/K]:   {plant.dx_oat_coeff.item():.4f}")
    print(f"  oa_fraction [-]:      {plant.oa_fraction.item():.3f}")

    return plant, (x_val, val, va)


if __name__ == "__main__":
    main()
