"""Analyze saved weight-edit rollouts without starting a simulator."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(folder):
    folder = Path(folder)
    data = np.load(folder / "trace.npz")
    meta = json.loads((folder / "provenance.json").read_text())
    specs = json.loads((folder / "candidates.json").read_text())
    dt = meta["physics_dt"]
    p = data["physics"]
    steps, envs = data["actions"].shape[:2]
    substeps = round(meta["policy_dt"] / dt)
    assert len(p) == steps * substeps
    names = meta["feet"]
    front = [next(i for i, name in enumerate(names) if name.startswith(prefix)) for prefix in ("FL", "FR")]
    rear = [next(i for i, name in enumerate(names) if name.startswith(prefix)) for prefix in ("RL", "RR")]
    contact = p[:, :, :4] > 0.5
    height = p[:, :, 4:8]
    # Contact sensor and articulation order can differ. Classification uses contact labels
    # plus base height; foot clearance is reported only when its own order was saved.
    raised = ~np.any(contact[:, :, front], axis=-1) & (p[:, :, 11] > 0.3)
    support = contact[:, :, rear].sum(-1)
    command = np.repeat(data["observations"][:, :, 18], substeps, axis=0)
    times = np.arange(len(p)) * dt
    # Exclude initial rise and 0.8 s after each requested command change from steady scores.
    changes = np.r_[True, np.any(command[1:] != command[:-1], axis=1)]
    last_change = np.maximum.accumulate(np.where(changes, times, -1e6))
    mask = (times >= 2.0) & (times - last_change >= 0.8)
    good = np.broadcast_to(mask[:, None], command.shape)
    result = []
    for group, spec in enumerate(specs):
        sl = slice(group * meta["replicas"], (group+1)*meta["replicas"])
        ok = good[:, sl]
        r = raised[:, sl]
        s = support[:, sl]
        cmd = command[:, sl]
        vx = p[:, sl, 8]
        err = vx - cmd
        moving = ok & (np.abs(cmd) > 0.05)
        denom = max(int(ok.sum()), 1)
        term = data["dones"][:, sl, 0]
        switches = []
        for env in range(meta["replicas"]):
            side = np.where(contact[:, sl, rear[0]][:, env], 1, -1)
            eligible = (ok[:, env] & r[:, env] & (s[:, env] == 1))
            signal = np.where(eligible, side, 0)
            bounds = np.r_[0, np.flatnonzero(signal[1:] != signal[:-1])+1, len(signal)]
            stable = [(a,b,int(signal[a])) for a,b in zip(bounds[:-1],bounds[1:])
                      if signal[a] and (b-a)*dt >= 0.03]
            n = 0
            for previous, current in zip(stable[:-1],stable[1:]):
                if previous[2] != current[2] and (current[0]-previous[1])*dt <= 0.3:
                    n += 1
            switches.append(n)
        row = {"name": spec.get("name", str(group)), "spec": spec,
            "terminations": int(term.sum()), "ever_failed_fraction": float(term.any(0).mean()),
            "front_raised_fraction": float(r[ok].mean()),
            "flight_fraction": float(((s==0) & r & ok).sum()/denom),
            "single_support_fraction": float(((s==1) & r & ok).sum()/denom),
            "double_support_fraction": float(((s==2) & r & ok).sum()/denom),
            "support_switches_per_s": float(np.mean(switches) / (mask.sum()*dt)),
            "tracking_rmse_m_s": float(np.sqrt(np.mean(err[moving]**2))),
            "mean_abs_vertical_speed_m_s": float(np.abs(p[:,sl,10][ok]).mean()),
            "lateral_rmse_m_s": float(np.sqrt(np.mean(p[:,sl,9][ok]**2))),
            "max_abs_raw_action": float(np.max(np.abs(data["actions"][:,sl]))),
            "max_abs_joint_speed_rad_s": float(np.max(np.abs(data["states"][:,sl,12:24]))),
            "target_step_rms_rad": float(np.sqrt(np.mean(np.diff(data["actions"][:,sl],axis=0)**2))*0.25),
        }
        if "feet_height_order" in meta:
            front_h = [next(i for i, name in enumerate(meta["feet_height_order"]) if name.startswith(prefix)) for prefix in ("FL", "FR")]
            row["mean_front_clearance_m"] = float(np.min(height[:,sl,front_h],axis=-1)[ok].mean())
        for direction, selector in (("positive", cmd > 0.05), ("negative", cmd < -0.05)):
            selected = moving & selector
            row[direction+"_tracking_rmse_m_s"] = float(np.sqrt(np.mean(err[selected]**2)))
            row[direction+"_actual_vx_m_s"] = float(vx[selected].mean())
            row[direction+"_command_vx_m_s"] = float(cmd[selected].mean())
        # Screening score only; never interpret it as a success certificate.
        row["screen_score"] = (row["single_support_fraction"] + 0.15*min(row["support_switches_per_s"],4)
            - 1.5*row["flight_fraction"] - 2*row["tracking_rmse_m_s"] - 2*row["lateral_rmse_m_s"]
            - 3*row["ever_failed_fraction"] - 2*(1-row["front_raised_fraction"]))
        result.append(row)
    (folder/"metrics.json").write_text(json.dumps(result,indent=2))
    for row in sorted(result,key=lambda x:x["screen_score"],reverse=True)[:15]:
        print(json.dumps(row))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    args = parser.parse_args()
    analyze(args.folder)
