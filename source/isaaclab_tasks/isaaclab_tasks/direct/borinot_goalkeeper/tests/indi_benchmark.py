# SPDX-License-Identifier: BSD-3-Clause
"""Time the controller alone at training scale; no simulator, learning, or cluster job."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=10000)
parser.add_argument("--device", default="cuda")
parser.add_argument("--ip", type=float, default=0.0)
parser.add_argument("--method", choices=("geom", "method_1", "method_2"), default="geom")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("benchmark_indi", Path(__file__).resolve().parents[3]/"controllers/borinot_indi.py")
indi = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = indi
spec.loader.exec_module(indi)
task = SimpleNamespace(indi=indi.IndiControllerCfg(rotor_spin_inertia_kg_m2=args.ip), indi_tilt_method=args.method, physics_hz=300,
    thrust_to_weight=2.1, kf=3.847e-6, km_over_kf=5.361e-8/3.847e-6,
    rotor_spin=(1., -1., 1., -1., -1., 1.), body_rate_scale=(2., 2., 1.5),
    robot_home_quat=(math.sqrt(.5), 0., 0., -math.sqrt(.5)), motor_time_constant_s=.04,
    hold_arm=False, kp=9., kd=.2, effort_lim_sim=2.7)
positions = torch.tensor(((0., -.185), (0., .185), (.1602147, .0925), (-.1602147, -.0925),
                          (.1602147, -.0925), (-.1602147, .0925)), device=args.device)
allocation = torch.stack((torch.ones(6, device=args.device), positions[:, 1]-.00244, -positions[:, 0],
                           torch.tensor(task.rotor_spin, device=args.device)*task.km_over_kf))
inertia = torch.diag(torch.tensor([.02724642, .02685468, .03980717], device=args.device))
ctrl = indi.IndiController(task, args.num_envs, allocation, inertia, 2.91, 2.91*9.81/6)
torch.manual_seed(42)
actions = torch.randn(args.num_envs, 9, device=args.device).clamp(-1, 1)*.1
quaternion = torch.tensor(task.robot_home_quat, device=args.device).expand(args.num_envs, -1)
rates = torch.zeros(args.num_envs, 3, device=args.device)
joints = torch.zeros(args.num_envs, 2, device=args.device)
samples = []
with torch.inference_mode():
    for step in range(25):
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        ctrl.step(actions, quaternion, rates, joints, joints)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        if step >= 5:
            samples.append(1000*(time.perf_counter()-start))
result = {"num_envs": args.num_envs, "device": args.device, "indi_tilt_method": args.method, "rotor_spin_inertia_kg_m2": args.ip,
    "allocation_iterations": task.indi.allocation_iterations, "nonlinear_iterations": task.indi.nonlinear_iterations,
    "physics_hz": task.physics_hz, "warmup_ticks": 5, "measured_ticks": len(samples),
    "samples_ms": samples, "median_ms": sorted(samples)[len(samples)//2], "mean_ms": sum(samples)/len(samples),
    "torch_version": torch.__version__, "scope": "controller only, eager, no physics or optimizer"}
if args.device.startswith('cuda'):
    result['gpu'] = torch.cuda.get_device_name()
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
