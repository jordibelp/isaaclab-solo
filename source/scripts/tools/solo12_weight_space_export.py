"""Export inference-only checkpoints and verify exact ordinary-RSL inference."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    root = Path(args.experiment)
    source = Path(args.checkpoint)
    torch.set_num_threads(1)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    # Load only pure tensor-edit functions, without launching Isaac Sim.
    script = Path(__file__).with_name("solo12_weight_space_edit.py")
    tree = ast.parse(script.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("first_layer_edit", "final_layer_edit", "edited_state")]
    scope = {"torch": torch, "np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(script), "exec"), scope)
    edit = scope["edited_state"]
    data = np.load(root/"baseline/trace.npz")
    center = torch.from_numpy(data["action_center"])
    unchanged = edit(state, {"name":"original"}, center)
    assert all(torch.equal(unchanged[key],state[key]) for key in state)
    x = torch.from_numpy(data["observations"].reshape(-1,57)[::7].copy())
    policy_cfg = dict(json.loads((root/"wandb_config.json").read_text())["agent_cfg"]["policy"])
    policy_cfg.pop("class_name")
    with contextlib.redirect_stdout(io.StringIO()):
        model = ActorCritic(TensorDict({"policy": torch.zeros(1,57), "critic": torch.zeros(1,60)}, batch_size=[1]),
                            {"policy":["policy"],"critic":["critic"]},12,**policy_cfg)
    model.eval()
    selected = json.loads((root/"validation_candidates.json").read_text())
    dest = root/"checkpoints"
    dest.mkdir(exist_ok=True)
    receipts = []
    for spec in selected:
        if spec["name"] == "original":
            continue
        new = edit(state,spec,center)
        for key in state:
            assert new[key].shape == state[key].shape
            assert torch.isfinite(new[key]).all()
            if not key.startswith("actor."):
                assert torch.equal(new[key],state[key]), key
        model.load_state_dict(new,strict=True)
        with torch.inference_mode():
            actual = model.act_inference(TensorDict({"policy":x},batch_size=[len(x)]))
            ref = (x-state["actor_obs_normalizer._mean"])/(state["actor_obs_normalizer._std"]+.01)
            for layer in (0,2,4):
                ref = torch.nn.functional.elu(ref@new[f"actor.{layer}.weight"].T+new[f"actor.{layer}.bias"])
            ref = ref@new["actor.6.weight"].T+new["actor.6.bias"]
        torch.testing.assert_close(actual,ref,atol=2e-5,rtol=2e-5)
        edited_count = sum(int(torch.count_nonzero(new[k]!=state[k])) for k in new if k.startswith("actor."))
        actor_count = sum(v.numel() for k,v in state.items() if k.startswith("actor."))
        path = dest/f"1104_z7ft0vsj_{spec['name']}_EXPERIMENTAL.pt"
        info = {"source_checkpoint":str(source),"source_sha256":original_hash,"source_iteration":checkpoint["iter"],
                "spec":spec,"edited_actor_scalars":edited_count,"total_actor_scalars":actor_count,
                "normalizers_and_critic_unchanged":True,"status":"experimental; walking goal not achieved; not hardware-validated"}
        torch.save({"model_state_dict":new,"iter":checkpoint["iter"],"infos":checkpoint.get("infos"),
                    "inference_only":True,"weight_space_edit":info},path)
        reloaded = torch.load(path,map_location="cpu",weights_only=False)
        model.load_state_dict(reloaded["model_state_dict"],strict=True)
        assert "optimizer_state_dict" not in reloaded
        receipts.append({**info,"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                         "rsl_max_error":float((actual-ref).abs().max()),"common_input_samples":len(x)})
    hashes = set()
    cases = 0
    for run in ("screen","projection","circuit","refine","transfer","imu"):
        for spec in json.loads((root/run/"candidates.json").read_text()):
            new = edit(state,spec,center)
            digest = hashlib.sha256()
            for layer in (0,2,4,6):
                for kind in ("weight","bias"):
                    digest.update(new[f"actor.{layer}.{kind}"].numpy().tobytes())
            hashes.add(digest.hexdigest())
            cases += 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    report = {"exports":receipts,"screened_policy_cases":cases,"unique_actor_states_including_original":len(hashes),
              "original_checkpoint_unchanged":True,"no_op_state_bit_exact":True,"optimizer_updates":0}
    (root/"export_verification.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="exports"}))
    for r in receipts:
        print(r["path"],r["edited_actor_scalars"],r["rsl_max_error"])


if __name__ == "__main__":
    main()
