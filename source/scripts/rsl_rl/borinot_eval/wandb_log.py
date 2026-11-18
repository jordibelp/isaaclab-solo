# SPDX-License-Identifier: BSD-3-Clause
"""Mirror metrics (not checkpoints or local paths) to a verified private W&B project."""
from .spec import read_json, write_json

PROJECT = "borinot-eval"


def publish(out, spec, runs, mode, entity):
    import wandb

    if mode == "online":
        from wandb.apis.public.api import gql

        api = wandb.Api(timeout=15)
        entity = entity or api.default_entity
        result = api.client.execute(gql("""
            query EvalProject($name: String!, $entity: String!) {
                project(name: $name, entityName: $entity) { name access }
            }
        """), {"name": PROJECT, "entity": entity})
        if (result.get("project") or {}).get("access") != "PRIVATE":
            raise RuntimeError(f"W&B reporting requires a verified private project {entity}/{PROJECT}. "
                               "The full evaluation is still available locally.")
    receipt_path = out / "wandb-runs.json"
    receipts = read_json(receipt_path) if receipt_path.exists() else []
    done = {(row["policy"], row["condition"]) for row in receipts if row.get("mode") == mode}
    for r in runs:
        record, data = r["record"], r["data"]
        if (record["policy"], record["condition"]) in done:
            continue
        columns = list(dict.fromkeys(key for row in data["episodes"] for key in row))
        run = wandb.init(project=PROJECT, entity=entity, group=out.name, job_type="evaluation",
                         name=f"{record['policy']} · {record['condition']}",
                         mode=mode, dir=str(out),
                         settings=wandb.Settings(disable_git=True, disable_code=True, save_code=False,
                                                 x_disable_meta=True, x_disable_stats=True),
                         config={k: record[k] for k in ("policy", "condition", "checkpoint_name", "training_run",
                                                        "policy_hz", "physics_hz", "control_mode", "stage", "seed")}
                         | {"task": spec["task"],
                            "perturbations": next(c for c in spec["conditions"] if c["label"] == record["condition"])})
        try:
            scalars = {f"eval/{k}": v for k, v in data["summary"].items() if isinstance(v, (int, float))}
            if "outcome_counts" in data["summary"]:
                scalars.update({f"outcomes/{k}": v for k, v in data["summary"]["outcome_counts"].items()})
            run.log(scalars | {"eval/trials": wandb.Table(columns=columns,
                    data=[[row.get(key) for key in columns] for row in data["episodes"]])})
            run.summary.update(scalars)
            receipt = {"id": run.id, "url": run.url, "mode": mode, "policy": record["policy"], "condition": record["condition"]}
        finally:
            run.finish()
        receipts.append(receipt)
        write_json(receipt_path, receipts)
    print(f"W&B: {sum(row.get('mode') == mode for row in receipts)} runs saved to {entity or 'default entity'}/{PROJECT} ({mode})", flush=True)
