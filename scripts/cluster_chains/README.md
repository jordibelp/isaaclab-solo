# Run a command chain in an existing Slurm job

Pass one Bash script to the allocation reuse helper. Bash runs its commands in
order and waits for each to finish before starting the next.

## Make a chain from pasted commands

From the cluster's `~/IsaacLab` directory, open a new file:

```bash
nano run_chain.sh
```

Paste this header, then your commands. Keep one complete command per line.
Keep existing quotes around run names and list arguments.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Paste the first command here.
# Paste the second command here.
# Continue with the remaining commands. Do not add & at the end.
```

Save with Ctrl+O, Enter, then exit with Ctrl+X. Check the shell syntax:

```bash
bash -n run_chain.sh
```

When the allocation is running and has no active training process, inject the
whole chain. Replace `4659` with the intended job ID:

```bash
./reuse_isaaclab_gpu_job.sh --jobid 4659 -- bash -e -o pipefail run_chain.sh
```

No `chmod` is needed because Bash reads the file. The file must exist on the
cluster, at the path passed to Bash. Relative paths are resolved from the
helper's working directory, which defaults to `~/IsaacLab`.

## Prepared nine-run example

`borinot_min_time_agility.sh` contains the nine supplied training commands in
their original order. Run it after the file is available on the cluster:

```bash
./reuse_isaaclab_gpu_job.sh --jobid 4659 -- bash -e -o pipefail scripts/cluster_chains/borinot_min_time_agility.sh
```

The order is `thrust_and_torque_control`, `wrench`, `indi`, `model_wrench`,
`model_wrench_indi`, `model_wrench_yawless_indi`, `cbrt`, `cbrt_indi`, then
`model_wrench_yawless`. All supplied arguments are preserved, including the two
`env.use_terminal_success_bonus` settings. Their last value is `True`.

## What to expect

- Commands run sequentially inside one Slurm step. This is not a new allocation.
- A nonzero command exit stops the chain. A failed pipeline also stops it.
- This is a static list, not a queue service. Save edits before starting it.
- Rerunning the script starts at its first command; there is no automatic resume.
- The helper uses `srun --overlap`. It does not wait for existing training to
  finish. Only inject into an allocation whose training process has stopped.
- The job's existing memory, GPU resources and remaining time limit still apply.
  Starting the chain does not extend the allocation.
- The command stays attached to your terminal. Use an existing `tmux` or `screen`
  session if it must survive a disconnected SSH session.

## Why a bare paste does not work

This sends only `command_1` through the helper:

```bash
./reuse_isaaclab_gpu_job.sh --jobid 4659 -- command_1
command_2
```

Your current shell runs `command_2`, outside the injected step. An unquoted `;`
or `&&` has the same problem. Save the commands in a file to keep the complete
chain inside the allocation and avoid nested quoting.
