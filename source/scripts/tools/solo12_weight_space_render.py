"""Kinematic replay of saved IsaacLab states. Never runs MuJoCo dynamics."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--experiment", required=True)
parser.add_argument("--run", default="final_clean")
args = parser.parse_args()
root = Path(args.experiment)
meta = json.loads((root/args.run/"provenance.json").read_text())
spec = json.loads((root/args.run/"candidates.json").read_text())
trace = np.load(root/args.run/"trace.npz")
states = trace["states"]
model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[3]/"mujoco/solo12.xml"))
model.geom_group[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"ground")] = 0
data = mujoco.MjData(model)
indices = [model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in meta["joint_names"]]
renderer = mujoco.Renderer(model,height=360,width=480)
option = mujoco.MjvOption()
option.geomgroup[3] = False
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",16)
labels = ["Original", "Vertical-mode edit", "Antiphase compromise", "Stronger alternating edit"]
steps = np.r_[np.arange(200,400,2),np.arange(700,900,2),np.arange(2700,2900,2),np.arange(3200,3400,2)]
with imageio.get_writer(root/"kinematic_comparison.mp4",fps=25,codec="libx264",quality=8) as writer:
    for sequence_index, step in enumerate(steps):
        images = []
        for group in range(4):
            row = states[step,group*meta["replicas"]]
            data.qpos[:3] = row[27:30]
            data.qpos[3:7] = row[30:34]
            data.qpos[indices] = row[:12]
            mujoco.mj_forward(model,data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = row[27:30] + np.array([0,0,0.05])
            camera.distance, camera.azimuth, camera.elevation = 1.35, 150, -15
            renderer.update_scene(data,camera=camera,scene_option=option)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,0,480,51),fill=(15,20,28))
            draw.text((8,3),labels[group],font=font,fill="white")
            command = trace["observations"][step,group*meta["replicas"],18]
            draw.text((8,27),f"t={step*.02:.2f} s | vx command={command:+.1f} m/s",font=font,fill="white")
            images.append(np.array(frame))
        merged = np.concatenate((np.concatenate(images[:2],axis=1),np.concatenate(images[2:],axis=1)),axis=0)
        if sequence_index == 40:
            imageio.imwrite(root/"kinematic_comparison.png",merged)
        writer.append_data(merged)
renderer.close()
print(root/"kinematic_comparison.mp4")
