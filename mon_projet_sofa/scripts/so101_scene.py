import os
import sys
import numpy as np
import Sofa, Sofa.Core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SO101_DIR = os.path.join(BASE_DIR, "..", "SO101")   # adaptez si besoin
TORQUES_FILE = os.path.join(BASE_DIR, "torques.txt")

sys.path.insert(0, BASE_DIR)
from so101_kinematics import (link_transform_matrices, matrix_to_quat_xyzw,
                                make_T, LINK_MESH, JOINT_ORDER)
from so101_dynamics import dynamics_step

VIS_T = {link: make_T(vis_xyz, vis_rpy) for link, (_, vis_xyz, vis_rpy) in LINK_MESH.items()}


def mesh_world_poses(angles):
    link_mats = link_transform_matrices(angles)
    poses = {}
    for link, T_link in link_mats.items():
        T_mesh = T_link @ VIS_T[link]
        poses[link] = list(T_mesh[:3, 3]) + matrix_to_quat_xyzw(T_mesh[:3, :3])
    return poses


def read_torques_file():
    """Lit 6 couples (N.m) dans torques.txt, dans l'ordre shoulder_pan, shoulder_lift,
    elbow_flex, wrist_flex, wrist_roll, gripper."""
    try:
        with open(TORQUES_FILE, "r") as f:
            values = [float(v) for v in f.read().split()]
        if len(values) == len(JOINT_ORDER):
            return values
    except (FileNotFoundError, ValueError):
        pass
    return [0.0] * len(JOINT_ORDER)


class SO101Controller(Sofa.Core.Controller):
    def __init__(self, mo_by_link, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.listening = True
        self.mo_by_link = mo_by_link
        self.angles = [0.0] * len(JOINT_ORDER)
        self.angvels = [0.0] * len(JOINT_ORDER)
        self.apply_visual(self.angles)

    def apply_visual(self, angles):
        poses = mesh_world_poses(angles)
        for link, mo in self.mo_by_link.items():
            with mo.position.writeable() as pos:
                pos[0] = poses[link]

    def onAnimateBeginEvent(self, event):
        dt = self.getContext().getDt()
        torques = read_torques_file()
        self.angles, self.angvels = dynamics_step(self.angles, self.angvels, torques, dt)
        self.apply_visual(self.angles)


def createScene(root):
    root.gravity = [0, 0, -9.81]   # la gravité est gérée par notre moteur physique Python,
    root.dt = 0.02                  # ce champ SOFA n'est là que pour cohérence de la scène

    for p in ["Sofa.Component.StateContainer", "Sofa.Component.IO.Mesh",
              "Sofa.Component.Mapping.NonLinear", "Sofa.GL.Component.Rendering3D",
              "Sofa.Component.Visual", "Sofa.Component.AnimationLoop",
              "Sofa.GL.Component.Shader"]:
        root.addObject("RequiredPlugin", pluginName=p)

    root.addObject("DefaultAnimationLoop")
    root.addObject("DefaultVisualManagerLoop")
    root.addObject("VisualStyle", displayFlags="showVisualModels")

    root.addObject("InteractiveCamera", name="camera",
                    position=[0.4, 0.3, 0.3], lookAt=[0, 0, 0.05])

    root.addObject("LightManager")
    root.addObject("DirectionalLight", direction=[0, -1, -0.5])
    root.addObject("DirectionalLight", direction=[0.3, 0.5, -0.8], color=[0.3, 0.3, 0.35, 1])

    poses0 = mesh_world_poses([0.0] * len(JOINT_ORDER))
    mo_by_link = {}

    for link, (mesh_rel, _, _) in LINK_MESH.items():
        node = root.addChild(link)
        mo = node.addObject("MechanicalObject", template="Rigid3d", name="dof",
                             position=[poses0[link]], showObject=False)
        mo_by_link[link] = mo

        if mesh_rel is not None:
            mesh_path = os.path.join(SO101_DIR, mesh_rel)
            visu = node.addChild("visu")
            visu.addObject("MeshSTLLoader", name="loader", filename=mesh_path)
            visu.addObject("OglModel", name="visual", src="@loader",
                            color=[0.15, 0.15, 0.17, 1])
            visu.addObject("RigidMapping", template="Rigid3d,Vec3d",
                            input="@../dof", output="@visual")

    root.addObject(SO101Controller(mo_by_link, name="SO101Controller"))
    return root


def main():
    import Sofa.Gui
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init("main", "glfw")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 800)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()


if __name__ == "__main__":
    main()