#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot",
    type=str,
    default="franka.yml",
    help="Robot configuration to download",
)
parser.add_argument("--save_usd", default=False, action="store_true")
args = parser.parse_args()

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": args.save_usd})

# Third Party
import omni.usd
import omni.kit.commands
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Sdf, Usd

ISAAC_SIM_45 = False
try:
    # Third Party
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    try:
        # Third Party
        from omni.importer.urdf import _urdf  # isaac sim 2023.1 - 4.0
    except ImportError:
        # Third Party
        from isaacsim.asset.importer.urdf import _urdf  # isaac sim 4.5+

        ISAAC_SIM_45 = True

# CuRobo
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    join_path,
    load_yaml,
)


def import_robot_from_urdf(my_world: World, robot_config):
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 10000
    import_config.default_position_drive_damping = 100
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    full_path = join_path(get_assets_path(), robot_config["robot_cfg"]["kinematics"]["urdf_path"])
    base_link_name = robot_config["robot_cfg"]["kinematics"]["base_link"]
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)

    physics_usd_path = None
    if ISAAC_SIM_45:
        dest_path = join_path(robot_path, get_filename(filename, remove_extension=True) + "_temp.usd")
        _, imported_root = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=f"{robot_path}/{filename}",
            import_config=import_config,
            dest_path=dest_path,
        )
        physics_usd_path = join_path(
            join_path(get_path_of_dir(dest_path), "configuration"),
            get_filename(dest_path, remove_extension=True) + "_physics.usd",
        )

        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + imported_root,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:
        urdf_interface = _urdf.acquire_urdf_interface()
        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
        robot_path = urdf_interface.import_robot(robot_path, filename, imported_robot, import_config, "")

    robot = my_world.scene.add(Robot(prim_path=robot_path + "/" + base_link_name, name="robot"))
    return robot, robot_path, physics_usd_path


def patch_link_visual_refs_for_lula(stage: Usd.Stage, robot_path: str, physics_usd_path: str):
    if physics_usd_path is None or (not os.path.exists(physics_usd_path)):
        return

    physics_stage = Usd.Stage.Open(physics_usd_path)
    if physics_stage is None:
        return

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(robot_path + "/"):
            continue
        prim_name = prim.GetName()
        if prim_name not in ["visuals", "collisions"]:
            continue
        link_name = prim.GetParent().GetName()
        source_root = "/visuals" if prim_name == "visuals" else "/colliders"
        source_prim_path = f"{source_root}/{link_name}"

        if not physics_stage.GetPrimAtPath(source_prim_path).IsValid():
            continue

        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(physics_usd_path, source_prim_path)


def export_lula_compatible_usd_from_physics(physics_usd_path: str, save_path: str):
    physics_stage = Usd.Stage.Open(physics_usd_path)
    if physics_stage is None:
        raise RuntimeError(f"Unable to open physics USD: {physics_usd_path}")

    flat_path = join_path(
        get_path_of_dir(save_path), get_filename(save_path, remove_extension=True) + "_flat_tmp.usda"
    )
    physics_stage.Flatten().Export(flat_path)
    stage = Usd.Stage.Open(flat_path)

    root = stage.GetDefaultPrim()
    root_path = str(root.GetPath())
    layer = stage.GetRootLayer()

    vis_root = stage.GetPrimAtPath("/visuals")
    if vis_root.IsValid():
        for child in vis_root.GetChildren():
            src = str(child.GetPath())
            dst = f"{root_path}/{child.GetName()}/visuals"
            try:
                Sdf.CopySpec(layer, src, layer, dst)
            except Exception:
                pass

    col_root = stage.GetPrimAtPath("/colliders")
    if col_root.IsValid():
        for child in col_root.GetChildren():
            if "CollisionGroup" in child.GetName():
                continue
            src = str(child.GetPath())
            dst = f"{root_path}/{child.GetName()}/collisions"
            try:
                Sdf.CopySpec(layer, src, layer, dst)
            except Exception:
                pass

    layer.Export(save_path)
    try:
        os.remove(flat_path)
    except OSError:
        pass


def resolve_robot_config_path(robot_arg: str) -> str:
    if os.path.isabs(robot_arg) and os.path.exists(robot_arg):
        return robot_arg
    if os.path.exists(robot_arg):
        return robot_arg

    config_relative_path = join_path(get_robot_configs_path(), robot_arg)
    if os.path.exists(config_relative_path):
        return config_relative_path

    raise FileNotFoundError(
        f"Could not find robot config file from --robot={robot_arg}. "
        f"Tried: {robot_arg} and {config_relative_path}"
    )


def save_usd():
    my_world = World(stage_units_in_meters=1.0)

    # Get the urdf file path
    robot_config = load_yaml(resolve_robot_config_path(args.robot))
    robot, robot_path, physics_usd_path = import_robot_from_urdf(my_world, robot_config)
    if ISAAC_SIM_45:
        patch_link_visual_refs_for_lula(my_world.stage, robot_path, physics_usd_path)
    # robot.disable_gravity()

    my_world.reset()

    save_path = join_path(get_assets_path(), robot_config["robot_cfg"]["kinematics"]["usd_path"])
    if ISAAC_SIM_45 and physics_usd_path is not None and os.path.exists(physics_usd_path):
        export_lula_compatible_usd_from_physics(physics_usd_path, save_path)
    else:
        usd_help = UsdHelper()
        usd_help.load_stage(my_world.stage)
        usd_help.write_stage_to_file(save_path, True)
    print("Wrote usd file to " + save_path)
    simulation_app.close()


def debug_usd():
    my_world = World(stage_units_in_meters=1.0)

    # Get the urdf file path
    robot_config = load_yaml(resolve_robot_config_path(args.robot))
    default_config = robot_config["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
    j_names = robot_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
    robot, robot_path, physics_usd_path = import_robot_from_urdf(my_world, robot_config)
    if ISAAC_SIM_45:
        patch_link_visual_refs_for_lula(my_world.stage, robot_path, physics_usd_path)
    # robot.disable_gravity()
    i = 0

    articulation_controller = robot.get_articulation_controller()
    my_world.reset()

    while simulation_app.is_running():
        my_world.step(render=True)
        if i == 0:
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            i += 1
        # if dof_n is not None:
        #    dof_i = [robot.get_dof_index(x) for x in j_names]
        #
        #    robot.set_joint_positions(default_config, dof_i)
        if robot.is_valid():
            art_action = ArticulationAction(default_config, joint_indices=idx_list)
            articulation_controller.apply_action(art_action)
    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)
    save_path = join_path(get_assets_path(), robot_config["robot_cfg"]["kinematics"]["usd_path"])
    usd_help.write_stage_to_file(save_path, True)
    simulation_app.close()


if __name__ == "__main__":
    if args.save_usd:
        save_usd()
    else:
        debug_usd()
