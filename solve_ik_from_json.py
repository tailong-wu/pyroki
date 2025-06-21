import json
import numpy as np
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
from jaxls import Cost, Var, VarValues

# Custom cost functions added directly to the script
@Cost.create_factory
def midpoint_cost(
    vals: VarValues,
    robot: pk.Robot,
    joint_var: Var[jax.Array],
    target_position: jax.Array,
    link1_index: int,
    link2_index: int,
    weight: float,
) -> jax.Array:
    """Computes the residual for matching the midpoint of two links to a target position."""
    joint_cfg = vals[joint_var]
    Ts_link_world = robot.forward_kinematics(joint_cfg)
    pos1 = jaxlie.SE3(Ts_link_world[..., link1_index, :]).translation()
    pos2 = jaxlie.SE3(Ts_link_world[..., link2_index, :]).translation()
    midpoint = (pos1 + pos2) / 2.0
    residual = midpoint - target_position
    return (residual * weight).flatten()

@Cost.create_factory
def orientation_cost(
    vals: VarValues,
    robot: pk.Robot,
    joint_var: Var[jax.Array],
    target_orientation: jaxlie.SO3,
    link_index: int,
    weight: float,
) -> jax.Array:
    """Computes the residual for matching a link's orientation to a target orientation."""
    joint_cfg = vals[joint_var]
    Ts_link_world = robot.forward_kinematics(joint_cfg)
    orientation_actual = jaxlie.SE3(Ts_link_world[..., link_index, :]).rotation()
    residual = (orientation_actual.inverse() @ target_orientation).log()
    return (residual * weight).flatten()

def solve_ik(
    robot: pk.Robot,
    midpoint_link_names: list[str],
    orientation_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """
    Solves the IK problem for a robot using a midpoint for position and another link for orientation.

    Args:
        robot: PyRoKi Robot.
        midpoint_link_names: List of two link names for the midpoint position target.
        orientation_link_name: String name of the link for the orientation target.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert len(midpoint_link_names) == 2
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    midpoint_link_indices = jnp.array([robot.links.names.index(name) for name in midpoint_link_names])
    orientation_link_index = robot.links.names.index(orientation_link_name)

    cfg = _solve_ik_jax(
        robot,
        midpoint_link_indices,
        orientation_link_index,
        jnp.array(target_wxyz),
        jnp.array(target_position),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return np.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    midpoint_link_indices: jax.Array,
    orientation_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    factors = [
        midpoint_cost(
            robot,
            joint_var,
            target_position,
            midpoint_link_indices[0],
            midpoint_link_indices[1],
            weight=50.0,
        ),
        orientation_cost(
            robot,
            joint_var,
            jaxlie.SO3(target_wxyz),
            orientation_link_index,
            weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]

def solve_ik_from_json(input_json_path, output_json_path):
    """Solves IK for a sequence of end-effector poses from a JSON file.

    Args:
        input_json_path (str): Path to the input JSON file with ee_action_seq.
        output_json_path (str): Path to the output JSON file to save joint angles.
    """

    # Load the action sequence from the JSON file
    with open(input_json_path, 'r') as f:
        action_data = json.load(f)
    ee_action_seq = action_data['ee_action_seq']

    # Load the robot description
    urdf = load_robot_description("piper_description")
    robot = pk.Robot.from_urdf(urdf)
    midpoint_link_names = ["link7", "link8"]
    orientation_link_name = "gripper_base"

    joint_angle_seq = []

    for action in ee_action_seq:
        # The last element is for the gripper, so we ignore it for IK
        pose = np.array(action[:7])
        target_position = pose[:3]
        # The quaternion in the file is (qx, qy, qz, qw), but jaxlie expects (qw, qx, qy, qz)
        target_so3 = jaxlie.SO3(np.array([pose[6], pose[3], pose[4], pose[5]]))

        # Apply a correction rotation (180 degrees around the x-axis)
        correction_so3 = jaxlie.SO3.from_x_radians(jnp.pi)
        corrected_so3 = target_so3 @ correction_so3

        target_wxyz = corrected_so3.wxyz

        # Solve IK
        solution = solve_ik(
            robot=robot,
            midpoint_link_names=midpoint_link_names,
            orientation_link_name=orientation_link_name,
            target_position=target_position,
            target_wxyz=target_wxyz,
        )
        joint_angle_seq.append(solution.tolist())

    # Save the joint angle sequence to the output JSON file
    with open(output_json_path, 'w') as f:
        json.dump({'joint_angle_seq': joint_angle_seq}, f, indent=4)

    print(f"Saved joint angle sequence to {output_json_path}")

if __name__ == "__main__":
    solve_ik_from_json('action.json', 'joint_angles.json')