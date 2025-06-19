import json
import numpy as np
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls

def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return np.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
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
    target_link_name = "gripper_base"

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
            target_link_name=target_link_name,
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