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

# TCP偏移量（沿gripper_base局部Z轴方向的距离）
TCP_OFFSET_DISTANCE = 0.135283

# 改进的成本函数
@Cost.create_factory
def tcp_position_cost(
    vals: VarValues,
    robot: pk.Robot,
    joint_var: Var[jax.Array],
    target_position: jax.Array,
    gripper_base_index: int,
    tcp_offset_distance: float,
    weight: float,
) -> jax.Array:
    """计算TCP位置与目标位置的残差（使用gripper_base + 沿Z轴偏移）"""
    joint_cfg = vals[joint_var]
    Ts_link_world = robot.forward_kinematics(joint_cfg)
    
    # 获取gripper_base的变换矩阵
    gripper_base_transform = jaxlie.SE3(Ts_link_world[..., gripper_base_index, :])
    
    # 计算真实TCP位置：gripper_base位置 + 沿局部Z轴方向的偏移
    local_z_offset = jnp.array([0.0, 0.0, tcp_offset_distance])
    tcp_position = gripper_base_transform.translation() + gripper_base_transform.rotation().apply(local_z_offset)
    
    residual = tcp_position - target_position
    return (residual * weight).flatten()

@Cost.create_factory
def tcp_orientation_cost(
    vals: VarValues,
    robot: pk.Robot,
    joint_var: Var[jax.Array],
    target_orientation: jaxlie.SO3,
    gripper_base_index: int,
    weight: float,
) -> jax.Array:
    """计算TCP方向与目标方向的残差（使用gripper_base方向）"""
    joint_cfg = vals[joint_var]
    Ts_link_world = robot.forward_kinematics(joint_cfg)
    orientation_actual = jaxlie.SE3(Ts_link_world[..., gripper_base_index, :]).rotation()
    residual = (orientation_actual.inverse() @ target_orientation).log()
    return (residual * weight).flatten()

def solve_ik(
    robot: pk.Robot,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
    tcp_offset_distance: float = None,
) -> np.ndarray:
    """
    改进的IK求解器，使用gripper_base + 沿Z轴偏移作为真实TCP

    Args:
        robot: PyRoKi Robot.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.
        tcp_offset_distance: float. TCP offset distance from gripper_base along Z-axis.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    if tcp_offset_distance is None:
        tcp_offset_distance = TCP_OFFSET_DISTANCE
    
    gripper_base_index = robot.links.names.index('gripper_base')

    cfg = _solve_ik_jax(
        robot,
        gripper_base_index,
        jnp.array(target_wxyz),
        jnp.array(target_position),
        tcp_offset_distance,
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return np.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    gripper_base_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    tcp_offset_distance: float,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    factors = [
        tcp_position_cost(
            robot,
            joint_var,
            target_position,
            gripper_base_index,
            tcp_offset_distance,
            weight=50.0,
        ),
        tcp_orientation_cost(
            robot,
            joint_var,
            jaxlie.SO3(target_wxyz),
            gripper_base_index,
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

    joint_angle_seq = []

    for action in ee_action_seq:
        # The last element is for the gripper, so we ignore it for IK
        pose = np.array(action[:7])
        target_position = pose[:3]
        # The quaternion in the file is (qx, qy, qz, qw), but jaxlie expects (qw, qx, qy, qz)
        target_so3 = jaxlie.SO3(np.array([pose[6], pose[3], pose[4], pose[5]]))
        # target_so3 = jaxlie.SO3(np.array([ pose[3], pose[4], pose[5],pose[6]]))

        # Apply a correction rotation (180 degrees around the x-axis)
        correction_so3 = jaxlie.SO3.from_x_radians(jnp.pi)
        corrected_so3 = target_so3 @ correction_so3

        target_wxyz = corrected_so3.wxyz

        # Solve IK using improved TCP method
        solution = solve_ik(
            robot=robot,
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