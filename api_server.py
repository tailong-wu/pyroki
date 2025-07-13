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
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# --- Pre-load robot model ---
urdf = load_robot_description("piper_description")
robot = pk.Robot.from_urdf(urdf)

# TCP偏移量（沿gripper_base局部Z轴方向的距离）
TCP_OFFSET_DISTANCE = 0.135283

# --- 改进的成本函数 ---
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

# --- 改进的IK求解器 ---
def solve_ik(
    robot: pk.Robot,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
    tcp_offset_distance: float = None,
) -> np.ndarray:
    """改进的IK求解器，使用gripper_base + 沿Z轴偏移作为真实TCP"""
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

def calculate_joint_angles(ee_action_seq: List[List[float]]) -> List[List[float]]:
    """改进的关节角度计算函数，使用精确的TCP位置"""
    joint_angle_seq = []

    for action in ee_action_seq:
        pose = np.array(action[:7])
        target_position = pose[:3]
        target_so3 = jaxlie.SO3(np.array([pose[6], pose[3], pose[4], pose[5]]))
        correction_so3 = jaxlie.SO3.from_x_radians(jnp.pi)
        corrected_so3 = target_so3 @ correction_so3
        target_wxyz = corrected_so3.wxyz

        solution = solve_ik(
            robot=robot,
            target_position=target_position,
            target_wxyz=target_wxyz,
        )
        joint_angle_seq.append(solution.tolist())
    return joint_angle_seq

# --- FastAPI Endpoint ---
class ActionRequest(BaseModel):
    ee_action_seq: List[List[float]]

@app.post("/solve-ik/")
def api_solve_ik(request: ActionRequest):
    joint_angle_seq = calculate_joint_angles(request.ee_action_seq)
    return {"joint_angle_seq": joint_angle_seq}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)