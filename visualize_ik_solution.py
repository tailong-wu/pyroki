import json
import time
import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

def visualize_ik_solution(ee_action_file, joint_angles_file):
    # Load the action sequence from the JSON file
    with open(ee_action_file, 'r') as f:
        action_data = json.load(f)
    ee_action_seq = action_data['ee_action_seq']

    # Load the joint angle sequence from the JSON file
    with open(joint_angles_file, 'r') as f:
        joint_angle_data = json.load(f)
    joint_angle_seq = joint_angle_data['joint_angle_seq']

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)

    # Load robot
    urdf = load_robot_description("piper_description")
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Display the target trajectory
    for i, action in enumerate(ee_action_seq):
        pose = np.array(action[:7])
        position = tuple(pose[:3])
        wxyz = tuple([pose[6], pose[3], pose[4], pose[5]])
        server.scene.add_transform_controls(
            f"/target_pose_{i}", scale=0.1, position=position, wxyz=wxyz
        )

    # Animate the robot according to the IK solution
    while True:
        for joint_angles in joint_angle_seq:
            urdf_vis.update_cfg(np.array(joint_angles))
            time.sleep(0.1)

if __name__ == "__main__":
    visualize_ik_solution('action.json', 'joint_angles.json')