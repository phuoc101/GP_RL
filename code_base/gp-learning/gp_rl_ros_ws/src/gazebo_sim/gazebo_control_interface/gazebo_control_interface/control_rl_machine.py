"""
Teemu Mökkönen, Tampere University

This python file contains interface for controlling real machine manipulator using gaussion process modeĺ

Controllable joints:

    velocity commands:
        - front_right_wheel_joint
        - front_left_wheel_joint
        - back_right_wheel_joint
        - back_left_wheel_joint
        - center_link

Positions are controlled by actions server which executes the action to move the joint
to the wanted position.

velocity commands are at the moment set as group:
    - This means that when message is sent to /velocity_controller/commands topic
    - all joints mentioned for the controller receive the speed that is defined in the
    command
"""

from math import atan
from loguru import logger
from rclpy.duration import Duration
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from pathlib import Path
from rclpy.time import Time
from ament_index_python import get_package_share_path

from gp_rl.cfg.configs import get_gp_train_config
from gp_rl.models.GPModel import GPModel
from gp_rl.utils.data_loading import load_data
from gp_rl.utils.torch_utils import to_gpu, get_tensor
from gp_rl.utils.rl_utils import calc_realization_mean

import torch
import numpy as np
import rclpy
import sys
import os

# Constants for the system
GEAR_MOVE = 1
GEAR_FREE = 0
GEAR_MOVE_BACKWARD = -1
GEAR = 0
STEERING = 1
GAS = 2
BOOM = 0
BOOM_LOW_LIM = -0.68
BOOM_HIGH_LIM = 0.68
BUCKET_LOW_LIM = -0.98
BUCKET_HIGH_LIM = 0.98
BUCKET = 1
TELESCOPE = 2
TELESCOPE_LOW_LIM = 0.02
TELESCOPE_UPPER_LIM = 0.68
# Constant for controller timeout
TIMEOUT = 0.1

dir_path = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(get_package_share_path("gazebo_control_interface")))



class SteeringActionClient(Node):
    """
    This class handles the communation with the avant model. It will take in the
    commands from high level nodes and parses them to match the simulation
    parameters
    """

    def __init__(self):
        super().__init__("gazebo_joint_controller")

        # ros args
        self.declare_parameters(
            namespace="",
            parameters=[
                ("gain_gas", 20),
                ("gain_steering", 2.5),
                ("gain_boom", 2),
                ("gain_bucket", 2),
                ("tf", 25),
                ("dt", 0.1),
                ("state_dim", 1),
                ("control_dim", 1),
                ("target_state", [0]),
                ("force_train_gp", False),
                ("logger_verbose", "DEBUG"),
                ("path_to_training_data", "../data/boom_trial_6_10hz.pkl"),
                ("path_to_model_data", str(dir_path) + "/../results/gp/"),
                ("controller_data", str(dir_path) + "/../results/controller/"),
            ],
        )

        self.get_logger().info(
            "Route to training data {}".format(
                self.get_parameter("path_to_training_data")
                .get_parameter_value()
                .string_value
            )
        )
        self.get_logger().info(
            "Route to model data {}".format(
                self.get_parameter("path_to_model_data")
                .get_parameter_value()
                .string_value
            )
        )
        self.get_logger().info(
            "Route to controller data {}".format(
                self.get_parameter("controller_data").get_parameter_value().string_value
            )
        )

        # gains for control

        self.gas_gain = self.get_parameter("gain_gas").value
        self.gain_steering = self.get_parameter("gain_steering").value
        self.gain_boom = self.get_parameter("gain_boom").value
        self.gain_bucket = self.get_parameter("gain_bucket").value

        # model opts

        logger.remove()
        logger.add(
            sys.stderr,
            level=self.get_parameter("logger_verbose")
            .get_parameter_value()
            .string_value,
        )
        self.state_dim = self.get_parameter("state_dim").value
        self.control_dim = self.get_parameter("control_dim").value

        self.model, self.gpmodel_bucket, self.gpmodel_telescope = self.init_model()
        (
            self.controller,
            self.controller_bucket,
            self.controller_telescope,
        ) = self.init_controller()

        # ros2 subscibers and publishers

        self.manipulator_state = self.create_subscription(
            JointState, "bag_joint_states", self.manipulator_state_callback, 10
        )

        self.manipulator_speed_publisher = self.create_publisher(
            JointState, "manipulator_commands", 10
        )  # TODO: this to jointstate for real machine!

        self.state_publisher = self.create_publisher(
            JointState, "/states", 10
        )  # for debugging

        self.pose_topic = self.create_subscription(
            JointState, "boom_pose", self.wanted_pos_callback, 10
        )

        self.prev_pose = None
        self.prev_time = None
        self.prev_target_time = None  # to monitor if target msgs are being published
        self.logger = 0

        # msg definations
        self.state_msg_out = JointState()
        self.state_msg_out.position.append(0.0)
        self.state_msg_out.position.append(0.0)
        self.state_msg_out.velocity.append(0.0)
        self.state_msg_out.velocity.append(0.0)

        # states to keep after in the joints
        self.steer = 0.0
        self.boom = 2.0
        self.telescope = 1.0
        self.bucket = 1.0

        self.bucket_pose = 0.0
        self.boom_pose = 0.0
        self.telescope_pose = 0.0
        self.resolver_pos = 0.0

        self.bucket_vel = 0.0
        self.boom_vel = 0.0
        self.telescope_vel = 0.0
        self.resolver_vel = 0.0

        # model dimensions
        self.center_wheel_dist = 0.6
        self.dist_between_wheels = 0.6414

        self.prev_msg = []
        self.prev_vel = 0.0
        self.prev_command = 0.0

    def init_model(self):
        """
        initialize gp model
        """

        gpmodel_boom_cfg = get_gp_train_config()
        gpmodel_boom_cfg["joint"] = "boom"
        gpmodel_boom = GPModel(**gpmodel_boom_cfg)
        gpmodel_boom.initialize_model(
            path_model=self.get_parameter("path_to_model_data")
            .get_parameter_value()
            .string_value
            + "GPmodel_boom.pkl",
            # uncomment the lines below for retraining
            path_train_data=self.get_parameter("path_to_training_data")
            .get_parameter_value()
            .string_value,
            force_train=self.get_parameter("force_train_gp").value,
        )

        gpmodel_bucket_cfg = get_gp_train_config()
        gpmodel_bucket_cfg["joint"] = "bucket"
        gpmodel_bucket = GPModel(**gpmodel_bucket_cfg)
        gpmodel_bucket.initialize_model(
            path_model=self.get_parameter("path_to_model_data")
            .get_parameter_value()
            .string_value
            + "GPmodel_bucket.pkl",
            # uncomment the lines below for retraining
            path_train_data=self.get_parameter("path_to_training_data")
            .get_parameter_value()
            .string_value,
            force_train=self.get_parameter("force_train_gp").value,
        )

        gpmodel_telescope_cfg = get_gp_train_config()
        gpmodel_telescope_cfg["joint"] = "telescope"
        gpmodel_telescope = GPModel(**gpmodel_telescope_cfg)
        gpmodel_telescope.initialize_model(
            path_model=self.get_parameter("path_to_model_data")
            .get_parameter_value()
            .string_value
            + "GPmodel_telescope.pkl",
            # uncomment the lines below for retraining
            path_train_data=self.get_parameter("path_to_training_data")
            .get_parameter_value()
            .string_value,
            force_train=self.get_parameter("force_train_gp").value,
        )

        return gpmodel_boom, gpmodel_bucket, gpmodel_telescope

    def init_controller(self):
        # boom
        controller_data = load_data(
            self.get_parameter("controller_data").get_parameter_value().string_value + "boom/"
            + "_all_boom.pkl"
        )
        controller = self.get_best_controller(controller_data)
        to_gpu(controller)

        # bucket
        controller_data = load_data(
            self.get_parameter("controller_data").get_parameter_value().string_value + "bucket/"
            + "_all_bucket.pkl"
        )
        controller_bucket = self.get_best_controller(controller_data)
        to_gpu(controller)

        # telescope
        controller_data = load_data(
            self.get_parameter("controller_data").get_parameter_value().string_value + "telescope/"
            + "_all_telescope.pkl"
        )
        controller_telescope = self.get_best_controller(controller_data)
        to_gpu(controller)

        logger.info("controller model loaded")
        return controller, controller_bucket, controller_telescope

    def get_best_controller(self, controller_data):
        """
        chooses the controller with lowest loss
        """

        losses = []
        for controller in controller_data["all_optimizer_data"]:
            losses.append(controller["optimInfo"]["loss"][-1])
        best_controller_data = controller_data["all_optimizer_data"][np.argmin(losses)]
        return best_controller_data["controller_final"]

    def manipulator_state_callback(self, msg):
        """
        get the state from real machine for prediction

        args:
            msg JoinState: position and velocities for each moving joint in the machine
        """
        self.boom_pose = msg.position[2]
        self.bucket_pose = msg.position[3]
        self.telescope_pose = msg.position[7]

    def wanted_pos_callback(self, target_msg):
        """
        moves the robot boom manipulator to the wanted state according to the given
        command

        args:
            msg Jointstate: pos [1]
        """

        # boom
        init_state = np.array(self.boom_pose)
        target_state = get_tensor(np.array(target_msg.position[BOOM]))

        M_instance = (
            calc_realization_mean(
                gp_model=self.model,
                controller=self.controller,
                state_dim=self.state_dim,
                control_dim=self.control_dim,
                dt=0.01,
                init_state=init_state,
                target_state=target_state,
            )
            .cpu()
            .numpy()
        )

        valve_boom_cmd = M_instance[:, -1, 0].item()

        # bucket

        init_state = np.array(self.bucket_pose)
        target_state = get_tensor(np.array(target_msg.position[BUCKET]))

        M_instance = (
            calc_realization_mean(
                gp_model=self.gpmodel_bucket,
                controller=self.controller_bucket,
                state_dim=self.state_dim,
                control_dim=self.control_dim,
                dt=0.01,
                init_state=init_state,
                target_state=target_state,
            )
            .cpu()
            .numpy()
        )

        valve_bucket_cmd = M_instance[:, -1, 0].item()

        # telescope

        init_state = np.array(self.telescope_pose)
        target_state = get_tensor(np.array(target_msg.position[TELESCOPE]))

        M_instance = (
            calc_realization_mean(
                gp_model=self.gpmodel_telescope,
                controller=self.controller_telescope,
                state_dim=self.state_dim,
                control_dim=self.control_dim,
                dt=0.01,
                init_state=init_state,
                target_state=target_state,
            )
            .cpu()
            .numpy()
        )

        valve_telescope_cmd = M_instance[:, -1, 0].item()

        # send control

        mani_speed_msg_out = JointState()
        mani_speed_msg_out.velocity = [
            valve_boom_cmd,
            valve_bucket_cmd,
            valve_telescope_cmd,
        ]
        self.manipulator_speed_publisher.publish(mani_speed_msg_out)

        # keep track of msg time, if more than timeout then stop manipulator
        self.prev_target_time = self.get_clock().now().nanoseconds / 1e9


def main(args=None):

    rclpy.init()
    # Turn on the ROS2 node and make it run in the loop
    action_client = SteeringActionClient()
    # saction_client.update_joints()
    rclpy.spin(action_client)


if __name__ == "__main__":
    main()
