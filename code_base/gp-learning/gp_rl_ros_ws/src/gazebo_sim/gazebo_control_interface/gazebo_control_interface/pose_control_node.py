"""
Teemu Mökkönen, Tampere University

This python file contains needed interfaces to send commands to the ros2 gazebo
controller which moves the joints according to the given commands in the gazebo
environment. This node handles the control for the manipulator joints and for the
moving the model. Under are specified the controllable joints and what is the method to
control them.

Controllable joints:

    velocity commands:
        - front_right_wheel_joint
        - front_left_wheel_joint
        - back_right_wheel_joint
        - back_left_wheel_joint
        - center_link

    position / Velocity commands:
        - boom_angle
        - telescope_length
        - fork_angle

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
                (
                    "path_to_training_data",
                    "/home/teemu/data/boom_trial_6_10hz.pkl",
                ),  # you need to use launch file to redeclare theses values to suit your workspace
                ("path_to_model_data", "/home/teemu/results/gp/GPmodel.pkl"),
                ("controller_data", "/home/teemu/results/controller/_all.pkl"),
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
        # action client is responsible for sending goals to the
        # joint_trajectory_controller which executes the joint command to the simulator model

        # prediction_model = run_model_single_input()

        self.speed_publisher = self.create_publisher(
            Float64MultiArray, "motion_controller/commands", 10
        )
        self.manipulator_speed_publisher = self.create_publisher(
            Float64MultiArray, "/manipulator_controller/commands", 10
        )
        self.state_publisher = self.create_publisher(JointState, "/states", 10)
        self.resolver_pub_ = self.create_publisher(JointState, "resolver", 10)
        self.telescope_pub_ = self.create_publisher(JointState, "telescope", 10)

        self.joint_state_sub = self.create_subscription(
            JointState, "joint_states", self.state_callback, 10
        )
        self.command_sub_ = self.create_subscription(
            JointState, "motion_commands", self.command_callback, 10
        )
        self.manipulator_vel_sub = self.create_subscription(
            JointState, "manipulator_commands", self.manipulator_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/boom_pose", self.wanted_pos_callback, 10
        )
        self.prev_pose = None
        self.prev_time = None
        self.prev_target_time = None  # to monitor if target msgs are being published
        self.logger = 0

        # msg definations
        self.state_msg_out = JointState()
        # values for boom
        self.state_msg_out.position.append(0.0)
        self.state_msg_out.position.append(0.0)
        self.state_msg_out.velocity.append(0.0)
        self.state_msg_out.velocity.append(0.0)
        # values for bucket
        self.state_msg_out.position.append(0.0)
        self.state_msg_out.position.append(0.0)
        self.state_msg_out.velocity.append(0.0)
        self.state_msg_out.velocity.append(0.0)
        # values for telescope
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
            self.get_parameter("controller_data").get_parameter_value().string_value
            + "_all_boom.pkl"
        )
        controller = self.get_best_controller(controller_data)
        to_gpu(controller)

        # bucket
        controller_data = load_data(
            self.get_parameter("controller_data").get_parameter_value().string_value
            + "_all_bucket.pkl"
        )
        controller_bucket = self.get_best_controller(controller_data)
        to_gpu(controller)

        # telescope
        controller_data = load_data(
            self.get_parameter("controller_data").get_parameter_value().string_value
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

    def update_joints(self):
        """
        Action server client node which sends the given command to the joint handler.

        ards:
            angle (float): wanted angle for the avants center joint
        """
        goal_msg = FollowJointTrajectory.Goal()

        joint_names = ["boom_angle", "telescope_length", "fork_angle"]
        points = []
        goal_data = JointTrajectoryPoint()
        goal_data.time_from_start = Duration(seconds=1, nanoseconds=0).to_msg()
        goal_data.positions = [self.boom, self.telescope, self.bucket]

        # Don't send new goal state if the initial state has not changed
        if goal_data.positions != self.prev_msg:
            points.append(goal_data)
            self.get_logger().info(str(points))
            goal_msg.goal_time_tolerance = Duration(seconds=1, nanoseconds=0).to_msg()
            goal_msg.trajectory.joint_names = joint_names
            goal_msg.trajectory.points = points

            self.action_client_.wait_for_server()
            self._send_goal_future = self.action_client_.send_goal_async(
                goal_msg, feedback_callback=self.feedback_callback
            )

            self._send_goal_future.add_done_callback(self.goal_response_callback)

            self.prev_msg = goal_data.positions

    def command_callback(self, msg):
        """
        Acts as interface for parsing avant motion commands and passes them to
        corresponding parties to make the simulation move.

        args:
            msg (JointState msg): JointState message containing abstact variables gas,
            steering and gear
        """

        gear = msg.position[GEAR]
        steer = msg.position[STEERING]
        gas = msg.position[GAS]

        if gear == GEAR_MOVE:
            self.gas = gas

        elif gear == GEAR_MOVE_BACKWARD:
            self.gas = -gas

        elif gear == GEAR_FREE:
            self.gas = 0.0

        self.publish_vel(self.gas, steer, gear)

    def state_callback(self, msg):
        """
        Reads the joint_state message and parses from there the same
        data that real avant would publish too using sensors.

        args:
            msg (ros2 JointState): JointState message type
        """

        # Form message for the center resolver
        resolver_msg = JointState()
        resolver_msg.header = msg.header

        # Form message for the telescope lenght
        telescope_msg = JointState()
        telescope_msg.header = msg.header

        curr_time = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        for i, name in enumerate(msg.name):
            if name == "center_link":
                resolver_msg.name.append("resolver")
                resolver_msg.velocity.append(msg.velocity[i])
                resolver_msg.position.append(msg.position[i])
                self.resolver_pos = msg.position[i]

                self.resolver_pub_.publish(resolver_msg)

            if name == "telescope_length":
                telescope_msg.name.append("telescope")
                telescope_msg.position.append(msg.position[i])
                self.telescope_pose = msg.position[i]
                self.telescope_vel = msg.velocity[i]
                self.telescope_pub_.publish(telescope_msg)

            if name == "boom_angle":
                self.boom_pose = msg.position[i]
                if self.prev_pose is not None and self.prev_time is not None:
                    self.boom_vel = (self.boom_pose - self.prev_pose) / (
                        curr_time - self.prev_time
                    )  # msg.velocity[i] * 10# scale the vel to match given vel
                # self.boom_vel = msg.velocity[i]
                self.prev_pose = self.boom_pose
                self.prev_time = curr_time

            if name == "fork_angle":
                self.bucket_pose = msg.position[i]
                self.bucket_vel = msg.velocity[i]

        if self.prev_target_time is not None:
            real_curr_time = self.get_clock().now().nanoseconds / 1e9
            if real_curr_time - self.prev_target_time > TIMEOUT:
                self.prev_target_time = None
                stop_command_msg = Float64MultiArray()
                stop_command_msg.data = [0.0, 0.0, 0.0]
                self.manipulator_speed_publisher.publish(stop_command_msg)

    def publish_vel(self, gas, steer, gear):
        """
        Publishes the wanted speed for the rear tires as "gas" value. This might need to
        be scaled to fit the simulation later on!

        args:
            gas (float): Gas is abstract command from the avant motion control interface
        """

        # do not send more messages to server than needed to avoid crowding it
        if self.prev_vel != gas or self.steer != steer:
            msg = Float64MultiArray()
            gas = gas * self.gas_gain  # scale the gas to match the real value
            steer = (
                steer * self.gain_steering
            )  # scale steer avoid over shooting with control
            gas_right = gas
            gas_left = gas
            if self.resolver_pos < -0.20:
                gas_right = 0.2 + gas * (
                    self.center_wheel_dist
                    / (atan(np.pi - (abs(self.resolver_pos)) / 2))
                    + (self.dist_between_wheels / 2)
                )
                gas_left = (
                    0.2
                    + gas
                    * (
                        self.center_wheel_dist
                        / atan(np.pi - abs(self.resolver_pos))
                        / 2
                    )
                    - (self.dist_between_wheels / 2)
                )

            if self.resolver_pos > 0.20:
                gas_left = 0.2 + gas * (
                    self.center_wheel_dist
                    / (atan(np.pi - (abs(self.resolver_pos)) / 2))
                    + (self.dist_between_wheels / 2)
                )
                gas_right = (
                    0.2
                    + gas
                    * (
                        self.center_wheel_dist
                        / atan(np.pi - abs(self.resolver_pos))
                        / 2
                    )
                    - (self.dist_between_wheels / 2)
                )

            # check if the central joint is able to move anymore
            if (self.resolver_pos > 0.52 and steer > 0) or (
                self.resolver_pos < -0.52 and steer < 0
            ):
                steer = 0.0

            # Publish speed for all four tires and for frame steering
            msg.data = [gas_right, gas_left, gas_right, gas_left, steer]
            self.speed_publisher.publish(msg)
            self.prev_vel = gas

    def manipulator_callback(self, msg):
        """
        Published wanted values for the manipulator as velocity commands given to the
        controller. The incoming message should contain the corresponding valve in the
        velocity part of the message in the following order:

        [boom, bucket, telescope]

        These are then read through, scaled and send to the gazebo joint contoller

        args:
            msg (ROS2 JointState): JointState message containting the wanted values for
            the control the manipulator
        """
        msg_out = Float64MultiArray()

        vel_boom = msg.velocity[BOOM]  # * self.gain_boom
        vel_tel = msg.velocity[TELESCOPE]
        vel_bucket = msg.velocity[BUCKET]  # * self.gain_bucket
        command = torch.from_numpy(np.asarray([self.boom_pose, vel_boom])).float()

        # boom
        model_input = torch.reshape(command, (1, 2))
        model_input = model_input.to(self.model.device, self.model.dtype)
        boom_prediction = self.model.predict(model_input)

        # bucket
        model_input = torch.reshape(
            torch.from_numpy(np.asarray([self.bucket_pose, vel_bucket])).float(), (1, 2)
        )
        model_input = model_input.to(self.model.device, self.model.dtype)
        bucket_prediction = self.gpmodel_bucket.predict(model_input)

        # telescope

        model_input = torch.reshape(
            torch.from_numpy(np.asarray([self.telescope_pose, vel_tel])).float(), (1, 2)
        )
        model_input = model_input.to(self.model.device, self.model.dtype)
        tel_prediction = self.gpmodel_bucket.predict(model_input)

        # check if the manipulator joint can move to the given direction to avoid joint
        # conflict with velocities
        if (self.telescope_pose <= TELESCOPE_LOW_LIM and vel_tel < 0.0) or (
            self.telescope_pose >= TELESCOPE_UPPER_LIM and vel_tel > 0.0
        ):
            vel_tel = 0.0

        if (self.boom_pose < BOOM_LOW_LIM and vel_boom < 0.0) or (
            self.boom_pose >= BOOM_HIGH_LIM and vel_boom > 0.0
        ):
            vel_boom = 0.0

        if (self.bucket_pose < BUCKET_LOW_LIM and vel_bucket < 0.0) or (
            self.bucket_pose >= BUCKET_HIGH_LIM and vel_bucket > 0.0
        ):
            vel_bucket = 0.0

        vel = (boom_prediction.mean[0][0]) / (self.get_parameter("dt").value)
        vel_bucket = (bucket_prediction.mean[0][0]) / (self.get_parameter("dt").value)
        vel_telescope = (tel_prediction.mean[0][0]) / (self.get_parameter("dt").value)

        # Send velocity to the ros2 controller which will move the joints

        # boom
        self.state_msg_out.position[0] = self.boom_pose
        self.state_msg_out.position[1] = boom_prediction.mean[0][0]
        self.state_msg_out.velocity[0] = self.boom_vel
        self.state_msg_out.velocity[1] = vel.item()

        # bucket
        self.state_msg_out.position[0] = self.bucket_pose
        self.state_msg_out.position[1] = bucket_prediction.mean[0][0]
        self.state_msg_out.velocity[0] = self.bucket_vel
        self.state_msg_out.velocity[1] = vel_bucket.item()

        # telescope

        self.state_msg_out.position[0] = self.telescope_pose
        self.state_msg_out.position[1] = tel_prediction.mean[0][0]
        self.state_msg_out.velocity[0] = self.telescope_vel
        self.state_msg_out.velocity[1] = vel_telescope.item()

        self.state_publisher.publish(self.state_msg_out)

        msg_out.data = [vel.item(), vel_telescope.item(), vel_bucket.item()]

        self.manipulator_speed_publisher.publish(msg_out)

        # keep track of msg time, if more than timeout then stop manipulator
        self.prev_target_time = self.get_clock().now().nanoseconds / 1e9

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
                dt=0.1,
                init_state=init_state,
                target_state=target_state,
            )
            .cpu()
            .numpy()
        )

        valve_cmd = M_instance[:, -1, 0].item()

        command = torch.from_numpy(np.asarray([self.boom_pose, valve_cmd])).float()

        model_input = torch.reshape(command, (1, 2))
        model_input = model_input.to(self.model.device, self.model.dtype)
        boom_prediction = self.model.predict(model_input)
        vel = (boom_prediction.mean[0][0]) / (self.get_parameter("dt").value)

        self.get_logger().info("Valve command is: {}".format(valve_cmd))
        self.get_logger().info("Calculated boom velocity is: {}".format(vel.item()))
        self.get_logger().info("Estimated boom velocity is: {}".format(self.boom_vel))

        # bucket

        init_state = np.array(self.bucket_pose)
        target_state = get_tensor(np.array(target_msg.position[BUCKET]))

        M_instance = (
            calc_realization_mean(
                gp_model=self.gpmodel_bucket,
                controller=self.controller_bucket,
                state_dim=self.state_dim,
                control_dim=self.control_dim,
                dt=0.1,
                init_state=init_state,
                target_state=target_state,
            )
            .cpu()
            .numpy()
        )

        valve_cmd = M_instance[:, -1, 0].item()

        command = torch.from_numpy(np.asarray([self.bucket_pose, valve_cmd])).float()

        model_input = torch.reshape(command, (1, 2))
        model_input = model_input.to(
            self.gpmodel_bucket.device, self.gpmodel_bucket.dtype
        )
        prediction = self.gpmodel_bucket.predict(model_input)
        vel_bucket = (prediction.mean[0][0]) / (self.get_parameter("dt").value)
        self.get_logger().info(
            "Calculated bucket velocity is: {}".format(vel_bucket.item())
        )
        self.get_logger().info(
            "Estimated bucket velocity is: {}".format(self.bucket_vel)
        )

        # telescope

        init_state = np.array(self.telescope_pose)
        target_state = get_tensor(np.array(target_msg.position[TELESCOPE]))

        M_instance = (
            calc_realization_mean(
                gp_model=self.gpmodel_telescope,
                controller=self.controller_telescope,
                state_dim=self.state_dim,
                control_dim=self.control_dim,
                dt=0.1,
                init_state=init_state,
                target_state=target_state,
            )
            .cpu()
            .numpy()
        )

        valve_cmd = M_instance[:, -1, 0].item()

        command = torch.from_numpy(np.asarray([self.telescope_pose, valve_cmd])).float()

        model_input = torch.reshape(command, (1, 2))
        model_input = model_input.to(self.model.device, self.model.dtype)
        prediction = self.gpmodel_telescope.predict(model_input)
        vel_telescope = (prediction.mean[0][0]) / (self.get_parameter("dt").value)
        self.get_logger().info(
            "Calculated telescope velocity is: {}".format(vel_telescope.item())
        )
        self.get_logger().info(
            "Estimated telescope velocity is: {}".format(self.telescope_vel)
        )

        # send control

        mani_speed_msg_out = Float64MultiArray()
        mani_speed_msg_out.data = [
            vel.item() * 10,
            vel_telescope.item(),
            vel_bucket.item(),
        ]
        self.manipulator_speed_publisher.publish(mani_speed_msg_out)

        # boom
        self.state_msg_out.position[0] = self.boom_pose
        self.state_msg_out.position[1] = target_msg.position[BOOM]
        self.state_msg_out.velocity[0] = self.boom_vel
        self.state_msg_out.velocity[1] = vel.item()

        # bucket
        self.state_msg_out.position[2] = self.bucket_pose
        self.state_msg_out.position[3] = target_msg.position[BUCKET]
        self.state_msg_out.velocity[2] = self.bucket_vel
        self.state_msg_out.velocity[3] = vel_bucket.item()

        # telescope

        self.state_msg_out.position[4] = self.telescope_pose
        self.state_msg_out.position[5] = target_msg.position[TELESCOPE]
        self.state_msg_out.velocity[4] = self.telescope_vel
        self.state_msg_out.velocity[5] = vel_telescope.item()

        self.state_publisher.publish(self.state_msg_out)

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
