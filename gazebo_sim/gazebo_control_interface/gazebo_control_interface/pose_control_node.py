"""
Teemu Mökkönen, Tampere University

This python file contains needed interfaces to send commands to the ros2 gazebo controller which moves the joints according to the 
given commands in the gazebo environment. This node handles the control for the manipulator joints and for the moving the model. Under are specified the 
controllable joints and what is the method to control them.

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
    - all joints mentioned for the controller receive the speed that is defined in the command
"""

from math import atan
from loguru import logger
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from pathlib import Path
from gazebo_control_interface import configs, GPModel
from rclpy.time import Time


from .control_utils.gp_rl.cfg.configs import get_gp_train_config
from .control_utils.gp_rl.models.GPModel import GPModel
from .control_utils.gp_rl.utils.data_loading import load_data
from .control_utils.gp_rl.utils.torch_utils import to_gpu, get_tensor
from .control_utils.gp_rl.utils.rl_utils import calc_realization_mean

import matplotlib.pyplot as plt
import torch
import numpy as np
import rclpy
import argparse
import sys
import os
# Constants for the system
GEAR_MOVE = 1
GEAR_FREE = 0
GEAR_MOVE_BACKWARD = -1
GEAR  = 0
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

dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

class SteeringActionClient(Node):
    """
    This class handles the communation with the avant model. It will take in the commands from high level nodes and parses them 
    to match the simulation parameters
    """

    def __init__(self, opts):
        super().__init__("gazebo_joint_controller")

        self.state_dim = 0
        self.control_dim = 0

        self.model = self.init_model(opts)
        self.controller = self.init_controller(opts)
        # action client is responsible for sending goals to the 
        # joint_trajectory_controller which executes the joint command to the simulator model

        #prediction_model = run_model_single_input()

        self.speed_publisher = self.create_publisher(Float64MultiArray, "motion_controller/commands", 10)
        self.manipulator_speed_publisher = self.create_publisher(Float64MultiArray, "/manipulator_controller/commands", 10)
        self.state_publisher = self.create_publisher(JointState, "/states", 10)
        self.resolver_pub_ = self.create_publisher(JointState, "resolver", 10)
        self.telescope_pub_ = self.create_publisher(JointState, "telescope", 10)

        self.joint_state_sub = self.create_subscription(JointState, "joint_states", self.state_callback, 10)
        self.command_sub_ = self.create_subscription(JointState, "motion_commands", self.command_callback, 10)
        self.manipulator_vel_sub = self.create_subscription(JointState, "manipulator_commands", self.manipulator_callback, 10)
        
        self.prev_pose = None
        self.prev_time = None
        self.logger = 0

        self.declare_parameters(
            namespace="",
            parameters=[
                ("gain_gas", 20),
                ("gain_steering", 2.5),
                ("gain_boom", 2),
                ("gain_bucket", 2)
            ]
        )

        # msg definations

        self.msg_out = JointState()
        self.msg_out.position.append(0.0)
        self.msg_out.position.append(0.0)
        self.msg_out.velocity.append(0.0)
        self.msg_out.velocity.append(0.0)

        # gains for control

        self.gas_gain = self.get_parameter("gain_gas").value
        self.gain_steering = self.get_parameter("gain_steering").value
        self.gain_boom = self.get_parameter("gain_boom").value
        self.gain_bucket = self.get_parameter("gain_bucket").value

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

        #model dimensions
        self.center_wheel_dist = 0.6
        self.dist_between_wheels = 0.6414

        self.prev_est_time = self.get_clock().now().nanoseconds / 1e9

        self.prev_msg = []
        self.prev_vel = 0.0
        self.prev_command = 0.0

    def init_model(self, opts): 
        """
        initialize gp model 
        """
        self.state_dim = opts.state_dim
        self.control_dim = opts.control_dim
        gpmodel = GPModel(**get_gp_train_config())
        gpmodel.initialize_model(
        path_model="/home/teemu/results/gp/GPmodel.pkl",
        # uncomment the lines below for retraining
         path_train_data="/home/teemu/data/boom_trial_6_10hz.pkl",
         force_train=opts.force_train_gp,
        )

        return gpmodel

    def init_controller(self, opts):
        controller_data = load_data("./results/controller/_all.pkl")
        controller = self.get_best_controller(controller_data)
        to_gpu(controller)
        logger.info("controller model loaded")


    def get_best_controller(controller_data):
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
            self._send_goal_future = self.action_client_.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

            self._send_goal_future.add_done_callback(self.goal_response_callback)

            self.prev_msg = goal_data.positions


    def command_callback(self, msg):
        """
        Acts as interface for parsing avant motion commands and passes them to corresponding 
        parties to make the simulation move.

        args:
            msg (JointState msg): JointState message containing abstact variables gas, steering and gear
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
                self.telescope_pub_.publish(telescope_msg)

            if name == "boom_angle":
                self.boom_pose = msg.position[i]
                if (self.prev_pose != None and self.prev_time != None):           
                    self.boom_vel = (self.boom_pose - self.prev_pose) / ((Time.from_msg(msg.header.stamp).nanoseconds  / 1e9) - self.prev_time) * 10 #msg.velocity[i] * 10# scale the vel to match given vel   
                self.prev_pose = self.boom_pose
                self.prev_time = Time.from_msg(msg.header.stamp).nanoseconds  / 1e9
                #self.boom_vel = msg.velocity[i] * 10# scale the vel to match given vel
		
            if name == "fork_angle":
                self.bucket_pose = msg.position[i]
                self.bucket_vel = msg.velocity[i]

    def publish_vel(self, gas, steer, gear):
        """
        Publishes the wanted speed for the rear tires as "gas" value. This might need to be scaled to fit 
        the simulation later on! 

        args:
            gas (float): Gas is abstract command from the avant motion control interface
            
        """

        # do not send more messages to server than needed to avoid crowding it
        if self.prev_vel !=  gas or self.steer != steer:
            msg = Float64MultiArray()
            gas = gas * self.gas_gain # scale the gas to match the real value
            steer = steer * self.gain_steering # scale steer avoid over shooting with control
            gas_right = gas
            gas_left = gas
            if self.resolver_pos < -0.20:
                gas_right = 0.2 + gas * (self.center_wheel_dist / (atan(np.pi - (abs(self.resolver_pos)) / 2)) + (self.dist_between_wheels / 2))
                gas_left = 0.2 + gas * (self.center_wheel_dist / atan(np.pi - abs(self.resolver_pos)) / 2) - (self.dist_between_wheels / 2) 
            
            if self.resolver_pos > 0.20:
                gas_left = 0.2 + gas * (self.center_wheel_dist / (atan(np.pi - (abs(self.resolver_pos)) / 2)) + (self.dist_between_wheels / 2))
                gas_right = 0.2 + gas * (self.center_wheel_dist / atan(np.pi - abs(self.resolver_pos)) / 2) - (self.dist_between_wheels / 2) 
            
            # check if the central joint is able to move anymore
            if (self.resolver_pos > 0.52 and steer > 0) or (self.resolver_pos < -0.52 and steer < 0):
                steer = 0.0

            # Publish speed for all four tires and for frame steering
            msg.data = [gas_right, gas_left, gas_right, gas_left, steer]
            self.speed_publisher.publish(msg)
            self.prev_vel = gas
    
    def manipulator_callback(self, msg):
        """
        Published wanted values for the manipulator as velocity commands given to the controller
        The incoming message should contain the corresponding valve in the velocity part of the message in the 
        following order:

        [boom, bucket, telescope]

        These are then read through, scaled and send to the gazebo joint contoller

        args:
            msg (ROS2 JointState): JointState message containting the wanted values for the control the manipulator
        """
        vel_boom = msg.velocity[BOOM] #* self.gain_boom
        vel_tel = msg.velocity[TELESCOPE]
        vel_bucket = msg.velocity[BUCKET] * self.gain_bucket
        command = torch.from_numpy(np.asarray([self.boom_pose, self.boom_vel, vel_boom])).float()
        
        model_input  = torch.reshape(command,(1,3))
        model_input = model_input.to(self.model.device, self.model.dtype)
        boom_prediction = self.model.predict(model_input)

        msg_out = Float64MultiArray()

        # check if the manipulator joint can move to the given direction to avoid joint conflict with velocities
        if (self.telescope_pose <= TELESCOPE_LOW_LIM and vel_tel < 0.0) or (self.telescope_pose >= TELESCOPE_UPPER_LIM and vel_tel > 0.0):
            vel_tel = 0.0

        if (self.boom_pose < BOOM_LOW_LIM and vel_boom < 0.0) or (self.boom_pose >= BOOM_HIGH_LIM and vel_boom > 0.0):
            vel_boom = 0.0

        if (self.bucket_pose < BUCKET_LOW_LIM and vel_bucket < 0.0) or (self.bucket_pose >= BUCKET_HIGH_LIM and vel_bucket > 0.0):
            vel_bucket = 0.0


        time = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        #self.get_logger().info("prev time: {} \n".format(self.prev_est_time))
        #self.get_logger().info("time now : {} \n".format(time))
        vel = (boom_prediction.mean[0][0]) / (time - self.prev_est_time) * 8 # (8 / 10)
    
        self.prev_est_time = time
        # Send velocity to the ros2 controller which will move the joints
        
        self.get_logger().info("valve cmd: {} \n".format(vel_boom))
        self.get_logger().info("predicted pos: {} \n".format(boom_prediction.mean[0][0]))
        self.get_logger().info("boom pose: {} \n".format(self.boom_pose))
        self.get_logger().info("calculated vel: {} \n".format(self.boom_vel))
        self.get_logger().info("estimated vel: {} \n".format(vel.item()))
        self.logger = 0
		# float(boom_vel.mean[0][1]),

        self.msg_out.position[0] =(self.boom_pose)
        self.msg_out.position[1] = boom_prediction.mean[0][0]
        self.msg_out.velocity[0] = self.boom_vel
        self.msg_out.velocity[1] = vel.item() 

        self.state_publisher.publish(self.msg_out)

        if -0.2 <= vel.item() <= 0.2:
            vel = 0.0
        
        else:
            vel = vel.item()
        
        msg_out.data = [vel, vel_tel, vel_bucket]
        
        self.manipulator_speed_publisher.publish(msg_out)
        self.logger += 1


    def wanted_pos_callback(self, msg):
        """
        moves the robot boom manipulator to the wanted state according to the given command

        args:
            msg Jointstate: pos [1]
        """

        init_state = np.array(self.boom_pose[BOOM])
        target_state = get_tensor(np.array(msg.position[BOOM]))
        X_sample_tensor = get_tensor([self.boom_pose, self.boom_vel]).reshape(1, 2)
        pred = self.model.predict(X_sample_tensor)

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

def main(args=None):
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--tf", default=25, type=float, help="Time to run simulation")  # noqa
    parser.add_argument("--dt", default=0.1, type=float, help="Sampling time")  # noqa
    parser.add_argument("--state-dim", default=1, type=int, help="Number of observed states")  # noqa
    parser.add_argument("--control-dim", default=1, type=int, help="Number of controlled states")  # noqa
    parser.add_argument("--init-state", nargs="+", default=[-1], help="Define initial state (based on state-dim)")  # noqa
    parser.add_argument("--target-state", nargs="+", default=[0], help="Define goal state (based on state-dim)")  # noqa
    parser.add_argument("--visualize-gp", action="store_true", help="Visualize GP")  # noqa
    parser.add_argument("--force-train-gp", action="store_true", help="Force train GP Model again")  # noqa
    parser.add_argument("--verbose", default="DEBUG", type=str, help="Verbosity level (INFO, DEBUG, WARNING, ERROR)")  # noqa
    # fmt: on
    opts = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level=opts.verbose)
    rclpy.init()
    # Turn on the ROS2 node and make it run in the loop
    action_client = SteeringActionClient(opts)
    #saction_client.update_joints()
    rclpy.spin(action_client)


if __name__ == "__main__":
    main()
