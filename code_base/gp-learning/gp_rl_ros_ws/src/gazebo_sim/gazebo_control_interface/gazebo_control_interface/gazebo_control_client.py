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
import rclpy
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import numpy as np

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

class SteeringActionClient(Node):
    """
    This class handles the communation with the avant model. It will take in the commands from high level nodes and parses them 
    to match the simulation parameters
    """

    def __init__(self):
        super().__init__("gazebo_joint_controller")
        # action client is responsible for sending goals to the 
        # joint_trajectory_controller which executes the joint command to the simulator model
        self.action_client_ = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory') 

        self.speed_publisher = self.create_publisher(Float64MultiArray, "motion_controller/commands", 10)
        self.manipulator_speed_publisher = self.create_publisher(Float64MultiArray, "/manipulator_controller/commands", 10)
        self.resolver_pub_ = self.create_publisher(JointState, "resolver", 10)
        self.telescope_pub_ = self.create_publisher(JointState, "telescope", 10)
        self.wanted_vel_pub = self.create_publisher(JointState, "wanted_speeds", 10)
#        self.manipulator_state_pub = self.create_publisher(JointState, "manipulator_joint_states", 10)

        self.joint_state_sub = self.create_subscription(JointState, "joint_states", self.state_callback, 10)
        self.command_sub_ = self.create_subscription(JointState, "motion_commands", self.command_callback, 10)
        self.manipulator_vel_sub = self.create_subscription(JointState, "manipulator_commands", self.manipulator_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, "avant_cmd_vel", self.cmd_vel_remapper, 10)

        self.declare_parameters(
            namespace="",
            parameters=[
                ("gain_gas", 20),
                ("gain_steering", 2.5),
                ("gain_boom", 2),
                ("gain_bucket", 2)
            ]
        )

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

        #model dimensions
        self.center_wheel_dist = 0.6
        self.dist_between_wheels = 0.6414

        self.prev_msg = []
        self.prev_vel = 0.0

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

    def goal_response_callback(self, future):
        """
        Receives the status of the actions server and tells the user if the goal plan has been accepted

        args:
            future (ros2 actions state): State of the server
        """

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('New state goal got rejected by the server')
            return

        self.get_logger().info('New state goal accepted by the server')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        """
        Gets the results from the action server

        args:
            future (ros2 actions state): State of the server
        """
        result = future.result().result
        self.get_logger().info('Result: '+str(result))

    def feedback_callback(self, feedback_msg):
        """
        Provides the feedback on the server status

        args:
            feedback_msg (feedback state): message containing the feedback data
        """
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback:'+str(feedback))

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


	# publish manipulator_state to separate topic for moveit
#        manipulator_state_msg = JointState()
#        manipulator_state_msg.header = msg.header
#        manipulator_state_msg.name = ["boom_angle","fork_angle","telescope_length"]
#        manipulator_state_msg.position = [0.0,0.0,0.0]
#        manipulator_state_msg.velocity = [0.0,0.0,0.0]
#        manipulator_state_msg.effort = [0.0,0.0,0.0]
	
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
                
#                manipulator_state_msg.position[0] = msg.position[7]
#                manipulator_state_msg.velocity[0] = msg.velocity[7]
#                manipulator_state_msg.effort[0] = msg.effort[7]
		
            if name == "boom_angle":
                self.boom_pose = msg.position[i]
                
#                manipulator_state_msg.position[1] = msg.position[2]
#                manipulator_state_msg.velocity[1] = msg.velocity[2]
#                manipulator_state_msg.effort[1] = msg.effort[2]
		
            if name == "fork_angle":
                self.bucket_pose = msg.position[i]
                
#                manipulator_state_msg.position[2] = msg.position[3]
#                manipulator_state_msg.velocity[2] = msg.velocity[3]
#                manipulator_state_msg.effort[2] = msg.effort[3]
	
#        self.manipulator_state_pub.publish(manipulator_state_msg)

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
        The incoming message should contain the velocities in the velocity part of the message in the 
        following order:

        [boom, bucket, telescope]

        These are then read through, scaled and send to the gazebo joint contoller

        args:
            msg (ROS2 JointState): JointState message containting the wanted values for the control the manipulator
        """
        vel_boom = msg.velocity[BOOM] * self.gain_boom
        vel_tel = msg.velocity[TELESCOPE]
        vel_bucket = msg.velocity[BUCKET] * self.gain_bucket
        msg_out = Float64MultiArray()

        # check if the manipulator joint can move to the given direction to avoid joint conflict with velocities
        if (self.telescope_pose <= TELESCOPE_LOW_LIM and vel_tel < 0.0) or (self.telescope_pose >= TELESCOPE_UPPER_LIM and vel_tel > 0.0):
            vel_tel = 0.0

        if (self.boom_pose < BOOM_LOW_LIM and vel_boom < 0.0) or (self.boom_pose >= BOOM_HIGH_LIM and vel_boom > 0.0):
            vel_boom = 0.0

        if (self.bucket_pose < BUCKET_LOW_LIM and vel_bucket < 0.0) or (self.bucket_pose >= BUCKET_HIGH_LIM and vel_bucket > 0.0):
            vel_bucket = 0.0

        # Send velocity to the ros2 controller which will move the joints
        msg_out.data = [vel_boom, vel_tel, vel_bucket]
        self.manipulator_speed_publisher.publish(msg_out)

    def cmd_vel_remapper(self, msg):
        """
        This function remaps the twist command /cmd_vel which is usually used to control models to wanted_speeds JointState msg

        args:
            msg (ROS2 twist): twist message containting the wanted values for the control the motion
        """
        vx = msg.linear.x
        wz = msg.angular.z
        cmd = JointState()
        cmd.position.append(vx)
        cmd.position.append(-wz) # the command itself is turned otherway around to match the inverted control
        self.wanted_vel_pub.publish(cmd)


def main(args=None):

    rclpy.init()
    # Turn on the ROS2 node and make it run in the loop
    action_client = SteeringActionClient()
    #saction_client.update_joints()
    rclpy.spin(action_client)


if __name__ == "__main__":
    main()
