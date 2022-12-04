import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(JointState, 'manipulator_commands', 10)
        self.manipulator_vel_sub = self.create_subscription(Twist, "man_com", self.com_call, 10)
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.msg = JointState()
        self.msg.velocity = [0.0, 0.0, 0.0]

    def timer_callback(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.msg)
        print("command sent")

    def com_call(self, msg):
        self.msg[0] = msg.linear.x


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()