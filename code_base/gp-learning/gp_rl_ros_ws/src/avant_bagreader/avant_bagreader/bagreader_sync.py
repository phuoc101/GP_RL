import os
import sys
import pickle
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from ament_index_python import get_package_prefix


INSTALL_PATH = get_package_prefix("avant_bagreader")
DATA_PATH = os.path.abspath(
    os.path.join(INSTALL_PATH, "../../src/avant_bagreader/data")
)
os.makedirs(DATA_PATH, exist_ok=True)


class DataSync(Node):
    def __init__(self):
        super().__init__("bagreader_node")
        self.logger = self.get_logger()
        self.logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.logger.info("Started bagreader node, waiting for messages...")
        self.init_ros()
        self.logger.info(
            "Output: {}, Frequency: {}".format(self.output, self.frequency)
        )
        self.logger.debug("Data path: {}".format(DATA_PATH))
        self.STARTED = False
        self.COLLECTING = True
        self.HAS_DATA = False
        self.timeout_cnt = 0
        self.boom_position = 0
        self.boom_velocity = 0
        self.input = 0
        self.msg_dict = {}
        self.msg_dict["boom_position"] = []
        self.msg_dict["boom_velocity"] = []
        self.msg_dict["input"] = []
        self.msg_dict["timestamp"] = []

    def init_ros(self):
        qos_profile = QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.SYSTEM_DEFAULT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # Parameters declared
        self.declare_parameter("frequency", 10)
        self.declare_parameter("output", "out.pkl")
        # self.declare_parameter("num_input", 2)
        # get parameters
        self.frequency = (
            self.get_parameter("frequency").get_parameter_value().integer_value
        )
        self.timer_period = 1 / self.frequency
        # timeout count after bag has finished playing
        self.timeout_max = 5 / self.timer_period
        # output file name
        self.output = self.get_parameter("output").get_parameter_value().string_value
        self.name = os.path.splitext(self.output)[0]
        # Pub sub
        self.joint_state_sub = self.create_subscription(
            JointState, "/bag_joint_states", self.joint_state_callback, qos_profile
        )
        self.input_sub = self.create_subscription(
            JointState, "/input_valve_cmd", self.input_callback, qos_profile
        )
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        # self.num_inputs = (
        #     self.get_parameter("num_input").get_parameter_value().integer_value
        # )

    def joint_state_callback(self, msg):
        if self.STARTED:
            self.boom_position = msg.position[2]
            self.boom_velocity = msg.velocity[2]

    def input_callback(self, msg):
        if not self.STARTED:
            self.STARTED = True
            self.logger.info("Data Collecting started")
        else:
            if msg.position[1] > 100:
                self.input = -(1 - (msg.position[1] - 155) / 100)
            else:
                self.input = msg.position[1] / 100
            self.HAS_DATA = True

    def timer_callback(self):
        if self.COLLECTING:
            if self.STARTED and self.HAS_DATA:
                timestamp = rclpy.clock.Clock().now().seconds_nanoseconds()
                sec = timestamp[0]
                nsec = timestamp[1]
                timestamp_sec = sec + nsec / 1e9
                self.msg_dict["timestamp"].append(timestamp_sec)
                self.msg_dict["boom_position"].append(self.boom_position)
                self.msg_dict["boom_velocity"].append(self.boom_velocity)
                self.msg_dict["input"].append(self.input)
                self.HAS_DATA = False
            elif self.STARTED and not self.HAS_DATA:
                timestamp = rclpy.clock.Clock().now().seconds_nanoseconds()
                sec = timestamp[0]
                nsec = timestamp[1]
                timestamp_sec = sec + nsec / 1e9
                self.msg_dict["timestamp"].append(timestamp_sec)
                self.msg_dict["boom_position"].append(self.boom_position)
                self.msg_dict["boom_velocity"].append(self.boom_velocity)
                self.msg_dict["input"].append(self.input)
                self.timeout_cnt += 1
                if self.timeout_cnt == self.timeout_max:
                    self.COLLECTING = False
        else:
            data = self.data_handling()
            output_file = os.path.join(DATA_PATH, self.output)
            self.export_data(data, output_file)

    def export_data(self, data, output):
        with open(output, "wb") as f:
            pickle.dump(data, f)
            self.logger.info(f"Data saved to {os.path.abspath(output)}")
            f.close()
        self.destroy_node()
        sys.exit(0)

    def data_handling(self):
        # start from index 1 to prevent random jumps in value
        t = np.array(self.msg_dict["timestamp"])[1:]
        # training input
        x = np.array(self.msg_dict["boom_position"])[1:]
        v = np.array(self.msg_dict["boom_velocity"])[1:]
        u = np.array(self.msg_dict["input"])[1:]
        # training output
        Y1 = np.concatenate(([0], np.diff(x)))
        Y2 = np.concatenate(([0], np.diff(v)))
        # stack inputs
        # if self.num_inputs == 3:
        xvu = np.stack([x, v, u], axis=1)
        self.logger.debug(f"xvu shape: {xvu.shape}")
        self.logger.debug(f"Y1 shape: {Y1.shape}")
        self.logger.debug(f"Y2 shape: {Y2.shape}")
        # data = {
        #     "name": self.name,
        #     "frequency": self.frequency,
        #     "X1_xvu": xvu,
        #     "Y1": Y1,
        #     "Y2": Y2,
        #     "timestamp": t,
        # }
        # elif self.num_inputs == 2:
        xu = np.stack([x, u], axis=1)
        self.logger.debug(f"xu shape: {xu.shape}")
        self.logger.debug(f"Y1 shape: {Y1.shape}")
        data = {
            "name": self.name,
            "frequency": self.frequency,
            "X1_xvu": xvu,
            "X1_xu": xu,
            "Y1": Y1,
            "Y2": Y2,
            "timestamp": t,
        }
        return data


def main(args=None):
    rclpy.init(args=args)
    data_sync = DataSync()
    rclpy.spin(data_sync)


if __name__ == "__main__":
    main()
