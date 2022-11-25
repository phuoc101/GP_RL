import os
import sys
import pickle
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile


class DataSync(Node):
    def __init__(self):
        super().__init__("data_generator_node")
        self.logger = self.get_logger()
        self.logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.logger.debug("Started data syncing node, waiting for messages...")
        qos_profile = QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.SYSTEM_DEFAULT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # 100 Hz
        self.timer_period = 0.01
        self.joint_state_sub = self.create_subscription(
            JointState, "/bag_joint_states", self.joint_state_callback, qos_profile
        )
        self.input_sub = self.create_subscription(
            JointState, "/input_valve_cmd", self.input_callback, qos_profile
        )
        self.STARTED = False
        self.COLLECTING = True
        self.HAS_DATA = False
        self.timeout_cnt = 0
        self.timeout_max = 1/self.timer_period
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.boom_position = 0
        self.boom_velocity = 0
        self.input = 0
        self.msg_dict = {}
        self.msg_dict["boom_position"] = []
        self.msg_dict["boom_velocity"] = []
        self.msg_dict["input"] = []
        self.msg_dict["timestamp"] = []

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
            self.export_data(data, "./out_10.pkl")

    def export_data(self, data, output):
        with open(output, "wb") as f:
            pickle.dump(data, f)
            self.logger.info(f"Data saved to {os.path.abspath(output)}")
            f.close()
        self.destroy_node()
        # rclpy.shutdown()
        sys.exit(0)

    def data_handling(self):
        x = np.array(self.msg_dict["boom_position"])
        v = np.array(self.msg_dict["boom_velocity"])
        u = np.array(self.msg_dict["input"])
        xvu = np.stack([x, v, u], axis=1)
        Y1 = np.concatenate(([0], np.diff(x)))
        Y2 = np.concatenate(([0], np.diff(v)))
        self.logger.debug(f"xvu shape: {xvu.shape}")
        self.logger.debug(f"Y1 shape: {Y1.shape}")
        self.logger.debug(f"Y2 shape: {Y2.shape}")
        data = {
            "X1_xvu": xvu,
            "Y1": Y1,
            "Y2": Y2,
            "timestamp": self.msg_dict["timestamp"],
        }
        return data


def main(args=None):
    rclpy.init(args=args)
    data_sync = DataSync()
    rclpy.spin(data_sync)


if __name__ == "__main__":
    main()
