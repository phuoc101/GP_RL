import pickle
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node


class DataVisualizer(Node):
    def __init__(self):
        super().__init__("data_visualizer_node")
        self.logger = self.get_logger()
        self.logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.logger.info("Data visualizer started...")
        self.init_ros()

    def init_ros(self):
        # Parameters declared
        self.declare_parameter("datafile", "out.pkl")
        self.datafile = (
            self.get_parameter("datafile").get_parameter_value().string_value
        )

    def visualize_data(self):
        if "xvu" in self.datafile:
            self.logger.debug(
                f"Position readings count: {len(self.data['X1_xvu'][:,0])}"
            )
            self.logger.debug(
                f"Velocity readings count: {len(self.data['X1_xvu'][:,1])}"
            )
            self.logger.debug(f"Input cmd count: {len(self.data['X1_xvu'][:,2])}")
        else:
            self.logger.debug(
                f"Position readings count: {len(self.data['X1_xu'][:,0])}"
            )
            self.logger.debug(f"Input cmd count: {len(self.data['X1_xu'][:,1])}")
        title = "Plots with self.data from dataset {}, frequency {}Hz".format(
            self.data["name"], self.data["frequency"]
        )
        fig, ax = plt.subplots(3)
        fig.suptitle(title)
        if "xvu" in self.datafile:
            ax[0].plot(
                self.data["timestamp"],
                self.data["X1_xvu"][:, 0],
                label="boom position (rad)",
            )
            ax[0].plot(
                self.data["timestamp"], self.data["X1_xvu"][:, 1], label="boom velocity"
            )
            ax[0].plot(
                self.data["timestamp"], self.data["X1_xvu"][:, 2], label="input u"
            )
        else:
            ax[0].plot(
                self.data["timestamp"],
                self.data["X1_xu"][:, 0],
                label="boom position (rad)",
            )
            ax[0].plot(
                self.data["timestamp"], self.data["X1_xu"][:, 1], label="input u"
            )
        ax[1].plot(self.data["timestamp"], self.data["Y1"], label="d_boom_pos (rad)")
        # ax[1, 0].plot(self.data["timestamp"], self.data["Y2"], label="d_boom_vel")
        ax[2].plot(np.diff(self.data["timestamp"]), label="timestmp_diff (sec)")
        for a in ax:
            a.legend()
        plt.legend()
        plt.show()

    def load_data(self):
        with open(self.datafile, "rb") as f:
            self.data = pickle.load(f)
            f.close()


def main():
    rclpy.init()
    visualizer = DataVisualizer()
    visualizer.load_data()
    visualizer.visualize_data()
    visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
