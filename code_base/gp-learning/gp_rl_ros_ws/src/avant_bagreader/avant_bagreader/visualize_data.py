import pickle
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node

BOOM = 0
BUCKET = 1
TELES = 2
JOINTS = ["boom", "bucket", "telescope"]


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
                f"Position readings count: {len(self.data['X1_xvu'][BOOM, :, 0])}"
            )
            self.logger.debug(
                f"Velocity readings count: {len(self.data['X1_xvu'][BOOM, :, 1])}"
            )
            self.logger.debug(
                f"Input cmd count: {len(self.data['X1_xvu'][BOOM, :, 2])}"
            )
        else:
            self.logger.debug(
                f"Position readings count: {len(self.data['X1_xu'][BOOM, :, 0])}"
            )
            self.logger.debug(f"Input cmd count: {len(self.data['X1_xu'][BOOM, : 1])}")
        title = "Plots with self.data from dataset {}, frequency {}Hz".format(
            self.data["name"], self.data["frequency"]
        )
        fig, ax = plt.subplots(2, 3)
        fig.suptitle(title)
        for i, joint in enumerate(JOINTS):
            if "xvu" in self.datafile:
                ax[0][i].plot(
                    self.data["timestamp"],
                    self.data["X1_xvu"][i, :, 0],
                    label=f"{joint} position (rad)",
                )
                ax[0][i].plot(
                    self.data["timestamp"],
                    self.data["X1_xvu"][i, :, 1],
                    label=f"{joint} velocity",
                )
                ax[0][i].plot(
                    self.data["timestamp"],
                    self.data["X1_xvu"][i, :, 2],
                    label=f"{joint} input",
                )
            else:
                ax[0][i].plot(
                    self.data["timestamp"],
                    self.data["X1_xu"][i, :, 0],
                    label=f"{joint} position (rad)",
                )
                ax[0][i].plot(
                    self.data["timestamp"],
                    self.data["X1_xu"][i, :, 1],
                    label=f"{joint} input",
                )
            ax[1][i].plot(
                self.data["timestamp"],
                self.data["Y1"][i, :],
                label=f"d_{joint}_pos (rad)",
            )
            # ax[2][i].plot(
            #     np.diff(self.data["timestamp"]), label="timestmp_diff (sec)"
            # )
        for a in ax:
            for p in ax:
                for pl in p:
                    pl.legend()
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
