import os
import sys
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import argparse
from loguru import logger
import matplotlib.pyplot as plt
import pickle
import numpy as np
from skimage.measure import block_reduce


def get_rosbag_options(path, serialization_format="cdr"):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3")

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    return storage_options, converter_options


def sync_input_sensor(bag_path, sensor_topic, input_topic):
    storage_options, converter_options = get_rosbag_options(bag_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()
    # Create a map for quicker lookup
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    # Set filter for topic of string type
    topics = [sensor_topic, input_topic]
    storage_filter = rosbag2_py.StorageFilter(topics=topics)
    reader.set_filter(storage_filter)

    msg_dict = {}
    msg_dict["timestamp"] = []
    msg_dict["input_cmd"] = []
    msg_dict["boom_position"] = []
    msg_dict["boom_velocity"] = []
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic in topics:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

        input = 0.0
        timestamp = 0.0
        if topic == sensor_topic:
            if len(msg_dict["timestamp"]) == 0:
                start = t
            boom_position = msg.position[2]
            boom_velocity = msg.velocity[2]
        elif topic == input_topic:
            if msg.position[1] > 100:
                input = -(1 - (msg.position[1] - 155) / 100)
            else:
                input = msg.position[1] / 100
            timestamp = (t - start) / 10e8
            msg_dict["boom_position"].append(boom_position)
            msg_dict["boom_velocity"].append(boom_velocity)
            msg_dict["timestamp"].append(timestamp)
            msg_dict["input_cmd"].append(input)
    return msg_dict


def data_handling(msg_dict, downsample, downsampling_factor):
    x = np.array(msg_dict["boom_position"])
    v = np.array(msg_dict["boom_velocity"])
    u = np.array(msg_dict["input_cmd"])
    if downsample:
        logger.debug(f"x shape before downsampling: {x.shape}")
        # full_data = block_reduce(
        #     full_data.T,
        #     block_size=(1, downsampling_factor),
        #     func=np.mean,
        # )
        # full_data = full_data.T
        x = block_reduce(
            x, block_size=downsampling_factor, func=np.mean, cval=np.mean(x)
        )
        v = block_reduce(
            v, block_size=downsampling_factor, func=np.mean, cval=np.mean(x)
        )
        u = block_reduce(
            u, block_size=downsampling_factor, func=np.mean, cval=np.mean(u)
        )
        msg_dict["timestamp"] = msg_dict["timestamp"][::downsampling_factor]
        logger.debug(f"x shape after downsampling: {x.shape}")
    Y1 = np.concatenate(([0], np.diff(x)))
    Y2 = np.concatenate(([0], np.diff(v)))
    orig_length = len(x)
    full_data = np.stack([x, v, u, Y1, Y2], axis=1)
    if downsample and orig_length % downsampling_factor != 0:
        full_data = full_data[:-2, :]
        msg_dict["timestamp"] = msg_dict["timestamp"][:-2]
    xvu = full_data[:, 0:3]
    Y1 = full_data[:, 3]
    Y2 = full_data[:, 4]
    logger.debug(f"xvu shape: {xvu.shape}")
    logger.debug(f"Y1 shape: {Y1.shape}")
    logger.debug(f"Y2 shape: {Y2.shape}")
    data = {"X1_xvu": xvu, "Y1": Y1, "Y2": Y2, "timestamp": msg_dict["timestamp"]}
    return data


def export_data(data, output):
    with open(output, "wb") as f:
        pickle.dump(data, f)
        logger.info(f"Data saved to {os.path.abspath(output)}")
        f.close()


def visualize_data(data, bag, downsample, downsampling_factor):
    logger.debug(f"Position readings count: {len(data['X1_xvu'][:,0])}")
    logger.debug(f"Velocity readings count: {len(data['X1_xvu'][:,1])}")
    logger.debug(f"Input cmd count: {len(data['X1_xvu'][:,2])}")
    title = f"Plots with data from bag {bag}"
    if downsample:
        title += f", data downsampled with factor {downsampling_factor}"
    fig, ax = plt.subplots(3)
    fig.suptitle(title)
    ax[0].plot(data["timestamp"], data["X1_xvu"][:, 0], label="boom position")
    ax[0].plot(data["timestamp"], data["X1_xvu"][:, 1], label="boom velocity")
    ax[0].plot(data["timestamp"], data["X1_xvu"][:, 2], label="input u")
    ax[1].plot(data["timestamp"], data["Y1"], label="Y1")
    ax[2].plot(data["timestamp"], data["Y2"], label="Y2")
    for a in ax:
        a.legend()
    plt.show()


def main(opts):
    msg_dict = sync_input_sensor(
        bag_path=opts.bag, sensor_topic=opts.sensor_topic, input_topic=opts.input_topic
    )
    data = data_handling(
        msg_dict=msg_dict,
        downsample=opts.downsample,
        downsampling_factor=opts.downsampling_factor,
    )
    visualize_data(
        data=data,
        bag=opts.bag,
        downsample=opts.downsample,
        downsampling_factor=opts.downsampling_factor,
    )
    export_data(data, opts.output_file)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bag",
        default="./manipulator_input_w_control_boom_trial_6",
        help="bagfile to read",
    )
    parser.add_argument(
        "-s",
        "--sensor-topic",
        default="/bag_joint_states",
        help="sensor topic from bag",
    )
    parser.add_argument(
        "-i",
        "--input-topic",
        default="/input_valve_cmd",
        help="input command topic from bag",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="./manipulator_input_w_control_boom_trial_6.pkl",
        help="name of output pkl file",
    )
    parser.add_argument(
        "-d",
        "--downsample",
        action="store_true",
        help="Downsample by averaging",
    )
    parser.add_argument(
        "-df",
        "--downsampling-factor",
        default=2,
        type=int,
        help="Downsample by averaging",
    )
    opts = parser.parse_args()
    main(opts)
