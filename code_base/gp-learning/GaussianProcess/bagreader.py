# Copy from
# https://github.com/ros2/rosbag2/blob/master/rosbag2_py/test/test_sequential_reader.py
import sys
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import argparse
from loguru import logger
import matplotlib.pyplot as plt
import pickle


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


def visualize_data(msg_dict):
    logger.debug(f"Position readings count: {len(msg_dict['boom_position'])}")
    logger.debug(f"Velocity readings count: {len(msg_dict['boom_velocity'])}")
    logger.debug(f"Input cmd count: {len(msg_dict['input_cmd'])}")

    plt.plot(msg_dict["timestamp"], msg_dict["input_cmd"], label="input u")
    plt.plot(msg_dict["timestamp"], msg_dict["boom_position"], label="boom position")
    plt.plot(msg_dict["timestamp"], msg_dict["boom_velocity"], label="boom velocity")
    plt.legend()
    plt.show()


def export_data(msg_dict, output):
    with open(output, "wb") as f:
        pickle.dump(msg_dict, f)
        f.close()


def main(opts):
    msg_dict = sync_input_sensor(
        bag_path=opts.bag, sensor_topic=opts.sensor_topic, input_topic=opts.input_topic
    )
    visualize_data(msg_dict=msg_dict)
    export_data(msg_dict, opts.output_file)


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
    opts = parser.parse_args()
    main(opts)
