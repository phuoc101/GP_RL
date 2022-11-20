import rclpy
from rclpy.node import Node
import os
from pathlib import Path
import sys

from rcl_interfaces.msg import Log
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String

import rosbag2_py 

import matplotlib.pyplot as plt


import numpy as np


from sensor_msgs.msg import JointState,Imu

def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


reader = rosbag2_py.SequentialReader()
bag_path = "mani_with_cmd"
storage_options, converter_options = get_rosbag_options(bag_path)

reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()

type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}


telescope_array = []
telescope_time_array = []


boom_array = []
boom_time_array = []



boom_start_nanosec = 0.0
telescope_start_nanosec = 0.0


while reader.has_next():
    (topic, data, t) = reader.read_next()
    if "CAN" not in topic:
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)


    if topic == "/boom_inclinometer":
        if len(telescope_array)==0:
            telescope_start_nanosec = t 
        diff = (t - telescope_start_nanosec)/10e8#
        telescope_array.append(msg.linear_acceleration.z)
        boom_array.append(msg.linear_acceleration.y)
        telescope_time_array.append(diff)




plt.plot(telescope_time_array,telescope_array)
plt.plot(telescope_time_array,boom_array)

plt.show()