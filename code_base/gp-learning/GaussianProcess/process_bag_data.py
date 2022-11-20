from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py 

import matplotlib.pyplot as plt


import numpy as np


def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


reader = rosbag2_py.SequentialReader()
bag_path = "mani_with_input_filtered"
storage_options, converter_options = get_rosbag_options(bag_path)

reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()

type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}



boom_start_nanosec = 0.0
boom_input_start_nanosec = 0.0

boom_position_array = []
boom_velocity_array = []
boom_time_array = []


boom_position_delta_array = []
boom_velocity_delta_tarray = []

boom_input_array = []
boom_input_time_array = []



while reader.has_next():
    (topic, data, t) = reader.read_next()
    if "CAN" not in topic:
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)


    if topic == "/bag_joint_states":
        if len(boom_time_array)==0:
            boom_start_nanosec = t 
        diff = (t - boom_start_nanosec)/10e8
        print("| t = " + str(diff)+" seconds |")

        boom_position_array.append(msg.position[2])
        boom_velocity_array.append(msg.velocity[2])
        boom_time_array.append(diff)

    if topic == "/input_valve_cmd":
        if len(boom_input_time_array)==0:
            boom_input_start_nanosec = t 
        diff = (t - boom_start_nanosec)/10e8
        print("| t = " + str(diff)+" seconds |")
        if (msg.position[1]>100):
            boom_input_array.append(-(msg.position[1]-150)/100)
        else:
            boom_input_array.append(msg.position[1]/100)

        boom_input_time_array.append(diff)

print("position/velocity length: " + str(len(boom_position_array)))
print("input length: " + str(len(boom_input_array)))

plt.plot(boom_time_array,boom_position_array)
plt.plot(boom_time_array,boom_velocity_array)
plt.plot(boom_input_time_array,boom_input_array)

#TODO downsampling and state difference


plt.show()