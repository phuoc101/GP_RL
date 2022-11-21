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
bag_path = "manipulator_input_w_control_boom_trial_6"
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
boom_velocity_delta_array = []

boom_input_array = []
boom_input_time_array = []

joint_states_drop_rate = 10
drop = 0


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
        if (drop % joint_states_drop_rate ==0):
            boom_position_array.append(msg.position[2])
            boom_velocity_array.append(msg.velocity[2])
            boom_time_array.append(diff)
        drop +=1

    if topic == "/input_valve_cmd":
        if len(boom_input_time_array)==0:
            boom_input_start_nanosec = t 
        diff = (t - boom_start_nanosec)/10e8
        print("| t = " + str(diff)+" seconds |")

        # for positive u (pulling joystick toward yourself) --> minimum is 0 and maximum is 99 (take 100)
        # for negative u (pulling joystick away from yourself) --> the farther away joystick is (full force) --> 157 and closer to you then 250 
        # ---> for negative force -1 --> 157 and -0.001 --> 250

        # 1 - (message-150 / 100)  157 --> -1 and 254/255 --> -0.0 ===> 1 - (message - 155)/100
        if (msg.position[1]>100):
            boom_input_array.append(    -(1 -(msg.position[1]-155)/100)  )
        else:
            boom_input_array.append(msg.position[1]/100)

        boom_input_time_array.append(diff)

print("position/velocity length: " + str(len(boom_position_array)))
print("input length: " + str(len(boom_input_array)))

plt.plot(boom_time_array,boom_position_array,label="position")
plt.plot(boom_time_array,boom_velocity_array,label="velocity")
plt.plot(boom_input_time_array,boom_input_array,label="input")
plt.legend()
#TODO downsampling and state difference

boom_position_delta_array = (np.diff(np.array(boom_position_array))).tolist()
boom_velocity_delta_array = (np.diff(np.array(boom_velocity_array))).tolist()

plt.show()