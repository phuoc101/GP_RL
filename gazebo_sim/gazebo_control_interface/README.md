# Avant gazebo control

This package adds and controllability interface to the gazebo simulation. With this package the avant simulation model can be sent messages and they will moved to the model.

with this interface it is possible to control the movement of the model. Over the topic /motion_controller/commands the interface listens the commands

which come as JointState commands in form of:

```
ros2 topic pub /Motion_commands sensor_msgs/msg/JointState "{name: ['gear', 'steering', 'gas'], position: [0.0, 0.0, 0.0]}"
```

These values are then turned to velocity commands for the wheels and articulated joint.

For the manipulator there are two controllers possible to use. 

JointTrajectory controller:

With this user can give angles wanted to execute for the manipulator, but there is no command interface for the user yet.

Velocity controller:

With this controller user can give commands in same fashion as for the motion control. Messages used to control manipulator come in as

```
ros2 topic pub /manipulator_commands sensor_msgs/msg/JointState "{name: ['boom', 'bucket', 'telescope'], velocity: [1.0, 0.0, 0.0]}"
```


## To run and install needed files

This package is used with avant description package which launches this package when needed.: 

There you can find commands which will instruct how to run each controller mode and how the simulation is started!
