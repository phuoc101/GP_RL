# Avant Description

This package contains necessary ROS2 packages to run gazebo simulation with avant.
This package has multiple plugin depecies to show the model in the gazebo environment.

## Sensors

There are sensors also added to the model:
    - IMU for front axle
    - IMU for the boom
    - IMU for the bucket
    - Lidar
    - GNSS
### Depencies

To run this file you will need to have installed the ros2 galactic, rviz2 and gazebo 
environments. 

### plugins

You need to have installed following plugins to run the package.

You install required depencies by running following command in the workspace:

```
rosdep install -r --from-paths src -y --ignore-src
```

or by installing the following manually:

ros2 controller:
```
sudo apt-get install ros-galactic-ros2-control
```

ros2 controllers:
```
sudo apt-get install ros-galactic-ros2-controllers
```

ros2 gazebo controller:
```
sudo apt-get install ros-galactic-gazebo-ros2-control
```

xarco:
```
sudo apt install ros-galactic-xacro
```

robot_localization
```
sudo apt install ros-galactic-robot-localization
```

Please add the following line to the end of your bashrc file to have the april tag adding work!

open bashrc:

```
gedit ~/.bashrc
```

add the following line modified to fit your configuration at the end in the file:

```
export GAZEBO_RESOURCE_PATH=~/<route/to/your/package>:/usr/share/gazebo-<your-gazebo-version>/
```

This will add your package as one of the gazebo resource paths, which allows using custom textures on the model.


### to run the project

download the files to your workspace folder inside the src file after which build it with:

```
colcon build
```

and source the setup file. 

```
source install/local_setup.bash
```

and the you can run needed file to control the model with:

```
ros2 launch avant_description control.launch.py
```

This will launch gazebo, rviz and control node for the model. To control model you need to send it JointState command
to control the linear velocity and angular velocity. motion control node will turn the commands then suitable for the 
controller. 

If the user wants to modify the spawn location of the model, the starting location can be defined after the launch command as follows:

```
ros2 launch avant_description control.launch.py x:=1.0 y:=2.0 z:=0.2 yaw:=1.0
```


example command to use in command line tool:

```
ros2 topic pub /wanted_speeds sensor_msgs/msg/JointState "{name: ['vel', 'ang_vel'], position: [5.0, 10.0], velocity: [0.0, 0.0], effort: [0.0, 0.0]}"
```

This command will send linear velocity command of 5.0 and angular velocity command of 10.0 to motion controller which will scale the command to fit to the model.

