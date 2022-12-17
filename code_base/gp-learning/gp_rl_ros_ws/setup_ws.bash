#!/usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "$0" )" &> /dev/null && pwd )
export GAZEBO_RESOURCE_PATH=$SCRIPT_DIR/src/gazebo_sim/avant_description:/usr/share/gazebo-11 
if [[ ! -f ./install/setup.bash ]]; then
  printf "Attempting to build ROS packages\n"
  colcon build --symlink-install
fi
if [[ ! -d $SCRIPT_DIR/src/gazebo_sim/gazebo_control_interface/data ]]; then
  ln -s "$SCRIPT_DIR/../data" "$SCRIPT_DIR/src/gazebo_sim/gazebo_control_interface/data"
fi
if [[ ! -d $SCRIPT_DIR/src/gazebo_sim/gazebo_control_interface/results ]]; then
  ln -s "$SCRIPT_DIR/../training/results" "$SCRIPT_DIR/src/gazebo_sim/gazebo_control_interface/results"
fi
if [[ ! -d $SCRIPT_DIR/src/gazebo_sim/gazebo_control_interface/gp_rl ]]; then
  ln -s "$SCRIPT_DIR/../training/gp_rl" "$SCRIPT_DIR/src/gazebo_sim/gazebo_control_interface/gp_rl"
fi
source $SCRIPT_DIR/install/setup.bash
