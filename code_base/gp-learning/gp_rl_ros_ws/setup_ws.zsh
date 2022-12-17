#!/usr/bin/zsh
SCRIPT_DIR=$( cd -- "$( dirname -- "$0" )" &> /dev/null && pwd )
export GAZEBO_RESOURCE_PATH=$SCRIPT_DIR/src/gazebo_sim/avant_description:/usr/share/gazebo-11 
if [[ ! -f ./install/setup.zsh ]] then
  printf "Attempting to build ROS packages\n"
  colcon build --symlink-install
fi
source ./install/setup.zsh
