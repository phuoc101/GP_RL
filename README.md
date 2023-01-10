# 🤖 Data driven model learning for robot simulation and control with Gazebo and ROS2

This is the implementation of our project: Data driven model learning for robot
simulation and control with Gazebo and ROS2.
It is part of the course KONE.533-2022-2023-1-TAU: Robotics Project Work.

# ⚙ Requirements

- The packages were built and tested with ROS Galactic, so it is highly recommended to
  install it with the following instructions:
  [Galactic Installation](https://docs.ros.org/en/galactic/Installation.html)

- Set up a virtual environment with ROS dependencies:

  ```bash
  # the command below will create a python virtualenv inheriting from system packages
  virtualenv --system-site-packages -p python3.8 ~/torch_galactic_venv
  source ~/torch_galactic_venv/bin/activate
  ```

- Install the following packages:

  - [PyTorch](https://pytorch.org/get-started/previous-versions/) (the version used in
    this project is 1.12.1+cu113, but other versions >= 1.8 should work as well)
  - [GPytorch](https://gpytorch.ai/#install)
  - numpy, matplotlib, loguru

# 🚀 Training

Model training is performed in the directory `./code_base/gp-learning/training`. The
joints to be trained and controlled are: boom, bucket and telescope. Some sample
datasets are located in `./code_base/gp-learning/data`. The training datasets are in pkl
format.

## Convert from ROS bag to pkl format:

```bash
# Navigate to ros workspace (change to match your environment)
cd ./code_base/gp-learning/gp_rl_ros_ws
source ./setup_ws.bash # or setup_ws.zsh, depending on your default shell
```

- On one terminal, run the bagreader node, it will wait until there are datas to collect.
  The example below collect data at frequency of 10Hz, change the parameters to your needs

```bash
ros2 run avant_bagreader bagreader_node --ros-args \
    --param frequency:=10 \
    --param output:=boom_trial_1_10hz.pkl
```

- Open another terminal and play a ROS 2 bag that you want to collect data from:

```bash
ros2 bag play <BAG_NAME>
```

## Train GP models and controller

- To train everything with default configurations:

```bash
./gp_rl.sh
```

- To train a specific joint with specific configurations:

```bash
python gp_rl_main.py --verbose DEBUG --trials 5 --trial-max-iter 20\
  --num-states 1 --train-data ../data/bucket_trial_2_10hz.pkl --tf 10 --dt 0.1\
  --test-data ../data/bucket_trial_2_10hz.pkl \
  --optimizer NAdam --nondet-init --force-train-gp \
  --verbose INFO --joint bucket --input-lim 1\
  --gpmodel ./results/gp/GPmodel_bucket.pkl --plot-mc
```

- Important parameters to tune:

  - `--train-data` and `--test-data`: Paths to specific dataset
  - `--input-lim`: Hard limits set for the input (0-1)
  - `--joint`: Joint to train (options: boom, bucket, telescope; default: boom)
  - `--tf`: The time horizon to train the trajectories
  - For more details on the parameters:

  ```bash
  python gp_rl_main.py --help
  ```

- The results are stored in `./code_base/gp-learning/training/results`
- To visualize the results:
  ```bash
  python demo_gp_controller.py --verbose INFO --joint bucket --visualize-gp \
      --test-gp-data ../data/bucket_trial_2_10hz.pkl --dt 0.1
  ```

# 🧰 Run the simulation

```bash
# Navigate to ros workspace (change to match your environment)
source ~/torch_galactic_venv/bin/activate
cd ./code_base/gp-learning/gp_rl_ros_ws
source ./setup_ws.bash # or setup_ws.zsh, depending on your default shell
```

- The setup file will set attempt to build the package if it is not built yet, set up
  symlinks to the necessary directories (important) and source the package. To rebuild
  the package, run:
  ```bash
  colcon build --symlink-install
  ```
- Run the simulation:

```
ros2 launch avant_description gp.launch.py
```

- Open up another terminal and send the desired positions for the joints:

```bash
ros2 topic pub -r 10 /goal_pose sensor_msgs/msg/JointState "{position: [0.5, 0.2, 0.4]}"
```
