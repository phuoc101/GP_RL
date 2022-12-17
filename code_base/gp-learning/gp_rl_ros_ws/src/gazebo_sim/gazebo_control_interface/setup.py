from setuptools import setup
import os
from glob import glob


package_name = "gazebo_control_interface"
submodule_name = "gazebo_control_interface/GaussianProcess"
submodule_config = "gazebo_control_interface/control_utils/gp_rl/cfg"
submodule_model = "gazebo_control_interface/control_utils/gp_rl/models"
submodule_utils = "gazebo_control_interface/control_utils/gp_rl/utils"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name, submodule_config, submodule_model, submodule_utils],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "data"), glob("data/*.pkl")),
        (os.path.join("share", package_name, "results/gp"), glob("results/gp/*.pkl")),
        (
            os.path.join("share", package_name, "results/controller"),
            glob("results/controller/*.pkl"),
        ),
        (
            os.path.join("share", package_name, "gp_rl/models"),
            glob("./gazebo_control_interface/control_utils/gp_rl/models/*.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="localadmin",
    maintainer_email="teemu.mokkonen@tuni.fi",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "control_client = gazebo_control_interface.gazebo_control_client:main",
            "control_client_gp_node = gazebo_control_interface.gp_control_node:main",
            "pose_control_node = gazebo_control_interface.pose_control_node:main",
        ],
    },
)