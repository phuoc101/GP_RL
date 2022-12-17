from setuptools import setup
import os
from glob import glob


package_gazebo = "gazebo_control_interface"
package_control = "gp_rl"
submodule_config = "gp_rl/cfg"
submodule_model = "gp_rl/models"
submodule_utils = "gp_rl/utils"

setup(
    name=package_gazebo,
    version="0.0.0",
    packages=[
        package_gazebo,
        package_control,
        submodule_config,
        submodule_model,
        submodule_utils,
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_gazebo]),
        ("share/" + package_gazebo, ["package.xml"]),
        (os.path.join("share", package_gazebo, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_gazebo, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_gazebo, "data"), glob("data/*.pkl")),
        (os.path.join("share", package_gazebo, "results/gp"), glob("results/gp/*.pkl")),
        (
            os.path.join("share", package_gazebo, "results/controller"),
            glob("results/controller/*.pkl"),
        ),
        (
            os.path.join("share", package_gazebo, "models"),
            glob("gp_rl/models/*.py"),
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
