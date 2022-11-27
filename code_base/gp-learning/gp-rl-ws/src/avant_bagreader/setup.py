import os
from glob import glob
from setuptools import setup

package_name = "avant_bagreader"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*'))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="phuoc101",
    maintainer_email="nguyenthuanphuoc101@protonmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bagreader_node = avant_bagreader.bagreader_sync:main",
            "visualize_data = avant_bagreader.visualize_data:main",
        ],
    },
)
