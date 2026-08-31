from setuptools import find_packages, setup


package_name = "robot_vision"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/robot_vision"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="adam",
    maintainer_email="adam@todo.todo",
    description="ROS 2 camera inference node for laptop and mouse detection.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "yolo_detector = robot_vision.yolo_detector:main",
        ],
    },
)
