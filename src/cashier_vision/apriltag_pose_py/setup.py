from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'apriltag_pose_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daehyuk',
    maintainer_email='eogur1472@gmail.com',
    description='Python nodes for AprilTag and YOLO integration',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'apriltag_pose_node = apriltag_pose_py.apriltag_pose_node:main',
            'vision_scan_items_action_server = apriltag_pose_py.vision_scan_items_action_server:main',
        ],
    },
)
