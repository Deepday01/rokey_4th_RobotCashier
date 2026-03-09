import os
from glob import glob
from setuptools import setup

package_name = 'cashier_workflow'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deepday',
    maintainer_email='mhi1248@naver.com',
    description='Workflow node for cashier MVP',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'workflow_node = cashier_workflow.workflow_node:main',
            'demo_backend_node = cashier_workflow.demo_backend_node:main',
            'demo_voice_node = cashier_workflow.demo_voice_node:main',
            'demo_vision_node = cashier_workflow.demo_vision_node:main',
            'demo_plan_packing_node = cashier_workflow.demo_plan_packing_node:main',
            'demo_execute_packing_node = cashier_workflow.demo_execute_packing_node:main',
        ],
    },
)