from setuptools import find_packages, setup
import glob
import os

package_name = 'cashier_voice'

resource_files = []
if os.path.isdir('resource'):
    resource_files = [
        f for f in glob.glob('resource/*')
        if os.path.isfile(f) and os.path.basename(f) != package_name
    ]

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

if resource_files:
    data_files.append(
        ('share/' + package_name + '/resource', resource_files)
    )

launch_files = glob.glob('launch/*')
if launch_files:
    data_files.append(
        ('share/' + package_name + '/launch', launch_files)
    )

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey4090',
    maintainer_email='rokey4090@todo.todo',
    description='cashier voice package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'get_keyword = voice_processing.get_keyword:main',
            'dummy_voice_client = voice_processing.dummy_voice_client:main',
            'test_cancel_helper = voice_processing.test_cancel_helper:main',
        ],
    },
)