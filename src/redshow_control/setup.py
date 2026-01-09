from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'redshow_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='pi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'redshow_control_node = redshow_control.redshow_control_node:main',
            'bno085_node = redshow_control.bno085_test_node:main',
            'bno085_test_node = redshow_control.bno085_test_node:main',  # 하위 호환성 유지
            'test_feedback_publisher = redshow_control.test_feedback_publisher:main',
            'test_adaptation_module_publisher = redshow_control.test_adaptation_module_publisher:main',
        ],
    },
)
