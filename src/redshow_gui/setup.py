from setuptools import setup
import os
from glob import glob

package_name = 'redshow_gui'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'PySide6',
        'torch',
        'matplotlib>=3.6.0',  # PySide6 호환성을 위해 3.6.0 이상 필요
        'numpy',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Real-time monitoring GUI for Berkeley Humanoid robot using PyQt5 and ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'redshow_gui = redshow_gui.gui_node:main',
        ],
    },
)

