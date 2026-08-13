from setuptools import setup

package_name = 'openarm_pinocchio_ik'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenArm',
    maintainer_email='openarm@enactic.ai',
    description='Pinocchio FK/IK + gravity compensation ROS2 node for OpenArm',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ik_node = openarm_pinocchio_ik.ik_node:main',
            'fk = openarm_pinocchio_ik.fk:main',
            'move_joints = openarm_pinocchio_ik.move_joints:main',
            'home = openarm_pinocchio_ik.home:main',
        ],
    },
)
