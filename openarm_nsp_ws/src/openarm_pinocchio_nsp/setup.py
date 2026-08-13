from setuptools import setup

package_name = 'openarm_pinocchio_nsp'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ik.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenArm',
    maintainer_email='openarm@enactic.ai',
    description='Null-space-projection Pinocchio IK + offline Cartesian planning for OpenArm v1.0',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ik_node = openarm_pinocchio_nsp.ik_node:main',
            'fk = openarm_pinocchio_nsp.fk:main',
            'move_joints = openarm_pinocchio_nsp.move_joints:main',
            'home = openarm_pinocchio_nsp.home:main',
            'plan_cartesian = openarm_pinocchio_nsp.plan_cartesian:main',
            'plan_trajectory = openarm_pinocchio_nsp.plan_trajectory:main',
        ],
    },
)
