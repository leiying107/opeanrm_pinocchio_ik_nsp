from setuptools import setup
from glob import glob

package_name = 'openarm_dashboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={'': 'src'},
    package_data={package_name: ['static/*', 'v1_simple.urdf']},
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/srv', glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenArm',
    maintainer_email='openarm@enactic.ai',
    description='Real-hardware control dashboard + IK preview system for OpenArm v1.0',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hardware_dashboard = openarm_dashboard.hardware_dashboard:main',
            'ik_dashboard = openarm_dashboard.ik_dashboard:main',
            'web_panel = openarm_dashboard.web_panel:main',
        ],
    },
)
