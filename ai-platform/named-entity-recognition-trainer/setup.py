from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'Keras==2.2.4',
    'cloudml-hypertune'
]

setup(
    name='',
    version='',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=''
)
