"""setup.py required for training a model using ML Engine."""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['numpy','sklearn','datetime','pandas','argparse','matplotlib','gcsfs']

setup(
    name='rltradingbot',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description=
    'Reinforcement learning trading bot for Hackathon')