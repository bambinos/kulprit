import os

from setuptools import find_packages, setup

REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


setup(
    name="pyprojpred",
    packages=find_packages(),
    version="0.1.0",
    description="projection predictive variable selection",
    author="Yann McLatchie",
    license="MIT",
    install_requires=get_requirements(),
)
