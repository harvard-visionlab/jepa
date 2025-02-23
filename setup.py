# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from setuptools import setup, find_packages

VERSION = "0.0.1"

def get_requirements():
    with open("./requirements.txt") as reqsf:
        reqs = reqsf.readlines()
    return reqs


if __name__ == "__main__":
    setup(
        name="jepa",
        version=VERSION,
        description="JEPA research code.",
        python_requires=">=3.9",
        packages=find_packages(),
        # install_requires=get_requirements(),
    )
