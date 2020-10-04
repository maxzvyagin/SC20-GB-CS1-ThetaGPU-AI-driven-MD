#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import subprocess
import shutil
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.config import read_configuration

setup_cfg = Path(__file__).parent.joinpath("setup.cfg")
conf_dict = read_configuration(setup_cfg)
setup()
