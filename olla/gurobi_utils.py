# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gurobipy
import os
import logging


def get_gurobi_env():
    env = gurobipy.Env()

    return env
