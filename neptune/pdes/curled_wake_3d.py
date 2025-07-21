import jax.numpy as jnp
import jax
import typing
from typing import Dict
from functools import partial

from neptune.geometry import Function

from .abstract_pde import AbstractPDE
from ..geometry import SparseGrid, Function
from ..geometry import mesh_utils as mu
from ..types import ModelInput

class CurledWake3D(AbstractPDE): 
    pass