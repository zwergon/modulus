# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from sympy import Symbol, sin, cos, pi,  Eq, exp
import torch
import modulus
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D,Point1D
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
        PointwiseBoundaryConstraint,
        PointwiseInteriorConstraint,
)
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)

from burger_eqn import BurgerEquation


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    
    xmax = 5  # Total size 
    tmax = 2  # Total time

    nx = 200  # Number of spatial grid points
    nt = 200 # Number of time points
    dx = xmax / nx
    dt = tmax / nt  # Temporal resolution
    
    
    
    #Creating Nodes and Domain
    be = BurgerEquation()
    burger_net = instantiate_arch(
        input_keys=[Key("t"), Key("x"), Key("nu")],
        output_keys=[Key("U")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = be.make_nodes() + [burger_net.make_node(name="burger_network")]

    U, x, t, nu = Symbol("U"), Symbol("x"), Symbol("t"), Symbol("nu")
    
    #Creating Geometry and adding constraint
    geo =  Line1D(0, xmax)
    
    #make domain
    burger_domain = Domain()

    #add constraint to solver
    
    ti_nu_range = {t :(0.0, tmax), nu: (0.03, 0.1)}
   
    #initial condition
    # Set boundary to be only left boundary
    IC = PointwiseInteriorConstraint(
            nodes = nodes,
            geometry = geo,
            outvar = {"U": exp(-2 * (x - 0.5 * xmax)**2) },
            batch_size = cfg.batch_size.initial_U,
            parameterization = {t:0.0, nu: (0.03, 0.1) }
    )
    burger_domain.add_constraint(IC,"IC")

    #boundary
    b1 = PointwiseBoundaryConstraint(
            nodes = nodes,
            geometry = geo,
            outvar = {"U": 0},
            batch_size = cfg.batch_size.bc,
            parameterization = ti_nu_range,
            criteria = Eq(x, 0)
    )   
    burger_domain.add_constraint(b1,"b1")

    
     #boundary
    b2 = PointwiseBoundaryConstraint(
            nodes = nodes,
            geometry = geo,
            outvar = {"U": 0},
            batch_size = cfg.batch_size.bc,
            parameterization = ti_nu_range,
            criteria = Eq(x, xmax)
    )   
    burger_domain.add_constraint(b2, "b2")
    
    
    
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_U": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=ti_nu_range,
      )
    burger_domain.add_constraint(interior, "interior")
    

    
    # Setup Inferencer
    t_infe = np.arange(0, tmax, dt)
    x_infe = np.arange(0, xmax, dx)
    X, T = np.meshgrid(x_infe, t_infe)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    Nu = np.zeros(shape=T.shape, dtype=T.dtype)
    Nu.fill(0.06)
    print(Nu.shape, np.min(Nu), np.max(Nu))
    
    invar_infe = {"t": T, "x": X, "nu": Nu}

    grid_inference = PointwiseInferencer(
        nodes=nodes,
        invar=invar_infe,
        output_names=["U"],
        batch_size=128,
        plotter=None,
    )
    burger_domain.add_inferencer(grid_inference, "inferencer_data")

    #make solver
    slv = Solver(cfg, burger_domain)

    #start solve
    slv.solve()
    
if __name__ == "__main__":
    run()

































        
