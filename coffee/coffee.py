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
from sympy import Symbol, sin, cos, pi,  Eq
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
        PointwiseConstraint
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

from coffee_eqn import CoffeeEquation

coffee_data = {
        't': [0.0, 33.333333333333336, 66.66666666666667, 100.0, 133.33333333333334, 166.66666666666669, 200.0, 233.33333333333334, 266.6666666666667, 300.0], 
        'T': [96.33333446492746, 89.84092342722415, 79.71453621540633, 69.50005731146942, 68.21188666641619, 53.239801352072234, 53.72053373609884, 49.70665132854304, 46.381650562276995, 42.32149651664422]
    }



@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    T0 = 100.0
    T_env = 25
    delta_T = 1
    r = 0.005
    t_learn = 250
    
    #Creating Nodes and Domain
    ce = CoffeeEquation(T_env = T_env, r = r)
    coffee_net = instantiate_arch(
        input_keys=[Key("t")],
        output_keys=[Key("T")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = ce.make_nodes() + [coffee_net.make_node(name="coffee_network")]

    t = Symbol("t")
    
    #Creating Geometry and adding constraint
    time_range = (0.0, t_learn)
    geo = Line1D(*time_range)
    
    #make domain
    coffee_domain = Domain()

    #initial condition
    # Set boundary to be only left boundary
    IC = PointwiseBoundaryConstraint(
            nodes = nodes,
            geometry = geo,
            outvar = {"T": T0},
            batch_size = cfg.batch_size.initial_T,
            parameterization = {t:0.0}
    )
    coffee_domain.add_constraint(IC,"IC")

    #Constraint relative to ode
    ode_C = PointwiseInteriorConstraint(
            nodes = nodes,
            geometry = geo,
            outvar = {"ode_T":0},
            batch_size = cfg.batch_size.interior,
            parameterization = {t: time_range}
    )   
    coffee_domain.add_constraint(ode_C,"ode_constraint")


    # Ajout de la contrainte relative aux données
    t_data = np.array(coffee_data["t"])[..., np.newaxis] # besoin d'un array 2D (N, 1)
    T_data = np.array(coffee_data["T"])[..., np.newaxis] # besoin d'un array 2D (N, 1)

    data_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"t": t_data} ,  # Points expérimentaux (temps)
        outvar={"T": T_data },  # Températures associées
        batch_size=len(coffee_data["t"]),  # Nombre de points à inclure
    )
    coffee_domain.add_constraint(data_constraint, "Data")

    
    # Setup validator
    
    t_val = np.arange(0., t_learn, delta_T)[..., np.newaxis] # besoin d'un array 2D (N, 1)
    T_val = (T0 - T_env)*np.exp(-r*t_val)+T_env
    
    validator = PointwiseValidator(
            nodes=nodes,
            invar={"t": t_val},
            true_outvar={"T": T_val},
            batch_size=128
    )
    coffee_domain.add_validator(validator)
    
    
    # Setup Inferencer
    t_infe = np.arange(0, 1000, delta_T)[..., np.newaxis]
    invar_infe = {"t": t_infe}

    grid_inference = PointwiseInferencer(
        nodes=nodes,
        invar=invar_infe,
        output_names=["T"],
        batch_size=128,
        plotter=None,
    )
    coffee_domain.add_inferencer(grid_inference, "inferencer_data")

    #make solver
    slv = Solver(cfg, coffee_domain)

    #start solve
    slv.solve()
    
if __name__ == "__main__":
    run()

































        
