#!/usr/bin/env python2

import random
import os
from timeit import default_timer as timer

from processes.all_processes import *
from model.parameters import ModelParameters, ParamCard
from phase_space_generator.flat_phase_space_generator import FlatInvertiblePhasespace

import jax
from jax.config import config
config.update('jax_enable_x64', True)

class Colour:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

module_name = os.path.basename(os.path.dirname(os.path.realpath( __file__ )))

all_process_classes = [{all_process_classes}]

# For now, the feature of specifying an SLHA param card to initialise
# the value of independent parameters is not supported yet.
active_model = ModelParameters(None)

# Center of mass of the collision in GeV
E_cm = 14000.

print("")
print("The module '%s' contains %d processes"%(module_name, len(all_process_classes)))
print("")
print(str(active_model))
print("")

for process_class in all_process_classes:
    
    print(">>> Running process %s%s%s"%(Colour.BLUE,process_class.__name__,Colour.END))

    # Generate a random PS point for this process
    process = process_class()
    external_masses = process.get_external_masses(active_model)

    # Ensure that E_cm offers enough twice as much energy as necessary 
    # to produce the final states
    this_process_E_cm = max( E_cm, sum(external_masses[1])*2. )

    ps_generator = FlatInvertiblePhasespace(
        external_masses[0], external_masses[1],
        beam_Es = (this_process_E_cm/2.,this_process_E_cm/2.),
        # We do not consider PDF for this standalone check
        beam_types=(0,0)
    )

    # Generate some random variables
    random_variables = [random.random() for _ in range(ps_generator.nDimPhaseSpace())]

    PS_point, jacobian = ps_generator.generateKinematics(this_process_E_cm, random_variables)
    
    print("> PS point:")
    print(PS_point)
    #print("> Matrix element evaluation : %s%.16e%s"%(Colour.GREEN,process.smatrix(PS_point, active_model),Colour.END))
    print("> Matrix element evaluation :", process.smatrix(PS_point.to_array(), active_model.params) )
    print("")
    #process.smatrix(PS_point, active_model)

    def matrix_element(c,PS_point):
        _Z_mass = c
        
        _pc = ParamCard()
        _pc.set_block_entry("mass", 23, _Z_mass) #9.118800e+01
        _active_model_params = ModelParameters(_pc).params
        
        _process = process_class()      
        return process.smatrix(PS_point, _active_model_params)

    matrix_element_jit = jax.jit(matrix_element)
    matrix_element_prime = jax.grad(matrix_element, 0)
    matrix_element_prime_jit = jax.jit(matrix_element_prime )

    print("ME:", matrix_element(9.918800e+01, PS_point.to_array()))
    start = timer()
    print("ME jit:", matrix_element_jit(9.918800e+01, PS_point.to_array()))
    end = timer()
    print("ME jit compilation + eval time:", (end - start))
    
    print("ME prime:", matrix_element_prime(9.918800e+01, PS_point.to_array()))
    start = timer()
    print("ME prime jit:", matrix_element_prime_jit(9.918800e+01, PS_point.to_array()))
    end = timer()
    print("ME primt jit compilation + eval time:", (end - start))

    # -----------------
    # Timing Test
    # -----------------
    start = timer()
    matrix_element(9.818800e+01, PS_point.to_array())
    matrix_element(9.718800e+01, PS_point.to_array())
    end = timer()
    print("ME ave time:", (end - start)/2.0 )

    start = timer()
    matrix_element_jit(9.818800e+01, PS_point.to_array())
    matrix_element_jit(9.718800e+01, PS_point.to_array())
    end = timer()
    print("ME jit ave time:", (end - start)/2.0 )

    start = timer()
    matrix_element_prime(9.818800e+01, PS_point.to_array())
    matrix_element_prime(9.718800e+01, PS_point.to_array())
    end = timer()
    print("ME prime ave time:", (end - start)/2.0 )

    start = timer()
    matrix_element_prime_jit(9.818800e+01, PS_point.to_array())
    matrix_element_prime_jit(9.718800e+01, PS_point.to_array())
    end = timer()
    print("ME prime ave time:", (end - start)/2.0 )

    
    # -----------------
    # Only eval one ME
    # -----------------
    break

# if True:
#     from processes.all_processes import Matrix_1_mupmum_epem
#     from phase_space_generator.flat_phase_space_generator import FlatInvertiblePhasespace, LorentzVectorList, LorentzVector

    
#     def matrix_element(x,c):
#         e = 90.
#         theta = x
#         Z_mass = c
        
#         pc = ParamCard()
#         pc.set_block_entry("mass", 23, Z_mass) #9.118800e+01
#         active_model = ModelParameters(pc)
    
    
#         process = Matrix_1_mupmum_epem()      
    
    
#         vectors = [
#             [e/2,0,0, e/2],
#             [e/2,0,0,-e/2],
#             [e/2.5, 0, e/2.5*jax.numpy.sin(theta), e/2*jax.numpy.cos(theta)],
#             [e/2.5, 0,-e/2.5*jax.numpy.sin(theta),-e/2*jax.numpy.cos(theta)],
#         ]
            
#         PS_point = LorentzVectorList(LorentzVector(v) for v in vectors)
#         return process.smatrix(PS_point, active_model)[0]

#     matrix_element_prime = jax.grad(matrix_element, 1)


#     print("ME", matrix_element(3.14159, 9.918800e+01))
#     print("ME again", matrix_element(3.14159, 8.018800e+01))
    
#     print("ME derivative:", matrix_element_prime( 3.14159, 9.918800e+01 ))
#     print("ME derivative again:", matrix_element_prime( 3.14159, 8.018800e+01 ))

    
#     def first_finite_differences(f, x, c):
#         eps = 1e-3
#         return jax.numpy.array([(f(x, c + eps * v) - f(x, c - eps * v)) / (2 * eps)
#                                 for v in jax.numpy.eye(len([c]))])
    
#     print( first_finite_differences(matrix_element, 3.14159, 9.918800e+01) )

    
#     def vmap_matrix_element(x_batched, c_batched):
#         return jax.vmap(matrix_element)(x_batched, c_batched)
    
#     x_batched=jax.np.array([3.14159,3.14159,3.14159])
#     c_batched=jax.np.array([9.918800e+01, 9.918800e+01, 9.918800e+01])
#     print("vmap_matrix_element", vmap_matrix_element(x_batched, c_batched))


#     matrix_element_jit = jax.jit(matrix_element)
#     matrix_element_prime_jit = jax.jit(jax.grad(matrix_element, 1))
#     vmap_matrix_element_jit = jax.jit(vmap_matrix_element)

#     print("Perfoming JIT, first eval triggers compilation")
#     print("matrix_element_jit", matrix_element_jit(3.14159, 9.918800e+01))
#     print("matrix_element_prime_jit", matrix_element_prime_jit(3.14159, 9.918800e+01))
#     print("vmap_matrix_element_jit", vmap_matrix_element_jit(x_batched, c_batched))


#     from timeit import default_timer as timer
    
#     start = timer()
#     matrix_element(3.14159, 9.918800e+01)
#     matrix_element(3.14159, 8.018800e+01)
#     end = timer()
#     print("ME time:", end - start)

#     start = timer()
#     matrix_element_prime( 3.14159, 9.918800e+01 )
#     matrix_element_prime( 3.14159, 8.018800e+01 )
#     end = timer()
#     print("ME deriv time:", end - start)

#     start = timer()
#     matrix_element_jit(3.14159, 9.918800e+01)
#     matrix_element_jit(3.14159, 8.018800e+01)
#     end = timer()
#     print("ME jit time:", end - start)

#     start = timer()
#     matrix_element_prime_jit(3.14159, 9.918800e+01)
#     matrix_element_prime_jit(3.14159, 8.018800e+01)
#     end = timer()
#     print("ME deriv jit time:", end - start)


    

    
