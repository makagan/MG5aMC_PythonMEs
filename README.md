# MadJAX

**MadJAX** is a JAX enable plugin to the High Energy Phsyics code [MadGraph5_aMC@NLO](https://launchpad.net/madgraph5) (v2.6.6+).
It offers a new output mode for the standalone computation of Matrix Elements of scattering amplitudes.
**MadJAX** is a fully python that relies on JAX, thus enabling differentiable, parallelizable,  and JIT-compileable Matric Element computations

## Usage

Copy this project in the `PLUGIN` folder located in the root directory of your `MG5aMC` distribution (v2.6.6+).

In the [Example Cards](data/example_cards) directory,  you can find example `MG5aMC` scripts that define the
scattering process and generate the standalone computations.

The `MG5aMC` example script `test_MG5aMC_PythonMEs.mg5` can then simply be run as follows (from within the root directory of `MG5aMC`):
```
./bin/mg5_aMC --mode=MG5aMC_PythonMEs PLUGIN/MG5aMC_PythonMEs/test_MG5aMC_PythonMEs.mg5
```

The Python+JAX code for this example selection of Matrix Elements will be generated in the folder `<MG5aMC_root_dir>/MG5aMC_PythonMEs_output_example`
and the script at `<MG5aMC_root_dir>/MG5aMC_PythonMEs_output_example/check_sa.py` will be run to test the standalone code.

Within `check_sa.py`you can see examples of how to generate phase space points, call the matrix element, and how to define a differentiable and JIT-able function to call the matrix element scattering process.

MadJAX will also generate a `processes` folder which defines the SMatrix computation for the process, a `model` folder with the underlying wavefunction computations, and a `phase_space_generator` folder that performs the N-body phase space sampling.

## Further Development

Note that for now the independent parameters of the model are hard-coded to their default value in the `parameters.py` script. Eventually we may want to add a facility for reading an SLHA input card, but this is not needed for now.

## Docker

```
docker run --rm -it -v $PWD/MG5aMC_PythonMEs:/code/madgraph/PLUGIN/MG5aMC_PythonMEs -v $PWD/data:/data -w /data lukasheinrich/diffmes bash
$> /code/madgraph/bin/mg5_aMC --mode=MG5aMC_PythonMEs example_cards/test_MG5aMC_PythonMEs.mg5
```
