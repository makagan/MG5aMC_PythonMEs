FROM lukasheinrich/madgraph-pythia:2.6.6
RUN pip install -U pip
RUN python -m pip install  jax jaxlib
