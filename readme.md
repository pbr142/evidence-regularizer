# Evidence-based Regularization of Neural Network



## Running the code

The code was developed and executed on Ubuntu 20.04 running via WSL2 on a Windows 10 machine.

### Python Environment set-up

The environment is defined via conda. To install the required packages, run

```
conda env create -f environment.yml
```

and confirm the installation when prompted. Once completed, activate the environment with

```
conda activate evidence
```

#### CUDA
If you have a CUDA-capable GPU and want to utilize it for the experiments, run:

```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
and then deactivate and reactivate the environment such that the `LD_LIBRARY_PATH` gets set.

To test that the CUDA installation has worked, run

```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If everything works, then the last line of the output should be a list showing your GPU(s). There may be some warning methods appearing before. These can be ignored.