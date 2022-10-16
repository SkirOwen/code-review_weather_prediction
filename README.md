## Installation

### Anaconda
[OUT OF DATE] install the library in the requirements.txt with `conda` (recommended) or `pip`  

Only tested with `python=3.8`

```shell
conda env create -f environment.yml
conda activate irp
```
Verify that the new environment was installed correctly
```shell
conda env list
# or
conda info -envs
```

#### Dependencies

See `requirements.txt`,  
`tensorboar` to see the logs in run  
`pynio` is optional and has strong dependencies

#### Tensorbord

Simply run:
```shell
tensorboard --logdir=runs
```
