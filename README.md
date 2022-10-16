## Installation

### Anaconda

Create a virtual env  
```shell
conda create -n weather python=3.10
```

Install the dependencies from the `requirements.txt`

```shell
conda install -c conda-forge --file requirements.txt
```

## Datasets

Create a folder `datasets`
Extract the archive `demo.rar` in the folder `datasets`  
You should have this structure:
```
.
├── datasets
│ 	├── cmorph
│ 	└── ecmwf
└── weather
    ├── tests
    ├── utils
    └── weather_object
```