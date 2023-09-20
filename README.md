# FairCAIPI
FairCAIPI extends CAIPI (Teso and Kersting, 2019) by a fairness objective. It is a novel bias mitigation
human-in-the-loop strategy that involves users by fairly correcting local explanations of ML models decisions.
## Install
FairCAIPI was developed in Python3 and only the following versions were 
tested:
`3.10.7`
`3.8.10`
FairCAIPI uses the frameworks aif360, modAL, and SHAP.

### Install Required Packages
- `pip install -r requirements.txt`

### Download Dataset

- follow the instructions in `lib/$YOUR_PYTHON_VERSION/site-packages/aif360/data/raw/german/README.md`
  - `wget -O $VENV_NAME/lib/$YOUR_PYTHON_VERSION/site-packages/aif360/data/raw/german/german.data "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"`
  - `wget -O $VENV_NAME/lib/$YOUR_PYTHON_VERSION/site-packages/aif360/data/raw/german/german.doc "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc"`

## Usage

Run FairCAIPI as follows: 
```
usage: main.py [-h] [--seed ?] [--Z Z] [--min_perc MIN_PERC] [--T T]
               {interactive,simulation,plot,default_rf,reweighed_rf} ...

Fair Interacting with Explanations

positional arguments:
  {interactive,simulation,plot,default_rf,reweighed_rf}
                        sub commands
    interactive         run interactive mode
    simulation          run simulation mode
    plot                print plots from existing csv-file
    default_rf          show evaluation results for default Random Forest
    reweighed_rf        show evaluation results for reweighed Random Forest

optional arguments:
  -h, --help            show this help message and exit
  --seed ?              seed to use for randomization
  --Z Z                 threshold under which the accuracy should not fall
                        during the interaction process for orientation
  --min_perc MIN_PERC   minimum percentage to determine contribution threshold
                        of protected attribute Shapley value
  --T T                 number of iterations
```
### Run in interactive mode
```
usage: main.py interactive [-h]

optional arguments:
  -h, --help  show this help message and exit
```

### Run in simulation mode 
```
usage: main.py simulation [-h] [--print_plot ?]

optional arguments:
  -h, --help      show this help message and exit
  --print_plot ?  print plot of data generated during simulation
```

### Print plots from CSV-file 

```
usage: main.py plot [-h] --plot_file ?

optional arguments:
  -h, --help     show this help message and exit
  --plot_file ?  path to csv-file
```

### Show evaluation results for Default Random Forest
```
usage: main.py default_rf [-h]

optional arguments:
  -h, --help  show this help message and exit
```
### Show evaluation results for Reweighed Random Forest
```
usage: main.py reweighed_rf [-h]

optional arguments:
  -h, --help  show this help message and exit
```



