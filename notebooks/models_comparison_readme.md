# LHCB Tracking models comparison

## Section 1: Jupyter Notebooks
In this section, the different Jupyter notebooks and their functionalities will be described.

### Notebook: Acceptance validation
The predicted labels of different models are superimposed on the same plots, to compare deviation from the true distributions.

### Notebook: Efficiency validation
Barplots shows different model's number side by side, while histograms of number of particles for each track type shows the difference of the predicted bin content w.r.t. the true distribution

### Notebook: Resolution validation
The first set of plots shows the pulls between the models predictions and the true distribution.
The Kolmogorov-Smirnov distributions are evaluated for each models separately, and then the distribution of KS in different bins is shown for each model (resulting in a bit overcrowded plots)

### Notebook: Covariance validation
The first set of plots shows the pulls between the models predictions and the true distribution.
The Kolmogorov-Smirnov distributions are evaluated for each models separately, and then the distribution of KS in different bins is shown as difference between each model and a reference model, chosen by the *target dataset* variable.

## Section 2: Snakefile Execution
This section illustrates how to use a Snakefile to automate the execution of Jupyter notebooks in a reproducible and scalable manner.

### Running the Snakefile
The snakefile `Snakefile_comparison` load a configuration YAML file (default name: `train_conf.yaml`) which should contain in a key named `training_folder`, the dictionary that identifies the different trained models home path.
When the snakefile is execute, the user can specify a *target dataset*, on which apply the different models, and this is done by the definition of the `tset` key in the config file at the time of execution (or can be written in the config file itself):
 ```bash
 snakemake -s Snakefile_comparison --config tset=j100
 ```
 Command line arguments overwrite config statements in the snakefile if the key is already defined, otherwise the two method results in a combined *config* dictionary
 ```bash
 snakemake -s Snakefile_comparison --configfile other_conf.yaml --config tset=2016MU
 ```