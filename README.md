# Simulate a rat in a watermaze using TD learning

This is a Python implementation of the **actor-critic model of a rat in a watermaze**, as presented in
[A model of hippocampally dependent navigation, using the temporal difference learning rule](https://www.ncbi.nlm.nih.gov/pubmed/10706212) (Foster _et al._, 2000). Both RMW and DMP experiments can be simulated, and several kind of plots can be produced. However, the coordinate system is not part of this implementation.

This work is part of a small project done for an introductory course to machine learning applied to neuroscience, which was given at the École normale supérieure of Paris in 2019.




## How to run simulations


### Requirements

The code targets **Python 3.7**, but should run on Python 3.5+.
It has the following dependencies:

* `numpy` (for the computations)
* `matplotlib` (for the figures)
* `tqdm` (for the progress bars)

They can be installed using `pip` (_e.g._ `pip install --user numpy matplotlib tqdm`).


### Usage

The **entry point** of the code is `main.py`. It can be ran as a script with _execute permission_ on an Unix system (if need be, you can assign it to the file by running `chmod +x main.py`).

The script accepts a few arguments, which are optional.

| Argument          | Description                                           |
|-------------------|-------------------------------------------------------|
| `-n <nb_runs>`    | Number of simulations of both experiments.            |
| `--rmw <nb_runs>` | Number of simulations of the RMW experiment.          |
| `--dmp <nb_runs>` | Number of simulations of the DMP experiment.          |
| `--no-trial-plot` | Only plot path length (_i.e._ do not plot any trial). |
| `-h`, `--help`    | Print help.                                           |

If you don't provide any, the **default behaviour** is to run both experiments 10 times (each), and to generate and save two kinds of figures:

* one figure per trial of the last run of each experiment (_i.e_ 36 figures per experiment);
* one figure of the performance of the rat (average path lengths over all runs) per experiment.


### Examples

Default behaviour:
```console
./main.py
```

Simulate both experiments 50 times:
```console
./main.py -n 50
```

Simulate the DMP experiment 20 times and skip trial figures:
```console
./main.py --rmw 0 --dmp 20 --no-trial-plot
```



## Organisation of the code

The code is mostly written in object-oriented style, and split in a few files with dedicated responsabilities.
Each module is shortly described by the following table.

| Module        | Description                                                                                          |
|---------------|------------------------------------------------------------------------------------------------------|
| `constants`   | Constants shared by all modules.                                                                     |
| `experiments` | RMW and DMP experiments. They can be ran one or more times, and the results can be saved as figures. |
| `figures`     | Figures used by the experiments. They can be created, displayed, saved and closed.                   |
| `main`        | Entry point of the script. It handles the arguments are run the right simulation(s).                 |
| `rat`         | RL model of the rat. It comprises the place cells, the Critic and the Actor.                         |
| `utilities`   | Miscellaneous utility functions.                                                                     |
| `watermaze`   | Environements for the experiments. It comprises a plateform.                                         |