**About**

Code to reproduce experiments and results for the paper titled "A Frequency Analysis of Filterbank Initialisation and Noise Augmentation for LEAF", currently under review at Scientific Reports. The implementation is based on [[autrainer](https://autrainer.github.io/autrainer/)] version `0.3.0`


**Installation**

Install the provided requirements file, which includes the autrainer package (preferrably in a virtual environment).

```
pip install -r requirements.txt
```

or simply

```
pip install autrainer==0.3.0
```

**Data Availability**
The datasets DCASE2020Task1a, DCASE2018Task3 and SpeechCommands are downloaded directly within this library by following the reproduction steps below. For access to the AIBO dataset please contact [manuel.milling@tum.de](mailto:manuel.milling@tum.de).

**Experiment Reproduction**

The deep learning experimental setup is defined in the `conf/config.yaml` and defines the training for four datasets, four initialisation types of the LEAF frontend and three random seeds. To reproduce the experiments, the following commands need to be executed, which include the download of the datasets (except for the SER Task [FAU-AIBO](https://www.isca-archive.org/interspeech_2009/schuller09_interspeech.pdf)) into the `data/` directory and the training of all necessary models. The outputs of the training including model states is saved into `results/default/`:

```
autrainer fetch
autrainer train
```

**Analysis**
The reproduction of the graphs used for the analysis of the training is based on the trained model states in the `results/default/training/` directory. To recreate the graphs in the `results/default/` and `results/noise/` directory (after completed training) execute the following python scripts:

```
python leaf_frequency_analysis.py  # filterbank analysis standard experiments
python leaf_frequency_noise_analysis.py  # filterbank analysis noise experiments
python leaf_frequency_noise_performance # performance analysis noise experiments
```
