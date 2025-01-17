# CodeLicensingExploration

This repository contains the reproduction and extra material for the paper:

***An Exploratory Investigation into Code License Infringements in Large Language Model Training Datasets***

Presented at FORGE 2024



## Tertiary study

The list of papers and the raw data extracted from the tertiary study can be found in the file [Tertiary_raw.csv](Tertiary_raw.csv).

The models extracted from the investigation, including references to their respective papers, which couldn't be fully included in the research paper due to space constraints, can be found in the file [models.md](models.md).



## Copyleft Dataset

We provide a list of all repositories that we downloaded to create the dataset of copyleft files in the folder [strong_copyleft](strong_copyleft).

We include the date we accessed each file to assist in reproducing our results.



## Code

The code that was used to extract comments, look for licenses, and cleaning PII from the comments dataset can be found in the folder [src](src).



## Dataset

We uploaded the dataset of leading comments to [huggingface](https://huggingface.co/datasets/AISE-TUDelft/leading-comments).


## Citation 

@inproceedings{katzy2024exploratory,
  title={An Exploratory Investigation into Code License Infringements in Large Language Model Training Datasets},
  author={Katzy, Jonathan and Popescu, Razvan-Mihai and van Deursen, Arie and Izadi, Maliheh},
  booktitle={FORGE ’24: 1st International Conference on AI foundation models and software engineering},
  pages={},
  year={2024},
  organization={ACM}
}
