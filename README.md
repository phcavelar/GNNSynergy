# GNNSynergy: A multi-view graph neural networkfor predicting anti-cancer drug synergy

### Requirements
1. Python == 3.8
1. Pytorch == 1.6.0
1. numpy == 1.18.4
1. scipy == 1.4.1
1. pandas == 1.0.3
### Repository Structure
- GNNSynergy/data: DrugComb dataset and DrugCombDB database
- GNNSynergy/code: our GNNSynergy model

### How to run our code
- preTrain_twoGraph.py # Train the GNNSynergy model on DrugComb dataset in Singl-View.
- finetune_twoGraph.py # Train the GNNSynergy model on DrugComb dataset in Multi-View.
- case_studey.py # The case study code.

### About the feature
- Download the chempy package according the paper of Chemopy.
- Use the data/drugSmile.txt to generate the 2D and 3D drug features.
- DrugCombDB database and DrugComb database can obtained in their opened server.

### About the python script of caseStudy_pubMed.py
- Preinstalled environment

chromedriver.exe, selenium

- Introduction

-- This script was developed to save the time of manually entering search criteria into the Pubmed search box when conducting a case study.As long as we write the search conditions and enter the correct string information, this script will simulate manual operations to help us automatically open the browser and enter the search conditions in Pubmed.And, it will record how many search records there are and record the number in the csv file.


### Supplementary of GNNSynergy
It is a supplementary materials of our GNNSynergy.
