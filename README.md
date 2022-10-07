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

### Supplementary of GNNSynergy
It is a supplementary materials of our GNNSynergy.
