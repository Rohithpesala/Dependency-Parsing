# Dependency-Parsing
This project was done over the summer 2017 as a part of IESL lab at UMass Amherst. I thank Luke Vilnis, Emma Strubell and Andrew McCallum for their guidance with the project.
## Overview
The main objective of the project was to develop a good parser that could be used to parse PubMed data. My work in the project included understanding the data, acquiring it required formats and try different parsers and see their performance on out of domain data. The latter point is very important because we do not possess any labeled data of PubMed and so we were aiming to develop parsers that would be flexible over different domains. 
* The Summary of the project is present [here](https://docs.google.com/document/d/1xsH7Y7tWuy1zSomEMrPNCmwHKaEFz3WUepXVonsejD8/edit?usp=sharing). You can also find more links in the document.
## Code details
In this section, I will mainly highlight the different files present and their usage.
### Code
In this folder, we have the main files that are used for running and generating word embeddings. All the non python files in this folder are just different Gensim embedding models.
* main.py - Used to train the model.
* eval.py - Used to evaluate the learned model on dev or test data.
* model_generate.py - Used to generate word embeddings
* model_transformation.py - Contains the MLP that was used to non-linearly transform the word embeddings from one space to the other.
### Data
The train, dev and test files are present in the Code/data folder. If you want to train the parser using different dataset, please replace these files or change the variables in the constants.py file in Code/gtnlplib folder.
