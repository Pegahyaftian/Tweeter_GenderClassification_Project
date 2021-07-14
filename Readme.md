README for Tweeter Gender Classification
----------------

Our project relies mainly on numpy, matplotlib, sklearn, pandas, PyTorch, Skorch, Gensim, nltk

./environment.yaml	<-- conda environment for the project
./data_process.py	<-- contains the preprocessor class which is responsible for data loading,preprocessing,saving data
./trainer.py		<-- contains 3 main classes Trainer, VotingTrainer and SkorchTrainer which are responsible for training
			    sklearn models, voting model and torch models
./evaluator.py		<-- contains Evaluator class which evaluates models based on given data
./preprocess_app.py	<-- script to transform raw data into transformed data
./train_app.py		<-- script to train all models
./evaluate_app.py	<-- script to evaluate models based on latest dataset used to train and test models
./voting.py		<-- contains a modified version of sklearn.ensemble.VotingClassifier which works with pretrained models
./model_optimizer.py	<-- contains a ModelSelector class which applies hyperparameter search for a given model
./models.py		<-- contains the declaration of all torch models used in the project
./gui.py		<-- script to load a gui to predict a custom text with a selected model or evaluate a model on test data

./data/*.csv		<-- contains all csv data produced in the project
./figs/*.png		<-- figures created by notebooks or evaluate_app script
./models/*.pkl		<-- contains models which where created after training
./nlp_models/d2v.pkl	<-- Doc2Vec object which embeds the sentences
./nlp_models/encoder.pkl	<-- LabelEncoder to convert string classes into numerical values and vice versa

./Preprocess.ipynb	<-- Preprocessing step by step operations which is used in data_process.py
./Training.ipynb	<-- Training step by step operations with elapsed time and results which were used in trainer.py
./Training_Torch.ipynb	<-- Training operations on torch models using SkorchTrainer class
./Evaluation.ipynb	<-- Contains all model evaluation process,report and figures 


* The files should be run in the order:
	preprocess_app.py
	train_app.py
	evaluate_app.py
running these files automatically follow preprocessing, training and evaluating the models. However, it takes a lot of time. It is recommended to view Training.ipynb and Training_Torch.ipynb notebooks to view the results. In train_app.py models are trained on
sample parameters as a showcase. To follow the actual process of the application remove the parameters of trainer.train_*.(params)

* Train time of each model is written in the Training.ipynb and Training_Torch.ipynb notebooks. On average each model takes about 1 hour on a 12 core computer system.
* GPU is recommended for training torch models (SkorchTrainer)
* running gui.py opens a graphic interface. On the left side there is a list of models which are located in './models/' directory. on the right side there is a text box which you can write a custom text in it. Select a model by clicking on one of the items in the model list. Press Evaluate to show the Report and Confusion matrix figure of the model on latest dataset. Press Predict button to make the model predict the class of the text written in the textbox.


NLP pipline process was inspired by this tutorial and links:
https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8
https://stackoverflow.com/questions/24399820/expression-to-remove-url-links-from-twitter-tweet
https://www.nltk.org/book/ch03.html
https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

VotingClassifier located in voting.py:
https://gist.github.com/tomquisel/a421235422fdf6b51ec2ccc5e3dee1b4

CBR_Network located in models.py is inspired by this link:
https://www.kaggle.com/juiyangchang/cnn-with-pytorch-0-995-accuracy