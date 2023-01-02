# Testing-Machine-Learning-Models-on-the-California-Housing-Dataset

There are numerous machine learning models that help us make the predictions required for dealing with datasets. 

In this code, I have tried to implement different machine learning models on the California Housing Dataset in order to determine how accurately different machine learning models are capable of determining the price of a real estate properties on the basis of different parameters. 

In this code, you will find me using 4 different machine learning models each with their own accuracy score. I have divided the original dataset into two parts: Training and Test Dataset with a 20% divide ration. This means that 80% of the original data will be used to train the model while the remaining 20% will be used to test the trained model. 

     a. The Ridge() model: The Ridge model retuns an accuracy of about 57%. 
     b. The Linear Lasso() model: The Linear Lasso model returns a accuracy of 53%. 
     c. The ElasticNet Model(): The ElasticNet model returns a accuracy of 14% which is the worst of the four models that I have tested. 
     d. The Ensemble Regressors model: The Ensemble Regressors model perform the best for the California Housing dataset, and it returns an accuracy of over 85%. 
    

It is worth noting that I have not tuned the hyperparameters in this code, and I have relied solely on the default values of the dataset, and the default paramemetrs of the training models. 
