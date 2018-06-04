# Chemical_Property_Classification (12/05/2017-12/17/2017)
## Description:  
In the first project, we have trained our machine to detect the good signals from MS data from a single cell. It is found that even these good signals come with different shapes. It is noticed that some signals show very unique shape and consistent for that signal from different cell samples. Below shows one such example: S-adenosyl-methionine (SAM) from 7 samples show very similar signal shape. Is there any relation between the signal shape of the metabolites and their chemical properties?  
Extended Connectivity Fingerprints (ECFPs) are circular topological fingerprints designed for molecular characterization. Here we use ECFP as the representation of chemical properties and the ECFPs (128 bits vector) of 32 different signals will be provided, as well as the 32 signal shape images (60*12, same as project 1) from 7 cell samples.  
## Project:  
![Imgur](https://i.imgur.com/uZHKwqO.png)
Code File: Chemical_property.py
--Decoder checked the performance of encoder model  
--Extracted latent feature to feed a 4-layer MLP model to classify 32 signal categories  
--Cross Validation Test Acc: 80.77%  
## Conclusion  
The final Test Accuracy is 80.77% and it can be better if we have more training set.  
(1) We tried Euclidean Distance to compute the accuracy between latent feature and ECFP(worest!!)  
(2) Then we tried MLP to classify category (It looks better!)
## Future work
If time available, we expect to modify model like below, which has a better functionality of image feature selection
Model: Setp 1: raw data --> CNN --> Encoder --> Latent features--> Decoder --> Transposed CNN
       Step 2: Latent features --> MLP 
           

