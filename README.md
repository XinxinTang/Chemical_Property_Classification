# Chemical_Property_Classification (12/05/2017-12/17/2017)
## Description:

In the first project, we have trained our machine to detect the good signals from MS data from a single cell. It is found that even these good signals come with different shapes. It is noticed that some signals show very unique shape and consistent for that signal from different cell samples. Below shows one such example: S-adenosyl-methionine (SAM) from 7 samples show very similar signal shape. Is there any relation between the signal shape of the metabolites and their chemical properties?

Extended Connectivity Fingerprints (ECFPs) are circular topological fingerprints designed for molecular characterization. Here we use ECFP as the representation of chemical properties and the ECFPs (128 bits vector) of 32 different signals will be provided, as well as the 32 signal shape images (60*12, same as project 1) from 7 cell samples.

## Project:

file: Chemical_property.py
--Decoder checks the performance of encoder model

--Use latent feature feed a 4-layer MLP model to classify 32 signal categories

--Acc: 80.77%

## Conclusion

This is course-based unsupervised learning project. The dataset posted on Dec 05, 2017 and The day of presentation is Dec 17.

(1) I tried to use Euclidean Distance to compute the accuracy between latent feature and ECFP(worest!!)

(2) I tried to use MLP to classify category

## Future work
Try model: raw data --> CNN --> Encoder --> Latent features--> Decoder --> Transposed CNN 
           Latent features --> MLP 
           
This is a rough idea thought out during doing the project. Maybe there are more well-performed model will be found in the future. 

