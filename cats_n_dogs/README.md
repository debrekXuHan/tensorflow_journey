# README
Illustrate each files in this path.

## process_folder.py
Preprocess the dataset path and make it meet the requirement for image generator.<br>

__Before__: <br>
|--dataset <br>
&emsp;&emsp;|--train <br>
&emsp;&emsp;|--test  <br>

__After__: <br>
|--dataset <br>
&emsp;&emsp;|--train <br>
&emsp;&emsp;&emsp;&emsp;|--dog <br>
&emsp;&emsp;&emsp;&emsp;|--cat <br>
&emsp;&emsp;|--test  <br>
&emsp;&emsp;&emsp;&emsp;|--test <br>

## export_para.py
Use the pre-trained models: ResNet50, InceptionV3 and Xception, which are pre-trained from Imagenet.
Calculate the outputs of three models and save them as .h5 files. Use them as the input of fine-tuning layers. 
The dimension should be [Num, 2048] for each of the three models.

## training_model.py
Concatenate three arrays and use it as the input here. Set up the fine-tuning model and train the model using training data.
Save the model as .h5 model file and test this model on testing data.

## testing_model.py
Find a dog or cat picture from anywhere you like and rename it as test.jpg here.
This can help you classify whether it is a dog or a cat.
