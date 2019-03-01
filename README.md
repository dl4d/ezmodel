## EZmodel : From Data to Deep Learning models (for humans ^^)

![EZ logo](./images/ezmodel.png)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/dl4d/ezmodel/blob/master/LICENSE)

### You have just found Ezmodel.

Ezmodel is a high-level Deep Learning model generation library. Written in Python and capable of running on top of [Keras](https:/github.com/keras-team/keras/). It facilitates data loading and preprocessing, keras model construction and training, model monitoring and evaluation.

Just give to Ezmodel :
- Path to your Dataset:
- Model type (classification, regression, segmentation, autoencoder)
- A Keras neural network

 ... and that's it !

Use Ezmodel if you need to easily develop Deep learning models without strong programming skills.

Read the documentation  ad [ezmodel.io] (https://ezmodel.io)

Ezmodel is compatible with : Python 3.6+


#### Ezdata object

__Parameters__ (Python _dict_ )

- name : [String] Dataset name
- type : [String] Data will be used for : "classification", "regression", "segmentation", "encoder"
- format : [String] Data format : "images" , "table"
- from : [String] Data comes from : "directory", "file", "repository", "flickr"
- path : [String] Path to the data, could be directory if "from" is set to "directory", or  "file" if "from" is set to "file"
- table_delimiter : [String] if "format" is set to "table", use the given delimiter to parse the table
- table_target_column : [List of String] if "format" is set to "table" and "type" is set to "classification", use the given column name as target of classification
- table_drop_column: [List of String] if "format" is set to "table", column name will be dropped from the table.
- table_target_column_type : [String] "number" if target column contains number, "string" is target columns contains string (this columm will then be further encoded using sklearn.LabelEncoder and a "synsets" entry will be added to the Ezdata object)
