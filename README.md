Guidelines for creating new preprocessing functions:
* instantiating an object of class PreProcess requires you to pass a pandas dataframe
* this dataframe is saved as a variable within the class
* each function does an operation that is tailored to the characterisits of a specific column
* however any column can be passed as an argument for a given function
* to create a new function, ensure that it is unique and is designed for a specific column

Guidelines for creating new feature-creating functions:
* if you need to create a function, you can add it under the FeatureCreate class in the code
* The FeatureCreate class takes a dataframe and a column name (as a string) as inputs.
* If it is the case that you need more than one column to create a new feature, then you should create a new class that takes multiple columns as inputs.
