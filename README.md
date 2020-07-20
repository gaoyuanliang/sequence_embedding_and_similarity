# Deep sequence embedding for sequence similarity comparison

Training deep learning models for the purpose of sequence similarity comparison, and then use the embedding layers of the similarity model to represent the sequences

<img src="https://github.com/gaoyuanliang/sequence_embedding_and_similarity/raw/master/WeChat%20Screenshot_20200717164438.png" width="600">

## Installation 

```bash
git clone https://github.com/gaoyuanliang/sequence_embedding_and_similarity.git

cd sequence_embedding_and_similarity

pip3 install -r requirements.txt
```

## Training the similarity deep model

### Preparing the training data

to use the deep embedding and similarity comparison functions, firstly import the packages

```python

from jessica_behaviour_dl import *
from jessica_behaviour_spark import * 
from jessica_behaviour_local_spark_building import sqlContext

```

then create the similarity training data and save to a json folder


```python

sqlContext.createDataFrame([
('7335', ['t1','t2'],['l1','l2'], ['t1','t2'],['l1','l2'], [2.0]),
('5897', ['t1','t3'],['l1','l3'], ['t1','t3'],['l1','l3'], [2.0]),
('1234', ['t3','t4'],['l3','l4'], ['t3','t4'],['l3','l4'], [2.0]),
('6789', ['t1','t2'],['l1','l2'], ['t1','t3'],['l1','l3'], [1.0]),
('6895', ['t1','t3'],['l1','l3'], ['t3','t4'],['l3','l4'], [1.0]),
('6895', ['t4'],['l4'], ['t4'],['l4'], [1.0]),
('3456', ['t1','t2'],['l1','l2'], ['t3','t4'],['l3','l4'], [0.0]),
],
['document_id','x_time', 'x_location', 'y_time', 'y_location','label']).write.mode('Overwrite').json('example.json')

sqlContext.read.json('example.json').show()
```

you will see how the input table looks like

```
+-----------+-----+----------+--------+----------+--------+
|document_id|label|x_location|  x_time|y_location|  y_time|
+-----------+-----+----------+--------+----------+--------+
|       7335|[2.0]|  [l1, l2]|[t1, t2]|  [l1, l2]|[t1, t2]|
|       5897|[2.0]|  [l1, l3]|[t1, t3]|  [l1, l3]|[t1, t3]|
|       1234|[2.0]|  [l3, l4]|[t3, t4]|  [l3, l4]|[t3, t4]|
|       6789|[1.0]|  [l1, l2]|[t1, t2]|  [l1, l3]|[t1, t3]|
|       6895|[1.0]|  [l1, l3]|[t1, t3]|  [l3, l4]|[t3, t4]|
|       6895|[1.0]|      [l4]|    [t4]|      [l4]|    [t4]|
|       3456|[0.0]|  [l1, l2]|[t1, t2]|  [l3, l4]|[t3, t4]|
+-----------+-----+----------+--------+----------+--------+
```

here since we want to compare the similarity of two sequences, we have the columns of the first sequence, x_location and x_time, and the columns of the second sequence, y_location and y_time, and their similarity column, label. Each row is a pair of sequences, x and y.

Then we convert the data table into npy files to fit the deep learning input formats. For the sequence data, each timestamp is composed of two attributes, time and location, so we want to concatenate their embeddings for the convolutional layers. To this end, set the CNN layers parameters as 

```python 

cnn_layers = [
['x_time', 'x_location'],
['y_time', 'y_location'],
]
```

also set the embedding parameters of each attribute, and do the conversion

```python

padding_length = {'x_time':5, 'x_location':5, 'y_time':5, 'y_location':5}
vacabulary_size = {'x_time':24, 'x_location':100, 'y_time':24, 'y_location':100}
embedding_dim = {'x_time':20, 'x_location':300, 'y_time':20, 'y_location':300}

training_data = behaviour_json2npy(
	input_json = 'example.json',
	output_npy_file_name_prefix = 'x',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)
  
  ```
  
the training data attributes looks like 
  
  ```python
 
 training_data = [
{'atrribute_name': 'document_id', 'npy_file': 'trip_document_id.npy'}, 
{'atrribute_name': 'label', 'npy_file': 'trip_label.npy'}, 
{'vacabulary_size': 100, 'atrribute_name': 'x_location', 'embedding_dim': 300, 'npy_file': 'trip_x_location.npy', 'padding_length': 5}, 
{'vacabulary_size': 24, 'atrribute_name': 'x_time', 'embedding_dim': 20, 'npy_file': 'trip_x_time.npy', 'padding_length': 5}, 
{'vacabulary_size': 100, 'atrribute_name': 'y_location', 'embedding_dim': 300, 'npy_file': 'trip_y_location.npy', 'padding_length': 5}, 
{'vacabulary_size': 24, 'atrribute_name': 'y_time', 'embedding_dim': 20, 'npy_file': 'trip_y_time.npy', 'padding_length': 5}
]

```

### train the similarity model 

```python

x_behaviour_attributes = ['x_time', 'x_location']
y_behaviour_attributes = ['y_time', 'y_location']

model, x_input_data_format, y_input_data_format = train_behaviour_similary_model(
	training_data,
	x_behaviour_attributes,
	y_behaviour_attributes,
	cnn_layers,
	sqlContext,
	epochs = 1000,
	gpus = None,
	batch_size = 500,
	model_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	prediction_json = 'similary.json',
	dropout_rate = 0.3,
	cnn_layer_num = 1)

sqlContext.read.json('similary.json').show()
```

check the similarity prediction results of the training set

```
+-----------+-----+--------------------+
|document_id|label|          prediction|
+-----------+-----+--------------------+
|       7335|[2.0]|[1.9618165493011475]|
|       5897|[2.0]|[1.9188060760498047]|
|       1234|[2.0]|[2.0617730617523193]|
|       6789|[1.0]|[0.9564264416694641]|
|       6895|[1.0]|[0.9779235124588013]|
|       6895|[1.0]|[1.1057801246643066]|
|       3456|[0.0]|[0.02453222498297...|
+-----------+-----+--------------------+
```

### Testing the similarity model 

prepare the test data of pairs of sequences 

```python
sqlContext.createDataFrame([
('0', ['t1','t2'],['l1','l2'], ['t1','t3'],['l1','l3']),
('1', ['t1','t2'],['l1','l2'], ['t1'],['l1']),
('2', ['t1','t2'],['l1','l2'], ['t4','t3'],['l4','l3']),
('3', ['t1','t2'],['l1','l2'], ['t1','t2'],['l1','l2']),
('4', ['t1','t2'],['l1','l2'], ['t6','t3'],['l6','l3']),
],
['document_id','x_time', 'x_location', 'y_time', 'y_location']).write.mode('Overwrite').json('test.json')

test_data = behaviour_json2npy(
	input_json = 'test.json',
	output_npy_file_name_prefix = 'test',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)
```

predict the similarity

```python
y_similarity = predict_behaviour_similary_from_model(
	model_weight_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	test_data = test_data,
	x_input_data_format = x_input_data_format,
	y_input_data_format = y_input_data_format,
	prediction_json = 'test_prediction.json')

sqlContext.read.json('test_prediction.json').show()
```

get the prediciton results

```
+-----------+--------------------+
|document_id|          prediction|
+-----------+--------------------+
|          0|[0.9617815613746643]|
|          1|[1.4290894269943237]|
|          2|[0.03791223838925...|
|          3|[1.7355303764343262]|
|          4|[0.25222671031951...|
+-----------+--------------------+
```

this results is good because the prediction reflects the overlapping timestamp number of two sequences. Meanwhile, the test pairs are not included in the training set.  

## Building the embedding model from the trained deep similarity model

```python

emb_model = building_embedding_layer_from_pretrained_model(
	model_weight_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	embedding_layer_name = 'x_behaviour_embedding_model',
	emb_model_structure_json = 'emb_model.json',
	emb_model_weight_file = 'emb_model.h5py')
```

## Test the embedding results

building the test data set, but this time each row only have one sequence with attributes of time and location

```python
sqlContext.createDataFrame([
('0', ['t1','t2'],['l1','l2']),
('1', ['t1','t3'],['l1','l3']),
('2', ['t1','t4'],['l1','l4']),
('3', ['t2','t3'],['l2','l3']),
('4', ['t1'],['l1']),
('5', ['t4','t3'],['l4','l3']),
('6', ['t1','t2'],['l1','l2']),
],
['document_id','x_time', 'x_location']).write.mode('Overwrite').json('example1.json')
```

it looks like 

```
+-----------+----------+--------+
|document_id|x_location|  x_time|
+-----------+----------+--------+
|          0|  [l1, l2]|[t1, t2]|
|          1|  [l1, l3]|[t1, t3]|
|          2|  [l1, l4]|[t1, t4]|
|          3|  [l2, l3]|[t2, t3]|
|          4|      [l1]|    [t1]|
|          5|  [l4, l3]|[t4, t3]|
|          6|  [l1, l2]|[t1, t2]|
+-----------+----------+--------+
```

re-organize the data according to the input data format

```python
test_data = behaviour_json2npy(
	input_json = 'example1.json',
	output_npy_file_name_prefix = 'x1',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

x = building_x_from_input_dataformat_and_npy(
	input_format = x_input_data_format,
	input_data_attributes = test_data)
```

use the embedding model to convert the x to embedding vectors

```python
y_vector = emb_model.predict(x)
```


calculate the similarities

```python
>>> print(np.inner(y_vector[0], y_vector[1]))
0.9617815
>>> print(np.inner(y_vector[0], y_vector[4]))
1.4290894
>>> print(np.inner(y_vector[0], y_vector[5]))
0.037912235
>>> print(np.inner(y_vector[0], y_vector[6]))
1.7355305
>>> print(np.inner(y_vector[7], y_vector[8]))
1.0690365
```

## TODO

online updating of the similarity model 

building a api for the embdding of a sequence, lists of attributs of timestamps 

## Contact

I am actively looking for data science/AI related job. If you have such oppertunity, thank you so much for contact me. I am ready for interview any time. My email is gaoyuanliang@outlook.com 
