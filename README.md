# Deep embedding of sequence and similarity comparision

Behaviour embedding, categorization, regression, and similarity comparision

## Training the similarity deep model

### Preparing the training data

to use the deep embedding and similarity comparision functions, firstly import the packages

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

check the model prediction results over the training set

```python
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

you will see how the input tabel looks like

```python
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

here since we want to compare the similarity of two sequence, we have the columns of the first sequence, x_location and x_time, and the columns of the seconde sequence, y_location and y_time, and their similarity column, label. Each row is a pair of sequences, x and y.

Then we conver the data tabel into npy files to fit to the deep learning input formats. For the sequence data, each timestamp is composed of two attributes, time and location, so we want to concatnate their embeddings for the convolutional layers. To this end, set the cnn layers parameters as 

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
	output_npy_file_name_prefix = 'trip',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)
  
  ```
  
  the training data attributs looks like 
  
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

train the similarity model 

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
