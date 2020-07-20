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

load the embedding model

```python
emb_model = load_model(
	model_structure_json = 'emb_model.json',
	model_weight_file = 'emb_model.h5py')
```

create the sequences data

```python
sequence_0 = {'x_time':['t1','t2'], 'x_location':['l1','l2']}
sequence_1 = {'x_time':['t1','t2'], 'x_location':['l1','l2']}
sequence_2 = {'x_time':['t1','t3'], 'x_location':['l1','l3']}
sequence_3 = {'x_time':['t3','t4'], 'x_location':['l3','l4']}
```

embed them

```python
vector_0 = sequence_embedding(
	sequence_x = sequence_0,
	x_input_data_format = x_input_data_format,
	emb_model = emb_model)

vector_1 = sequence_embedding(
	sequence_x = sequence_1,
	x_input_data_format = x_input_data_format,
	emb_model = emb_model)

vector_2 = sequence_embedding(
	sequence_x = sequence_2,
	x_input_data_format = x_input_data_format,
	emb_model = emb_model)

vector_3 = sequence_embedding(
	sequence_x = sequence_3,
	x_input_data_format = x_input_data_format,
	emb_model = emb_model)
```

see how the embedding vector looks like

```python
>>> print(vector_0)
[0.0, 0.0, 0.1853618025779724, 0.0, 0.0, 0.0, 0.0, 0.10898569226264954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1487523913383484, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16203799843788147, 0.0, 0.0, 0.01120348647236824, 0.09584314376115799, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10410629212856293, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22039426863193512, 0.1625249683856964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16476482152938843, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06516748666763306, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25179678201675415, 0.0, 0.0, 0.0, 0.18329527974128723, 0.0, 0.0, 0.0, 0.0, 0.04614228010177612, 0.0, 0.0, 0.16278885304927826, 0.19747699797153473, 0.0, 0.0, 0.0, 0.1725696325302124, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16965776681900024, 0.12053556740283966, 0.0, 0.0, 0.0, 0.1700238138437271, 0.03461480885744095, 0.0, 0.1646643579006195, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07228746265172958, 0.0, 0.0, 0.0, 0.0, 0.1798122078180313, 0.014479868113994598, 0.0, 0.1318647712469101, 0.0, 0.0, 0.06298117339611053, 0.0, 0.0, 0.0, 0.0, 0.008996419608592987, 0.0, 0.0, 0.0, 0.0, 0.025638950988650322, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.055972903966903687, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05546148121356964, 0.0, 0.0, 0.0, 0.1411837935447693, 0.0, 0.24707868695259094, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2102406620979309, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12604397535324097, 0.0, 0.13420656323432922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20228546857833862, 0.0, 0.0, 0.0, 0.0, 0.03042147308588028, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13864339888095856, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13164456188678741, 0.0, 0.0, 0.0, 0.13826563954353333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14446806907653809, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15242840349674225, 0.0, 0.09900598973035812, 0.013653138652443886, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12170442938804626, 0.014936961233615875, 0.0, 0.0, 0.015092458575963974, 0.023580260574817657, 0.0, 0.0, 0.031032908707857132, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18934205174446106, 0.0, 0.1436716765165329, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.018671751022338867, 0.28708869218826294, 0.0, 0.12969857454299927, 0.0, 0.043105967342853546, 0.0, 0.18794995546340942, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12536540627479553, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.009568706154823303, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18776750564575195, 0.0, 0.017977476119995117, 0.0, 0.0, 0.1341341882944107, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11626799404621124, 0.0, 0.0, 0.13758361339569092, 0.0, 0.0, 0.0, 0.0, 0.25326284766197205, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18279488384723663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15976537764072418, 0.13664406538009644, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19157098233699799, 0.08895820379257202, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20508432388305664, 0.199848011136055, 0.0, 0.15503081679344177, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20220881700515747, 0.006289160810410976, 0.0, 0.0, 0.0, 0.0898699089884758, 0.0, 0.0, 0.0, 0.2646111249923706, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00997009128332138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7881393432617188e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09810349345207214, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005959514528512955, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1589651107788086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1815209984779358, 0.0, 0.0, 0.0, 0.0, 0.1482108235359192, 0.0, 0.0, 0.0, 0.0, 0.08812837302684784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.054182082414627075, 0.0, 0.07529880106449127, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1139359325170517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.009179752320051193, 0.0, 0.0]
>>> 
```

calculate the inner product of the vectors

```python
np.inner(np.array(vector_0), np.array(vector_1))
np.inner(np.array(vector_0), np.array(vector_2))
np.inner(np.array(vector_0), np.array(vector_3))
```

see the outputs

```python
>>> np.inner(np.array(vector_0), np.array(vector_1))
1.7355304520201214
>>> np.inner(np.array(vector_0), np.array(vector_2))
0.9617815310324274
>>> np.inner(np.array(vector_0), np.array(vector_3))
0.03791223773242554
```

## TODO

online updating of the similarity model 

## Contact

I am actively looking for data science/AI related job. If you have such oppertunity, thank you so much for contact me. I am ready for interview any time. My email is gaoyuanliang@outlook.com 
