################sequence_similarity_example.py##################
from jessica_behaviour_dl import *
from jessica_behaviour_spark import * 
from jessica_behaviour_local_spark_building import sqlContext

'''
data and model attributes
'''

padding_length = {'x_time':10, 'x_location':10, 'y_time':10, 'y_location':10}
vacabulary_size = {'x_time':24, 'x_location':100, 'y_time':24, 'y_location':100}
embedding_dim = {'x_time':20, 'x_location':300, 'y_time':20, 'y_location':300}

cnn_layers = [
['x_time', 'x_location'],
['y_time', 'y_location'],
]

x_behaviour_attributes = ['x_time', 'x_location']
y_behaviour_attributes = ['y_time', 'y_location']


'''
create data file and convert data to npy files
'''

sqlContext.createDataFrame([
('0', ['t1','t2'],['l1','l2'], ['t1','t2'],['l1','l2'], [2.0]),
('1', ['t1','t3'],['l1','l3'], ['t1','t3'],['l1','l3'], [2.0]),
('2', ['t3','t4'],['l3','l4'], ['t3','t4'],['l3','l4'], [2.0]),
('3', ['t1','t2'],['l1','l2'], ['t1','t3'],['l1','l3'], [1.0]),
('4', ['t1','t3'],['l1','l3'], ['t3','t4'],['l3','l4'], [1.0]),
('5', ['t4'],['l4'], ['t4'],['l4'], [1.0]),
('6', ['t1','t2'],['l1','l2'], ['t3','t4'],['l3','l4'], [0.0]),
],
['document_id','x_time', 'x_location', 'y_time', 'y_location','label']).write.mode('Overwrite').json('training1.json')

sqlContext.read.json('training1.json').show()

'''
convert data to npy
'''

training1_data = behaviour_json2npy(
	input_json = 'training1.json',
	output_npy_file_name_prefix = 'training1',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

'''
training the initial similarity model
'''

model, x_input_data_format, y_input_data_format = train_behaviour_similary_model(
	training1_data,
	x_behaviour_attributes,
	y_behaviour_attributes,
	cnn_layers,
	sqlContext,
	epochs = 500,
	gpus = None,
	batch_size = 3,
	model_weight_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	prediction_json = 'prediction1.json',
	dropout_rate = 0.3,
	cnn_layer_num = 1,
	training_type = 'initial',
	verbose = 1)

sqlContext.read.json('prediction1.json').show(100, False)

'''
update the model with more traning data
'''

sqlContext.createDataFrame([
('7', ['t4','t5'],['l4','l5'], ['t1','t2'],['l1','l2'], [0.0]),
('8', ['t4','t5'],['l4','l5'], ['t4','t3'],['l4','l3'], [1.0]),
('9', ['t6','t4'],['l6','l4'], ['t6'],['l6'], [1.0]),
('10', ['t1','t2'],['l1','l2'], ['t4','t5'],['l4','l5'], [0.0]),
],
['document_id','x_time', 'x_location', 'y_time', 'y_location','label']).write.mode('Overwrite').json('training2.json')

sqlContext.read.json('training2.json').show()

training2_data = behaviour_json2npy(
	input_json = 'training2.json',
	output_npy_file_name_prefix = 'training2',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

model, x_input_data_format, y_input_data_format = train_behaviour_similary_model(
	training2_data,
	x_behaviour_attributes,
	y_behaviour_attributes,
	cnn_layers,
	sqlContext,
	epochs = 500,
	gpus = None,
	batch_size = 3,
	model_weight_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	prediction_json = 'prediction2.json',
	dropout_rate = 0.3,
	cnn_layer_num = 1,
	verbose = 1,
	training_type = 'update',
	x_input_data_format = x_input_data_format,
	y_input_data_format = y_input_data_format)

sqlContext.read.json('prediction2.json').show()

'''
build the embedding model from the trained similarity model
'''
emb_model = building_embedding_layer_from_pretrained_model(
	model_weight_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	embedding_layer_name = 'x_behaviour_embedding_model',
	emb_model_structure_json = 'emb_model.json',
	emb_model_weight_file = 'emb_model.h5py')

'''
create the test data set for the embedding
'''
sqlContext.createDataFrame([
('0', ['t1','t2'],['l1','l2']),
('1', ['t1','t3'],['l1','l3']),
('2', ['t1','t4'],['l1','l4']),
('3', ['t2','t3'],['l2','l3']),
('4', ['t1'],['l1']),
('5', ['t4','t3'],['l4','l3']),
('6', ['t1','t2'],['l1','l2']),
('7', ['t6'],['l6']),
('8', ['t6','t3'],['l6','l3']),
],
['document_id','x_time', 'x_location']).write.mode('Overwrite').json('test.json')

sqlContext.read.json('test.json').show()

test_data = behaviour_json2npy(
	input_json = 'test.json',
	output_npy_file_name_prefix = 'test',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

'''
building the input data from the test data npy and input format
'''
x = building_x_from_input_dataformat_and_npy(
	input_format = x_input_data_format,
	input_data_attributes = test_data)

'''
embdding
'''
y_vector = emb_model.predict(x)

print(y_vector)


'''
calcualte similarity
'''
print(np.inner(y_vector[0], y_vector[1]))
print(np.inner(y_vector[0], y_vector[4]))
print(np.inner(y_vector[0], y_vector[5]))
print(np.inner(y_vector[0], y_vector[6]))
print(np.inner(y_vector[7], y_vector[8]))

'''
>>> print(np.inner(y_vector[0], y_vector[1]))
1.0672512
>>> print(np.inner(y_vector[0], y_vector[4]))
1.5836941
>>> print(np.inner(y_vector[0], y_vector[5]))
0.034832597
>>> print(np.inner(y_vector[0], y_vector[6]))
2.1542969
>>> print(np.inner(y_vector[7], y_vector[8]))
1.0411918
'''
#################sequence_similarity_example.py##################
