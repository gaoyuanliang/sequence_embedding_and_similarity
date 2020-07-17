##################sequence_similarity_example.py##################
'''
example of similarity

git clone https://github.com/gaoyuanliang/jessica_behavour.git
'''

from jessica_behaviour_dl import *
from jessica_behaviour_spark import * 
from jessica_behaviour_local_spark_building import sqlContext

'''
create data file
'''
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

'''
convert data to npy files
'''

padding_length = {'x_time':5, 'x_location':5, 'y_time':5, 'y_location':5}
vacabulary_size = {'x_time':24, 'x_location':100, 'y_time':24, 'y_location':100}
embedding_dim = {'x_time':20, 'x_location':300, 'y_time':20, 'y_location':300}

cnn_layers = [
['x_time', 'x_location'],
['y_time', 'y_location'],
]

training_data = behaviour_json2npy(
	input_json = 'example.json',
	output_npy_file_name_prefix = 'trip',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

'''
training the similarity model
'''

x_profile_attributes = ['x_time', 'x_location']
y_profile_attributes = ['y_time', 'y_location']

model, x_input_data_format, y_input_data_format = train_behaviour_similary_model(
	training_data,
	x_profile_attributes,
	y_profile_attributes,
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


'''
build the embedding model from the trained similarity model
'''
emb_model = building_embedding_layer_from_pretrained_model(
	model_weight_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	embedding_layer_name = 'x_profile_embedding_model',
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
],
['document_id','x_time', 'x_location']).write.mode('Overwrite').json('example1.json')

test_data = behaviour_json2npy(
	input_json = 'example1.json',
	output_npy_file_name_prefix = 'emb',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

'''
building the input data from the test data npy and input format
'''
x = []
for a in x_input_data_format:
	for b in test_data:
		if a['atrribute_name'] == b['atrribute_name']:
			x.append(numpy.load(b['npy_file']))

'''
embdding
'''
y_vector = emb_model.predict(x)


'''
calcualte similarity
'''
numpy.inner(y_vector[0], y_vector[1])
numpy.inner(y_vector[0], y_vector[4])
numpy.inner(y_vector[0], y_vector[5])
numpy.inner(y_vector[0], y_vector[6])

##################sequence_similarity_example.py##################
