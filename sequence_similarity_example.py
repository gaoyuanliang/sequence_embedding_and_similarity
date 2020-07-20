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
	verbose = 0)

sqlContext.read.json('prediction1.json').show(100, False)

'''
update the model with more traning data
'''

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
	verbose = 0,
	training_type = 'update',
	x_input_data_format = x_input_data_format,
	y_input_data_format = y_input_data_format)

sqlContext.read.json('prediction2.json').show()
'''

'''
test the similarity score prediction
'''
sqlContext.createDataFrame([
('0', ['t1','t2'],['l1','l2'], ['t1','t3'],['l1','l3']),
('1', ['t1','t2'],['l1','l2'], ['t1'],['l1']),
('2', ['t1','t2'],['l1','l2'], ['t4','t3'],['l4','l3']),
('3', ['t1','t2'],['l1','l2'], ['t1','t2'],['l1','l2']),
('4', ['t1','t2'],['l1','l2'], ['t6','t3'],['l6','l3']),
],
['document_id','x_time', 'x_location', 'y_time', 'y_location']).write.mode('Overwrite').json('test.json')

sqlContext.read.json('test.json').show(100, False)

test_data = behaviour_json2npy(
	input_json = 'test.json',
	output_npy_file_name_prefix = 'test',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

y_similarity = predict_behaviour_similary_from_model(
	model_weight_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	test_data = test_data,
	x_input_data_format = x_input_data_format,
	y_input_data_format = y_input_data_format,
	prediction_json = 'test_prediction.json',
	sqlContext = sqlContext)

sqlContext.read.json('test_prediction.json').show()

'''
sqlContext.read.json('test_prediction.json').registerTempTable('test_prediction')
sqlContext.read.json('training*.json').registerTempTable('training')
sqlContext.sql(u"""
	SELECT training.document_id, 
	training.label, 
	test_prediction.prediction
	FROM training
	LEFT JOIN test_prediction
	ON test_prediction.document_id
	= training.document_id
	""").show()
'''

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
load the embedding model
'''
emb_model = load_model(
	model_structure_json = 'emb_model.json',
	model_weight_file = 'emb_model.h5py')

'''
create the sequences data
'''
sequence_0 = {'x_time':['t1','t2'], 'x_location':['l1','l2']}
sequence_1 = {'x_time':['t1','t2'], 'x_location':['l1','l2']}
sequence_2 = {'x_time':['t1','t3'], 'x_location':['l1','l3']}
sequence_3 = {'x_time':['t3','t4'], 'x_location':['l3','l4']}

'''
embed them
'''
vector_0 = bebaviour_embedding(
	input = sequence_0,
	x_input_data_format = x_input_data_format,
	emb_model = emb_model)

vector_1 = bebaviour_embedding(
	input = sequence_1,
	x_input_data_format = x_input_data_format,
	emb_model = emb_model)

vector_2 = bebaviour_embedding(
	input = sequence_2,
	x_input_data_format = x_input_data_format,
	emb_model = emb_model)

vector_3 = bebaviour_embedding(
	input = sequence_3,
	x_input_data_format = x_input_data_format,
	emb_model = emb_model)

'''
calculate the inner product of the vectors
'''
print('similartity between %s and %s: %f'%(
	str(sequence_0), 
	str(sequence_1),
	np.inner(np.array(vector_0), np.array(vector_1))))

print('similartity between %s and %s: %f'%(
	str(sequence_0), 
	str(sequence_2),
	np.inner(np.array(vector_0), np.array(vector_2))))

print('similartity between %s and %s: %f'%(
	str(sequence_0), 
	str(sequence_3),
	np.inner(np.array(vector_0), np.array(vector_3))))

'''
>>> np.inner(np.array(vector_0), np.array(vector_1))
1.7355304520201214
>>> np.inner(np.array(vector_0), np.array(vector_2))
0.9617815310324274
>>> np.inner(np.array(vector_0), np.array(vector_3))
0.03791223773242554
'''

#################sequence_similarity_example.py##################
