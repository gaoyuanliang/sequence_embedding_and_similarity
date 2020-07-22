######sequence_similarity_multi_batch_training_example.py########
'''
rm -r sequence_embedding_and_similarity
git clone https://github.com/gaoyuanliang/sequence_embedding_and_similarity.git
mv sequence_embedding_and_similarity/* ./
python3 sequence_similarity_example.py
'''
from jessica_koktail_dl import *
from jessica_koktail_spark import * 
from jessica_koktail_local_spark_building import sqlContext

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

x_sequence_attributes = ['x_time', 'x_location']
y_sequence_attributes = ['y_time', 'y_location']


'''
create data file and convert data to npy files
'''

sqlContext.createDataFrame([
('0', ['t1','t2'],['l1','l2'], ['t1','t2'],['l1','l2'], [2.0]),
('1', ['t1','t3'],['l1','l3'], ['t1','t3'],['l1','l3'], [2.0]),
('2', ['t3','t4'],['l3','l4'], ['t3','t4'],['l3','l4'], [2.0]),
],
['document_id','x_time', 'x_location', 'y_time', 'y_location','label']).write.mode('Overwrite').json('training1.json')

sqlContext.createDataFrame([
('3', ['t1','t2'],['l1','l2'], ['t1','t3'],['l1','l3'], [1.0]),
('4', ['t1','t3'],['l1','l3'], ['t3','t4'],['l3','l4'], [1.0]),
('5', ['t4'],['l4'], ['t4'],['l4'], [1.0]),
('6', ['t1','t2'],['l1','l2'], ['t3','t4'],['l3','l4'], [0.0]),
],
['document_id','x_time', 'x_location', 'y_time', 'y_location','label']).write.mode('Overwrite').json('training2.json')

sqlContext.createDataFrame([
('7', ['t4','t5'],['l4','l5'], ['t1','t2'],['l1','l2'], [0.0]),
('8', ['t4','t5'],['l4','l5'], ['t4','t3'],['l4','l3'], [1.0]),
('9', ['t6','t4'],['l6','l4'], ['t6'],['l6'], [1.0]),
('10', ['t1','t2'],['l1','l2'], ['t4','t5'],['l4','l5'], [0.0]),
],
['document_id','x_time', 'x_location', 'y_time', 'y_location','label']).write.mode('Overwrite').json('training3.json')

sqlContext.read.json('training1.json').show()

'''
convert data to npy
'''

training1_data = koktail_json2npy(
	input_json = 'training1.json',
	output_npy_file_name_prefix = 'training1',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

training2_data = koktail_json2npy(
	input_json = 'training2.json',
	output_npy_file_name_prefix = 'training2',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

training3_data = koktail_json2npy(
	input_json = 'training3.json',
	output_npy_file_name_prefix = 'training2',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

'''
train the model
'''
model, x_input_data_format, y_input_data_format = train_koktail_similary_model_from_multi_batchs(
	input_data_batchs = [training1_data, training2_data, training3_data],
	x_koktail_attributes = x_sequence_attributes,
	y_koktail_attributes = y_sequence_attributes,
	cnn_layers = cnn_layers,
	sqlContext = sqlContext,
	epochs = 1000,
	batch_size = 2,
	model_weight_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	dropout_rate = 0.3,
	gpus = None,
	cnn_layer_num = 1,
	verbose = 0)


'''
update the model
'''
model, x_input_data_format, y_input_data_format = train_koktail_similary_model_from_multi_batchs(
	input_data_batchs = [training1_data, training2_data, training3_data],
	x_koktail_attributes = x_sequence_attributes,
	y_koktail_attributes = y_sequence_attributes,
	cnn_layers = cnn_layers,
	sqlContext = sqlContext,
	epochs = 1000,
	batch_size = 2,
	model_weight_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	dropout_rate = 0.3,
	gpus = None,
	cnn_layer_num = 1,
	verbose = 0,
	training_type = 'update',
	x_input_data_format = x_input_data_format,
	y_input_data_format = y_input_data_format)

'''
test the similarity score prediction
'''
test_data = koktail_json2npy(
	input_json = 'training*.json',
	output_npy_file_name_prefix = 'test',
	sqlContext = sqlContext,
	padding_length = padding_length,
	vacabulary_size = vacabulary_size,
	embedding_dim = embedding_dim)

y_similarity = predict_koktail_similary_from_model(
	model_weight_file = 'model_similary.h5py',
	model_structure_json_file = 'model_similary.json',
	test_data = test_data,
	x_input_data_format = x_input_data_format,
	y_input_data_format = y_input_data_format,
	prediction_json = 'test_prediction.json',
	sqlContext = sqlContext)

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
	""").show(100, False)

######sequence_similarity_multi_batch_training_example.py########
