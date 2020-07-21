############jessica_koktail_dl.py##########
import h5py
import time
import numpy as np
from keras.losses import *
from keras.metrics import *
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.utils.training_utils import *
from keras.preprocessing.text import *
from keras.preprocessing.sequence import *

from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

from jessica_koktail_spark import *

####################

def build_koktail_embedding_model(
	data_attributes,
	cnn_layers,
	dropout_rate = 0.3,
	embedding_layer_name = 'emb',
	cnn_layer_num = 2):
	layer_input = []
	layer_merge = []
	input_data_inf = []
	##################no cnn layers
	for f in data_attributes:
		i, o, x = None, None, None
		'''
		for the attributs of numerical vectors, 
		'''
		if 'vector_length' in f and f['atrribute_name'] not in ['document_id', 'label']:
			i = Input(shape=(f['vector_length'], ))
			o = i
		'''
		for the attributes of strings but not squence for cnn, do embedding
		'''
		if 'vacabulary_size' in f and 'padding_length' not in f:
			i = Input(shape=(1, ))
			o = Dropout(dropout_rate)(Embedding(
				input_dim = f['vacabulary_size'],
				output_dim = f['embedding_dim'],
				input_length = 1)(i))
			o = Reshape(target_shape=(f['embedding_dim'],))(o)
		if i is not None and o is not None:
			layer_input.append(i)
			layer_merge.append(o)
			input_data_inf.append(f)
	##################cnn layers
	for m in cnn_layers:
		#concatenation then cnn
		'''
		for the sequence of multiple attributes, embedding and then 
		concatnate, and then cnn
		'''
		if len(m) >= 2:
			o_merge = []
			for f in m:
				'''
				for each cnn attributes, go through the attribute list
				'''
				for f1 in data_attributes:
					i, o, x = None, None, None
					if f == f1['atrribute_name']:
						i = Input(shape=(f1['padding_length'], ))
						'''
						embedding of sequence of strings
						'''
						if 'vacabulary_size' in f1:
							o = Dropout(dropout_rate)(\
								Embedding(
								input_dim = f1['vacabulary_size'],
								output_dim = f1['embedding_dim'],
								input_length = f1['padding_length'])\
								(i))
						else:
							o = Reshape(target_shape=(f1['padding_length'],1,))(i)
					if i is not None and o is not None:
						#print('adding %s'%(f1))
						o_merge.append(o)
						layer_input.append(i)
						input_data_inf.append(f1)
			o_merge = concatenate(o_merge)
			o_merge = Conv1D(filters = 1000,
				kernel_size = 1, 
				padding='valid',
				activation='relu', 
				strides=1)(o_merge)
			if cnn_layer_num >=2:
				for i_layer in range(1,cnn_layer_num):
					o_merge = Dropout(dropout_rate)(MaxPooling1D()(o_merge))
					o_merge = Conv1D(filters = 1000,
						kernel_size = 2, padding='valid',
						activation='relu', strides=1)(o_merge)
			o_merge = Dropout(dropout_rate)(GlobalMaxPooling1D()(o_merge))
			layer_merge.append(o_merge)
		#if only one attributes, no concatenation
		else:
			for f1 in data_attributes:
				i, o, x = None, None, None
				if m[0] == f1['atrribute_name']:
					i = Input(shape=(f1['padding_length'], ))
					if 'vacabulary_size' in f1:
						o = Dropout(dropout_rate)(\
							Embedding(
							input_dim = f1['vacabulary_size'],
							output_dim = f1['embedding_dim'],
							input_length = f1['padding_length'])\
							(i))
					else:
						o = Reshape(target_shape=(f1['padding_length'],1,))(i)
					o = Conv1D(filters = 1000,
						kernel_size = 1, padding='valid',
						activation='relu', strides=1)(o)
					if cnn_layer_num >=2:
						for i_layer in range(1,cnn_layer_num):
							o = Dropout(dropout_rate)(MaxPooling1D()(o))
							o = Conv1D(filters = 1000,
								kernel_size = 2, padding='valid',
								activation='relu', strides=1)(o)
					o = Dropout(dropout_rate)(GlobalMaxPooling1D()(o))
				if i is not None and o is not None:
					layer_input.append(i)
					layer_merge.append(o)
					input_data_inf.append(f1)
	##################
	if len(layer_merge) >= 2:
		layer_merged = concatenate(layer_merge)
	else:
		layer_merged = layer_merge[0]
	##################
	o = Dropout(dropout_rate)(Dense(units = 1000, activation='relu')(layer_merged))
	o = Dropout(dropout_rate)(Dense(units = 512, activation='relu', name = embedding_layer_name)(o))
	##################
	model = Model(inputs=layer_input, outputs=o)
	return model, input_data_inf


def building_x_from_input_dataformat_and_npy(
	input_format,
	input_data_attributes):
	x = []
	for a in input_format:
		for b in input_data_attributes:
			if a['atrribute_name'] == b['atrribute_name']:
				x.append(np.load(b['npy_file']))
	return x

#########regression#########

def build_koktail_regression_model(
	data_attributes,
	cnn_layers,
	response_dim = 1,
	gpus = None,
	dropout_rate = 0.3):
	koktail_embedding_model, input_data_inf = build_koktail_embedding_model(
		data_attributes,
		cnn_layers,
		dropout_rate)
	o = Dense(units = response_dim)(koktail_embedding_model.output)
	model_regression = Model(inputs=koktail_embedding_model.input, outputs=o)
	if gpus is not None:
		model_regression = multi_gpu_model(model_regression, gpus = gpus)
	return model_regression, input_data_inf

def train_koktail_regression_model(data_attributes,
	cnn_layers,
	sqlContext,
	response_dim = 1,
	epochs = 4,
	gpus = None,
	batch_size = 500,
	model_file = None,
	prediction_json = None,
	dropout_rate = 0.3):
	start_time = time.time()
	#build the model
	print('building the model')
	model, input_data_inf = build_koktail_regression_model(
		data_attributes,
		cnn_layers,
		response_dim = response_dim, 
		gpus = gpus,
		dropout_rate = dropout_rate)
	model.compile(loss='root_mean_squared_error',
		optimizer='adam',
		metrics=['root_mean_squared_error', 'mean_squared_error'])
	#train the model
	######load the data
	print('laoding the training  data from the configuration')
	x = []
	for f in input_data_inf:
		print('loading x data from %s'%(f['npy_file']))
		x.append(np.load(f['npy_file']))
	for f in data_attributes:
		if f['atrribute_name'] == 'label':
			print('loading label data from %s'%(f['npy_file']))
			y = np.load(f['npy_file'])
	######
	print('training model')
	model.fit(x, y,
		batch_size = batch_size, \
		epochs = epochs)
	#####
	if model_file is not None:
		print('saving model to  %s'%(model_file))
		model.save_weights(model_file)
	#####
	if prediction_json is not None:
		print('predicting the output of the training samples')
		y_score = model.predict(x)
		print('constructing the predictiong data frame')
		for f in data_attributes:
			if f['atrribute_name'] == 'document_id':
				document_id = np.load(f['npy_file'])
		####
		df_list = [[list(label1), list(preidction), str(file_name)] 
			for label1, preidction, file_name
			in zip(y.tolist(),
				y_score.tolist(),
				document_id.tolist())]
		###
		df_schema = StructType([\
			StructField("label", ArrayType(DoubleType())),\
			StructField('prediction', ArrayType(DoubleType())),\
			StructField('document_id', StringType())])
		###
		df = sqlContext.createDataFrame(df_list, schema = df_schema)
		print('saving prediction results to %s'%(prediction_json))
		df.write.mode('Overwrite').json(prediction_json)
	####
	print('training time:\t%f scondes'%(time.time()-start_time))
	return model, input_data_inf

def koktail_regression_from_model(
	test_data_attributes,
	model_input_data_format,
	cnn_layers,
	model_file,
	sqlContext,
	prediction_json,
	response_dim = 1,
	gpus = None,
	dropout_rate = 0.3):
	print('building the model')
	model, input_data_inf = build_koktail_regression_model(
		model_input_data_format,
		cnn_layers,
		response_dim = response_dim, 
		gpus = gpus)
	print('loading model parameters')
	model.load_weights(model_file)
	model.compile(loss='mean_squared_error',
		optimizer='adam',
		metrics=['mean_squared_error'])
	#model._make_predict_function()
	###
	print('laoding the test data from npy')
	x = []
	for f in model_input_data_format:
		for f1 in test_data_attributes:
			if f1['atrribute_name'] == f['atrribute_name']:
				x.append(np.load(f1['npy_file']))
	print('predciting by model')
	y_score = model.predict(x)
	print('constructing the predictiong data frame')
	for f in test_data_attributes:
		if f['atrribute_name'] == 'document_id':
			document_id = np.load(f['npy_file'])
	####
	df_list = [[list(preidction), str(file_name)] 
		for preidction, file_name
		in zip(y_score.tolist(),
			document_id.tolist())]
	###
	df_schema = StructType([\
		StructField('prediction', ArrayType(DoubleType())),\
		StructField('document_id', StringType())])
	###
	df = sqlContext.createDataFrame(df_list, schema = df_schema)
	print('saving prediction results to %s'%(prediction_json))
	df.write.mode('Overwrite').json(prediction_json)

#########classification#########

def build_koktail_categorization_model(
	data_attributes,
	cnn_layers,
	max_class_num = 2,
	cnn_layer_num = 1,
	gpus = None):
	koktail_embedding_model, input_data_inf = build_koktail_embedding_model(
		data_attributes,
		cnn_layers,
		cnn_layer_num = cnn_layer_num)
	o = Dense(units = max_class_num, activation='softmax')(koktail_embedding_model.output)
	model_classification = Model(inputs=koktail_embedding_model.input, outputs=o)
	if gpus is not None:
		model_classification = multi_gpu_model(model_classification, gpus = gpus)
	return model_classification, input_data_inf

def train_koktail_categorization_model(
	data_attributes,
	cnn_layers,
	sqlContext,
	max_class_num = 2,
	positive_weight = 1,
	epochs = 4,
	gpus = None,
	batch_size = 500,
	model_weight_file = None,
	model_structure_file = None,
	cnn_layer_num = 1,
	prediction_json = None):
	start_time = time.time()
	class_weight = {}
	for class_indx in range(max_class_num):
		if class_indx == 0:
			class_weight[class_indx] = 1
		else:
			class_weight[class_indx] = positive_weight
	#build the model
	print('building the model')
	model, input_data_inf = build_koktail_categorization_model(
		data_attributes,
		cnn_layers,
		max_class_num = max_class_num, 
		cnn_layer_num = cnn_layer_num,
		gpus = gpus)
	model.compile(loss='categorical_crossentropy',
		optimizer='adam', metrics=['accuracy'])
	#train the model
	######load the data
	print('laoding the training  data from the configuration')
	x = []
	for f in input_data_inf:
		print('loading x data from %s'%(f['npy_file']))
		x.append(np.load(f['npy_file']))
	for f in data_attributes:
		if f['atrribute_name'] == 'label':
			print('loading label data from %s'%(f['npy_file']))
			y = np.load(f['npy_file'])
	######
	print('training model')
	model.fit(x, y,
		class_weight = class_weight,\
		batch_size = batch_size, \
		epochs = epochs)
	#####
	if model_weight_file is not None:
		print('saving model to  %s'%(model_weight_file))
		model.save_weights(model_weight_file)
	if model_structure_file is not None:
		print('saving model structure to %s'%(model_structure_file))
		model_json = model.to_json()
		with open(model_structure_file, "w") as json_file:
			json_file.write(model_json)
	#####
	if prediction_json is not None:
		print('predicting the output of the training samples')
		y_score = model.predict(x)
		print('constructing the predictiong data frame')
		for f in data_attributes:
			if f['atrribute_name'] == 'document_id':
				document_id = np.load(f['npy_file'])
		####
		df_list = [[int(label1), int(preidction),score, str(file_name)] 
			for label1, preidction, score, file_name
			in zip(np.argmax(y, axis=1).tolist(),
				np.argmax(y_score, axis=1).tolist(),
				np.max(y_score, axis=1).tolist(),
				document_id.tolist())]
		###
		df_schema = StructType([StructField("label", LongType()),\
			StructField('prediction', LongType()),\
			StructField('score', DoubleType()),
			StructField('document_id', StringType())])
		###
		df = sqlContext.createDataFrame(df_list, schema = df_schema)
		print('saving prediction results to %s'%(prediction_json))
		df.write.mode('Overwrite').json(prediction_json)
		print('showing the confusion matrix')
		sqlContext.read.json(prediction_json).registerTempTable('prediction_df')
		#####
		sqlContext.sql(u"""
			SELECT label, prediction, COUNT(*)
			FROM prediction_df
			GROUP BY label, prediction
			""").show()
	####
	print('training time:\t%f scondes'%(time.time()-start_time))
	return model, input_data_inf

def koktail_categorization_from_model(
	test_data_attributes,
	model_input_data_format,
	model_weight_file,
	model_structure_file,
	sqlContext,
	prediction_json,
	max_class_num = 2,
	gpus = None):
	print('load the model structure')
	json_file = open(model_structure_file, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	similarity_model = model_from_json(loaded_model_json)
	print('loading model parameters')
	model.load_weights(model_weight_file)
	model.compile(loss='categorical_crossentropy',
		optimizer='adam', metrics=['accuracy'])
	model._make_predict_function()
	###
	print('laoding the test data from npy')
	x = []
	for f in model_input_data_format:
		for f1 in test_data_attributes:
			if f1['atrribute_name'] == f['atrribute_name']:
				x.append(np.load(f1['npy_file']))
	print('predciting by model')
	y_score = model.predict(x)
	print('loading the document id data')
	for f in test_data_attributes:
		if f['atrribute_name'] == 'document_id':
			document_id = np.load(f['npy_file'])
	####
	df_list = [[int(preidction),score, str(file_name)] 
		for preidction, score, file_name
		in zip(np.argmax(y_score, axis=1).tolist(),
			np.max(y_score, axis=1).tolist(),
			document_id.tolist())]
	###
	df_schema = StructType([
		StructField('prediction', LongType()),\
		StructField('score', DoubleType()),
		StructField('document_id', StringType())])
	###
	df = sqlContext.createDataFrame(df_list, schema = df_schema)
	print('saving prediction results to %s'%(prediction_json))
	df.write.mode('Overwrite').json(prediction_json)
	print('showing the prediction matrix')
	sqlContext.read.json(prediction_json).registerTempTable('prediction_df')
	#####
	sqlContext.sql(u"""
		SELECT prediction, COUNT(*)
		FROM prediction_df
		GROUP BY prediction
		""").show()

########similarity############

def build_koktail_similarity_model(
	data_attributes,
	cnn_layers,
	x_koktail_attributes,
	y_koktail_attributes,
	gpus = None,
	dropout_rate = 0.3,
	shared_embedding_layer = True,
	cnn_layer_num = 2):
	x_atttibute = []
	y_atttibute = []
	x_cnn_layers = []
	y_cnn_layers = []
	'''
	construct the attribute list of x and y
	'''
	for a1 in data_attributes:
		for a in x_koktail_attributes:
			if a1['atrribute_name'] == a:
				x_atttibute.append(a1)
		for a in y_koktail_attributes:
			if a1['atrribute_name'] == a:
				y_atttibute.append(a1)
	'''
	construct the cnn layers of x and y
	'''
	for l in cnn_layers:
		if set(l) & set(x_koktail_attributes):
			x_cnn_layers.append(l)
		if set(l) & set(y_koktail_attributes):
			y_cnn_layers.append(l)
	'''
	building the embedding layers
	'''
	x_koktail_embedding_model, x_input_data_inf = build_koktail_embedding_model(
		x_atttibute,
		cnn_layers = x_cnn_layers,
		dropout_rate = dropout_rate,
		cnn_layer_num = cnn_layer_num)
	x_koktail_embedding_model.name = 'x_koktail_embedding_model'
	y_koktail_embedding_model, y_input_data_inf = build_koktail_embedding_model(
		y_atttibute,
		cnn_layers = y_cnn_layers,
		dropout_rate = dropout_rate,
		cnn_layer_num = cnn_layer_num)
	y_koktail_embedding_model.name = 'y_koktail_embedding_model'
	'''
	building the similarity layers
	'''
	x_input = x_koktail_embedding_model.input
	y_input = y_koktail_embedding_model.input
	x_mebedding = x_koktail_embedding_model(x_input)
	if shared_embedding_layer is True:
		y_mebedding = x_koktail_embedding_model(y_input)
	else:
		y_mebedding = y_koktail_embedding_model(y_input)
	'''
	similarty score is the inner product of the two embedding vectors
	'''
	inner_product_layer = Dot(axes=1, name = 'inner_product')([x_mebedding, y_mebedding])
	#inner_product_layer = Reshape((1, 1))(inner_product_layer)
	model_similary = Model(
		inputs = x_input+y_input, 
		outputs = inner_product_layer)
	return model_similary, x_input_data_inf, y_input_data_inf

def train_koktail_similary_model(
	data_attributes,
	x_koktail_attributes,
	y_koktail_attributes,
	cnn_layers,
	sqlContext,
	epochs = 4,
	gpus = None,
	batch_size = 500,
	model_weight_file = None,
	model_structure_json_file = None, 
	prediction_json = None,
	dropout_rate = 0.3,
	cnn_layer_num = 2,
	training_type = 'initial',
	x_input_data_format = None,
	y_input_data_format = None,
	verbose = 1):
	start_time = time.time()
	'''
	if training_type == 'initial':
		#build the model
	'''
	print('building the model')
	model, x_input_data_format, y_input_data_format = build_koktail_similarity_model(
		data_attributes,
		cnn_layers,
		x_koktail_attributes,
		y_koktail_attributes,
		gpus = gpus,
		dropout_rate = dropout_rate,
		cnn_layer_num = cnn_layer_num)
	model.compile(loss='mean_squared_error',
		optimizer='adam',
		metrics=['mean_squared_error'])
	if training_type == 'update':
		print('loading initial model')
		'''
		json_file = open(model_structure_json_file, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.compile(loss='mean_squared_error',
			optimizer='adam',
			metrics=['mean_squared_error'])
		'''
		model.load_weights(model_weight_file)
	#train the model
	######load the data
	print('loading the training data')
	x = building_x_from_input_dataformat_and_npy(
		input_format = x_input_data_format+y_input_data_format,
		input_data_attributes = data_attributes)
	for f in data_attributes:
		if f['atrribute_name'] == 'label':
			print('loading label data from %s'%(f['npy_file']))
			y = np.load(f['npy_file'])
	'''
	for a in x:
		print(a.shape)
	print(x)
	'''
	######
	print('training model')
	model.fit(x, y,
		batch_size = batch_size,
		epochs = epochs,
		verbose = verbose)
	#####
	if model_structure_json_file is not None:
		print('saving model structure to %s'%(model_structure_json_file))
		model_json = model.to_json()
		with open(model_structure_json_file, "w") as json_file:
			json_file.write(model_json)
	#####
	if model_weight_file is not None:
		print('saving model weight to  %s'%(model_weight_file))
		model.save_weights(model_weight_file)
	#####
	if prediction_json is not None:
		print('predicting the output of the training samples')
		y_score = model.predict(x)
		print('constructing the predictiong data frame')
		for f in data_attributes:
			if f['atrribute_name'] == 'document_id':
				document_id = np.load(f['npy_file'])
		####
		df_list = [[list(label1), list(preidction), str(file_name)] 
			for label1, preidction, file_name
			in zip(y.tolist(),
				y_score.tolist(),
				document_id.tolist())]
		###
		df_schema = StructType([\
			StructField("label", ArrayType(DoubleType())),\
			StructField('prediction', ArrayType(DoubleType())),\
			StructField('document_id', StringType())])
		###
		df = sqlContext.createDataFrame(df_list, schema = df_schema)
		print('saving prediction results to %s'%(prediction_json))
		df.write.mode('Overwrite').json(prediction_json)
	####
	print('training time:\t%f scondes'%(time.time()-start_time))
	return model, x_input_data_format, y_input_data_format

'''
predict the similarity from the test data
'''
def predict_koktail_similary_from_model(
	model_weight_file,
	model_structure_json_file,
	test_data,
	x_input_data_format,
	y_input_data_format,
	sqlContext,
	prediction_json = None):
	print('loading pretrained model')
	json_file = open(model_structure_json_file, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	similarity_model = model_from_json(loaded_model_json)
	similarity_model.load_weights(model_weight_file)
	print('loading data')
	x = []
	for a in x_input_data_format+y_input_data_format:
		for b in test_data:
			if a['atrribute_name'] == b['atrribute_name']:
				x.append(np.load(b['npy_file']))
	y_score = similarity_model.predict(x)
	print('building the prediction dataframe')
	if prediction_json is not None:
		for b in test_data:
			if b['atrribute_name'] == 'document_id':
				x_document_id = np.load(b['npy_file'])
		df_list = [[list(preidction), str(file_name)] 
			for preidction, file_name
			in zip(y_score.tolist(),
				x_document_id.tolist())]
		###
		df_schema = StructType([\
			StructField('prediction', ArrayType(DoubleType())),\
			StructField('document_id', StringType())])
		###
		df = sqlContext.createDataFrame(df_list, schema = df_schema)
		print('saving prediction results to %s'%(prediction_json))
		df.write.mode('Overwrite').json(prediction_json)
	return y_score

'''
building the embedding model from pre-trained similarity model
'''
def building_embedding_layer_from_pretrained_model(
	model_weight_file,
	model_structure_json_file,
	embedding_layer_name,
	emb_model_structure_json,
	emb_model_weight_file):
	print('loading pretrained model')
	json_file = open(model_structure_json_file, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	similarity_model = model_from_json(loaded_model_json)
	similarity_model.load_weights(model_weight_file)
	'''
	find the embedding layer
	'''
	for l in similarity_model.layers:
		if l.name == 'x_koktail_embedding_model':
			output_layer = l
	'''
	build the model
	'''
	emb_model = Model(
		inputs = output_layer.inputs, 
		outputs = output_layer.outputs)
	emb_model.compile(loss='mean_squared_error',
			optimizer='adam',
			metrics=['mean_squared_error'])
	emb_model.set_weights(similarity_model.get_weights())
	######
	print('saving model strcutrue to %s'%emb_model_structure_json)
	model_json = emb_model.to_json()
	with open(emb_model_structure_json, "w") as json_file:
		json_file.write(model_json)
	print('saving model weights to %s'%(emb_model_weight_file))
	emb_model.save_weights(emb_model_weight_file)
	return emb_model


'''
load embedding model
'''
def load_model(model_structure_json,
	model_weight_file):
	json_file = open(model_structure_json, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	emb_model = model_from_json(loaded_model_json)
	emb_model.load_weights(model_weight_file)
	return emb_model


'''
convert a sequence to a list of npy
'''
def sequence2x(sequence_x,
	x_input_data_format):
	#######
	def pading(input, max_list_len, filled_element = 0):
		output = [filled_element]*max_list_len
		len_input = len(input)
		n = numpy.min([len_input, max_list_len])
		output[0:n] = input[0:n]
		return output
	#######
	x = []
	for a in x_input_data_format:
		for k in sequence_x:
			if a['atrribute_name'] == k:
				if 'vacabulary_size' in a:
					x1 = hash_list(sequence_x[k], num_word_max = a['vacabulary_size'])
				if 'padding_length' in a:
					if 'vacabulary_size' in a:
						x1 = pading(x1, max_list_len = a['padding_length'], filled_element = 0)
					else:
						x1 = pading(x1, max_list_len = a['padding_length'], filled_element = 0.0)
				x.append(np.array([x1]))
	return x

'''
embedding the sequence
'''
def bebaviour_embedding(input,
	x_input_data_format,
	emb_model):
	x = sequence2x(input,
		x_input_data_format)
	y_vector = emb_model.predict(x)
	return y_vector[0].tolist()

############jessica_koktail_dl.py##########
