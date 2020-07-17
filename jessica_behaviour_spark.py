########jessica_behaviour_spark.py#########
import time
import numpy
import hashlib

from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

hash_function = lambda w, num_word_max: (int(hashlib.md5(w.encode('utf-16')).hexdigest(), 16)%(num_word_max - 1) + 1)
hash_list = lambda input, num_word_max: [hash_function(r,num_word_max) for r in input]

#padding should be at lest 5
def behaviour_json2npy(input_json,
	output_npy_file_name_prefix,
	sqlContext,
	padding_length = {},
	vacabulary_size = {},
	embedding_dim = {}):
	output = []
	s = time.time()
	print('loading data from %s'%(input_json))
	input_df = sqlContext.read.json(input_json)
	for f in input_df.schema.fields:
		print('processing %s'%(f.name))
		try:
			if f.name not in ['document_id', 'label']:
				if isinstance(f.dataType.elementType, StringType):
					if f.name in padding_length:
						def pading(input, max_list_len):
							output = [0]*max_list_len
							len_input = len(input)
							n = numpy.min([len_input, max_list_len])
							output[0:n] = input[0:n]
							return output
						udf_padding = udf(lambda input: pading(input, padding_length[f.name]), ArrayType(IntegerType()))
					udf_hash_list = udf(lambda input: hash_list(input, num_word_max = vacabulary_size[f.name]), ArrayType(IntegerType()))
					data = input_df.select(f.name).withColumn(f.name, udf_hash_list(f.name))
					if f.name in padding_length:
						data = data.withColumn(f.name , udf_padding(f.name))
					data = data.collect()
					data = [r[0] for r in data]
					x = numpy.array(data)
					output_file = '%s_%s.npy'%(output_npy_file_name_prefix, f.name)
					conf = {'atrribute_name':f.name, 'npy_file':output_file, 'vacabulary_size': vacabulary_size[f.name], 'embedding_dim': embedding_dim[f.name]}
					if f.name in padding_length:
						conf['padding_length'] = padding_length[f.name]
					output.append(conf)				
					print('saving data to %s'%(output_file))
					numpy.save(output_file, x)
				if isinstance(f.dataType.elementType, DoubleType) or isinstance(f.dataType.elementType, IntegerType):
					data = input_df.select(f.name)
					if f.name in padding_length:
						def pading(input, max_list_len):
							output = [0.0]*max_list_len
							len_input = len(input)
							n = numpy.min([len_input, max_list_len])
							output[0:n] = input[0:n]
							return output
						udf_padding = udf(lambda input: pading(input, padding_length[f.name]), ArrayType(FloatType()))
						data = data.withColumn(f.name , udf_padding(f.name))
					data = data.collect()
					data = [r[0] for r in data]
					x = numpy.array(data)
					output_file = '%s_%s.npy'%(output_npy_file_name_prefix, f.name)
					conf = {'atrribute_name':f.name, 'npy_file':output_file}
					if f.name in padding_length:
						conf['padding_length'] = padding_length[f.name]
					else:
						conf['vector_length'] = x.shape[-1]
					output.append(conf)				
					print('saving data to %s'%(output_file))
					numpy.save(output_file, x)
		except:
			pass
		try:
			if f.name == 'document_id':
				data = input_df.select(f.name).collect()
				x = numpy.array([r[0] for r in data])
				output_file = '%s_%s.npy'%(output_npy_file_name_prefix, f.name)
				conf = {'atrribute_name':f.name, 'npy_file':output_file}
				output.append(conf)				
				print('saving data to %s'%(output_file))
				numpy.save(output_file, x)
		except:
			pass
		try:
			if f.name == 'label':
				#udf_label = udf(lambda input: label_to_vector(input, max_class_num), ArrayType(IntegerType()))
				#data = input_df.select(f.name).withColumn(f.name, udf_label(f.name)).collect()
				data = input_df.select(f.name).collect()
				x = numpy.array([r[0] for r in data])
				output_file = '%s_%s.npy'%(output_npy_file_name_prefix, f.name)
				conf = {'atrribute_name':f.name, 'npy_file':output_file}
				output.append(conf)				
				print('saving data to %s'%(output_file))
				numpy.save(output_file, x)
		except:
			pass
	print('running time:\t%f'%(time.time()-s))
	return output

########jessica_behaviour_spark.py#########
