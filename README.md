# Deep embedding of sequence and similarity comparision

Behaviour embedding, categorization, regression, and similarity comparision

## Training the similarity deep learning

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
