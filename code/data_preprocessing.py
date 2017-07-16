
# coding: utf-8

# # Data preprocessing
# - We use tensorflow framework to do some initial filtering and transform data into a sparse matrix.

# In[1]:

import csv
import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix


# In[2]:

get_ipython().run_cell_magic('time', '', "\nact_at_least = 3\n\n# ratings\nwith open('../data_processed/matrix/mat_ratings.csv') as cfs:\n    reader = csv.reader(cfs)\n    ratings = list(reader)\n    \nt_ratings = tf.constant(ratings)\nt_ratings = tf.string_to_number(ratings)\nt_ratings_mask = tf.ones(t_ratings.shape)\n\n# actresses\nwith open('../data_processed/matrix/mat_actresses_names.csv', encoding='ISO-8859-1') as cfs:\n    reader = csv.reader(cfs)\n    actress_names = list(reader)\n    actress_names = [name[0] for name in actress_names]\nt_actress_names = tf.constant(actress_names)\n\nactress_loader = np.load('../data_processed/matrix/mat_actresses_actin.npz')\nt_actress_actin = tf.SparseTensor(\n    indices=actress_loader['actin_pos'],\n    values=np.full(len(actress_loader['actin_pos']), 1),\n    dense_shape=actress_loader['shape']\n)\nt_actress_actin = tf.to_float(t_actress_actin)\n\n# actors\nwith open('../data_processed/matrix/mat_actors_names.csv', encoding='ISO-8859-1') as cfs:\n    reader = csv.reader(cfs)\n    actor_names = list(reader)\n    actor_names = [name[0] for name in actor_names]\nt_actor_names = tf.constant(actor_names)\n\nactor_loader = np.load('../data_processed/matrix/mat_actors_actin.npz')\nt_actor_actin = tf.SparseTensor(\n    indices=actor_loader['actin_pos'],\n    values=np.full(len(actor_loader['actin_pos']), 1),\n    dense_shape=actor_loader['shape']\n)\nt_actor_actin = tf.to_float(t_actor_actin)\n\nt_actin_all_gender = tf.sparse_concat(0, [t_actor_actin, t_actress_actin])\nt_names_all_gender = tf.concat([t_actor_names, t_actress_names], 0)\n\n# count number of movies acted\nt_count = tf.sparse_tensor_dense_matmul(\n    t_actin_all_gender,\n    t_ratings_mask\n)\n\nt_threshold_mask = tf.greater_equal(\n    t_count,\n    act_at_least\n)\n\nt_filtered_names = tf.boolean_mask(t_names_all_gender, tf.map_fn(lambda _: _[0], t_threshold_mask))\n\nwith tf.Session() as sess:\n    mask_threshold_actors = t_threshold_mask.eval()\n    sparse_actin = t_actin_all_gender.eval()\n    names = t_filtered_names.eval()")


# In[3]:

get_ipython().run_cell_magic('time', '', "\nnp.savez('../data_processed/filtered_matrix/filtered_names', names=names)")


# In[12]:

get_ipython().run_cell_magic('time', '', "\ndata = sparse_actin.values\nrow = sparse_actin.indices[:, 0]\ncol = sparse_actin.indices[:, 1]\nshape = sparse_actin.dense_shape\ndense_matrix = coo_matrix((data, (row, col)), shape=shape, dtype=int).toarray()\nfiltered = dense_matrix[mask_threshold_actors.T[0]]\n\nsparse_filtered = coo_matrix(filtered)\ndata = sparse_filtered.data\nrow = sparse_filtered.row\ncol = sparse_filtered.col\nshape = sparse_filtered.shape\nnp.savez('../data_processed/filtered_matrix/filtered_actin_data_for_clustering', data=data, row=row, col=col, shape=shape)")

