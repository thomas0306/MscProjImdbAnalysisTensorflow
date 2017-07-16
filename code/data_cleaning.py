
# coding: utf-8

# # Data cleaning
# We transform string data into matrices:
# - movie_names matrix
# - movie_rating matrix
# - actor/actress names matrix
# - actor/actress actin boolean matrix

# ## Setup

# ### Import library

# In[ ]:

import numpy as np
import re
import os


# ### Setup constant

# In[ ]:

pattern = re.compile('\s{2,}')
actor_pattern = re.compile('\t{1,}')
year_in_name_pattern = re.compile('^(?:.+\()([1-2][0-9]{3})(?:.*\))(?:\s\{.*\}){0,1}$')
min_vote = 1000


# ## Cleaning

# ### Movie and Ratings

# In[ ]:

get_ipython().run_cell_magic('time', '', '\n# movie name and rating matrix\nmovie_names = np.array([], dtype=str)\nratings = np.array([], dtype=float)\n\nwith open(\'../data_raw/ratings.list\', encoding=\'ISO-8859-1\') as fs:\n    line = fs.readline()\n    count = 0\n            \n    # skip top lines\n    while line != \'MOVIE RATINGS REPORT\\n\':\n        line = fs.readline()\n    \n    for i in range(3):\n        line = fs.readline()\n        \n    while line and line != \'\\n\':\n        line = line.strip()\n        arr = re.split(pattern, line)\n        \n        # consider only movies\' vote > 1000\n        # only take in movies\n        vote_count = int(arr[1])\n        if vote_count > min_vote and arr[3][0] != "\\"" and arr[3][-1] != "\\"":\n            count += 1\n            movie_names = np.append(movie_names, [str(arr[3])])\n            ratings = np.append(ratings, [float(arr[2])])\n        \n        line = fs.readline()\n\nnp.savetxt(\'../data_processed/matrix/mat_movie_names.csv\', movie_names.T, delimiter=",", fmt=\'\\"%s\\"\')\nnp.savetxt(\'../data_processed/matrix/mat_ratings.csv\', ratings.T, delimiter=",", fmt=\'%.1f\')')


# ### Actresses

# In[ ]:

get_ipython().run_cell_magic('time', '', '\nactors = np.array([], dtype=str)\nactin_pos = np.array([], dtype=int).reshape(0,2)\n\nwith open(\'../data_raw/actresses.list\', encoding=\'ISO-8859-1\') as fs:\n    line = fs.readline()\n    count = 0\n    \n    # skip top lines\n    while line != \'THE ACTRESSES LIST\\n\':\n        line = fs.readline()\n    \n    for i in range(5):\n        line = fs.readline()\n    \n    # for each actor\n    while line != \'-----------------------------------------------------------------------------\\n\':\n        arr = re.split(actor_pattern, line)\n        name = arr[0]\n        movie = arr[1].strip()\n        \n        movies = []\n        arr = re.split(pattern, movie)\n        if arr[0][0] != "\\"" and arr[0][-1] != "\\"":\n            movies.append(arr[0])\n        line = fs.readline()\n        # get rest of the movies\n        while line != \'\\n\':\n            line = line.strip()\n            arr = re.split(pattern, line)\n            if arr[0][0] != "\\"" and arr[0][-1] != "\\"":\n                movies.append(arr[0])\n            line = fs.readline()\n        \n        if len(movies) > 0:\n            mask = np.in1d(movie_names, movies)\n            actin = np.where(mask == 1)\n            involved_count = actin[0].shape[0]\n            if involved_count > 0:\n                actin_pos = np.append(actin_pos, np.append(\n                    np.full((1,involved_count), count),\n                    actin,\n                    axis=0).T, axis=0)\n                actors = np.append(actors, [str(name)])\n                count += 1\n        line = fs.readline()\n        \nprint (\'writing names...\')\nprint (\'names length: %d\' % len(actors))\nnp.savetxt(\'../data_processed/matrix/mat_actresses_names.csv\', actors.T, delimiter=",", fmt=\'\\"%s\\"\')\nprint (\'writing actin...\')\nnp.savez(\'../data_processed/matrix/mat_actresses_actin\', actin_pos=actin_pos, shape=(count, movie_names.shape[0]))')


# ### Actors

# In[ ]:

get_ipython().run_cell_magic('time', '', '\nactors = np.array([], dtype=str)\n\nactin_pos = np.array([], dtype=int).reshape(0,2)\n\nwith open(\'../data_raw/actors.list\', encoding=\'ISO-8859-1\') as fs:\n    line = fs.readline()\n    count = 0\n    \n    # skip top lines\n    while line != \'THE ACTORS LIST\\n\':\n        line = fs.readline()\n    \n    for i in range(5):\n        line = fs.readline()\n    \n    # for each actor\n    while line != \'-----------------------------------------------------------------------------\\n\':\n        arr = re.split(actor_pattern, line)\n        name = arr[0]\n        movie = arr[1].strip()\n        \n        movies = []\n        arr = re.split(pattern, movie)\n        if arr[0][0] != "\\"" and arr[0][-1] != "\\"":\n            movies.append(arr[0])\n        line = fs.readline()\n        # get rest of the movies\n        while line != \'\\n\':\n            line = line.strip()\n            arr = re.split(pattern, line)\n            if arr[0][0] != "\\"" and arr[0][-1] != "\\"":\n                movies.append(arr[0])\n            line = fs.readline()\n        \n        if len(movies) > 0:\n            mask = np.in1d(movie_names, movies)\n            actin = np.where(mask == 1)\n            involved_count = actin[0].shape[0]\n            if involved_count > 0:\n                actin_pos = np.append(actin_pos, np.append(\n                    np.full((1,involved_count), count),\n                    actin,\n                    axis=0).T, axis=0)\n                actors = np.append(actors, [str(name)])\n                count += 1\n        line = fs.readline()\n        \nprint (\'writing names...\')\nprint (\'names length: %d\' % len(actors))\nnp.savetxt(\'../data_processed/matrix/mat_actors_names.csv\', actors.T, delimiter=",", fmt=\'\\"%s\\"\')\nprint (\'writing actin...\')\nnp.savez(\'../data_processed/matrix/mat_actors_actin\', actin_pos=actin_pos, shape=(count, movie_names.shape[0]))')

