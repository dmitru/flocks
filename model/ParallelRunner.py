__author__ = 'dmitry'

from IPython import parallel
clients = parallel.Client(profile='nbserver')
clients.block = True  # use synchronous computations
print clients.ids

dview = clients[:]
lbview = clients.load_balanced_view()

dview.execute('import numpy as np')
dview.execute('import math')

dview.push(dict(

))


