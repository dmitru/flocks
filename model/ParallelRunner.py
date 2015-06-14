__author__ = 'dmitry'

from IPython import parallel

class ParallelRunner:
    def __init__(self, profile='nbserver'):
        clients = parallel.Client(profile=profile)
        clients.block = True
        self.dview = clients[:]
        self.lbview = clients.load_balanced_view()

    def execute(self, statements):
        if isinstance(statements, str):
            statements = [statements]
        for statement in statements:
            self.dview.execute(statement)

    def push(self, env):
        self.dview.push(env)




