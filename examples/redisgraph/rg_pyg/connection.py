import redis
from redis.commands.graph import Graph
from redis.commands.graph.node import Node
from redis.commands.graph.edge import Edge

from .query_result import QueryResult

graph = None
GRAPH_ID = "pyg"

class Connection:
    def __init__(self, host, port, db):
        self.host = host
        self.port = port
        self.db = db
        self._connection = None
        self.init_connection()

    def init_connection(self):
        if self._connection is None:
            self._connection = redis.Redis(self.host, self.port, self.db)
            global graph
            graph = Graph(self._connection, GRAPH_ID)

    def execute(self, query, parameters=[]):
        """
        Execute a query.

        Parameters
        ----------
        query : str
            A query string.
        parameters : list
            Parameters for the query.

        Returns
        -------
        QueryResult
            Query result.
        """
        self.init_connection()
        redisgraph_result = graph.query(query)
        return QueryResult(self, redisgraph_result)
