

class Database:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def get_torch_geometric_remote_backend(self):
        from torch_geometric_feature_store import RedisGraphFeatureStore
        from torch_geometric_graph_store import RedisGraphGraphStore
        return RedisGraphFeatureStore(self.host, self.port), RedisGraphGraphStore()


