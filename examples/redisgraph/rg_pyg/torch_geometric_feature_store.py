from typing import List, Optional, Tuple
from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.typing import FeatureTensorType
from .connection import Connection

from .torch_geometric_tensorattr import RedisGraphTensorAttr

class RedisGraphFeatureStore(FeatureStore):
    def __init__(self, server, port):
        super().__init__(RedisGraphTensorAttr)
        self.server = server
        self.port = port
        self.db = 0
        self.connection = None

    def __get_connection(self):
        if not self.connection:
            self.connection = Connection(self.server, self.port, self.db)
        return self.connection

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        raise NotImplementedError

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        table_name = attr.group_name
        attr_name = attr.attr_name
        return self.__get_tensor_by_query(attr)


    def __get_tensor_by_query(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        table_name = attr.group_name
        attr_name = attr.attr_name
        indices = attr.index

        self.__get_connection()

        match_clause = "MATCH (item:%s)" % table_name
        return_clause = "RETURN item.%s" % attr_name

        if indices is None:
            where_clause = ""
        elif isinstance(indices, int):
            where_clause = "WHERE id(item) = %d" % indices
        elif isinstance(indices, slice):
            if indices.step is None or indices.step == 1:
                where_clause = "WHERE id(item) >= %d AND id(item) < %d" % (
                    indices.start, indices.stop)
            else:
                where_clause = "WHERE id(item) >= %d AND id(item) < %d AND id(item) - %d) %% %d = 0" % (
                    indices.start, indices.stop, indices.start, indices.step)
        elif isinstance(indices, Tensor) or isinstance(indices, list) or isinstance(indices, np.ndarray) or isinstance(indices, tuple):
            where_clause = "WHERE"
            for i in indices:
                where_clause += " id(item) = %d OR" % int(i)
            where_clause = where_clause[:-3]
        else:
            raise ValueError("Invalid attr.index type: %s" % type(indices))

        query = "%s %s %s" % (match_clause, where_clause, return_clause)
        result_set = self.connection.execute(query).get_as_result_set()

        result_list = []
        for row in result_set:
            for val in row:
                if(type(val) is list):
                    if(len(val) == 1):
                        result_list.append(val[0])
                    else:
                        result_list.append(val)
                else:
                    result_list.append(val)
        try:
            tensor_result = torch.tensor(result_list)
            return tensor_result
        except:
            return result_list

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        raise NotImplementedError

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        raise NotImplementedError

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        raise NotImplementedError


    def __get_node_property(self, graph_name: str, attr_name: str) -> str:
        if table_name in self.node_properties_cache and attr_name in self.node_properties_cache[table_name]:
            return self.node_properties_cache[table_name][attr_name]
        self.__get_connection()
        if table_name not in self.node_properties_cache:
            self.node_properties_cache[table_name] = self.connection._get_node_property_names(
                table_name)
        if attr_name not in self.node_properties_cache[table_name]:
            raise ValueError("Attribute %s not found in graph %s" %
                             (attr_name, table_name))
        attr_info = self.node_properties_cache[table_name][attr_name]
        return attr_info