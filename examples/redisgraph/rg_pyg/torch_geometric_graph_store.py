from typing import Any, List, Optional, Tuple

from torch_geometric.data.graph_store import (
        EdgeAttr, GraphStore, ConversionOutputType
)
from torch_geometric.typing import EdgeTensorType


class RedisGraphGraphStore(GraphStore):
    def __init__(self):
        super().__init__()

    @staticmethod
    def key(attr: EdgeAttr) -> Tuple:
        return (attr.edge_type, attr.layout.value, attr.is_sorted)

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        raise NotImplementedError

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        print('_get_edge_index')
        # TODO: This is a copy from kuzu backend code
        if edge_attr.layout.value == EdgeLayout.COO.value:
            # We always return a sorted COO edge index, if the request is
            # for an unsorted COO edge index, we change the is_sorted flag
            # to True and return the sorted COO edge index.
            if edge_attr.is_sorted == False:
                edge_attr.is_sorted = True
        key = self.key(edge_attr)
        if key in self.store:
            rel = self.store[self.key(edge_attr)]
            if not rel.materialized:
                if rel.layout != EdgeLayout.COO.value:
                    raise ValueError("Only COO layout is supported")
            if rel.layout == EdgeLayout.COO.value:
                self.__get_edge_coo_from_database(self.key(edge_attr))
            return rel.edge_index
        else:
            return None

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        raise NotImplementedError

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        raise NotImplementedError

    # Layout Conversion #######################################################
    def coo(
        self,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:
        raise NotImplementedError

    def csr(
        self,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:
        raise NotImplementedError

    def csc(
        self,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:
        raise NotImplementedError
