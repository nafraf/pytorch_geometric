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
        raise NotImplementedError

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
