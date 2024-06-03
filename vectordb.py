from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

class MilvusDB:
    INDEX_TYPE = 'IVF_FLAT' #'HNSW'
    METRIC_TYPE = 'L2'

    def __init__(self, host='127.0.0.1', port=19530) -> None:
        # 连接数据库
        connections.connect(host=host, port=port)
    
    """创建表（如果存在会先删除），并创建索引"""
    def create_milvus_collection(self, collection_name, dim):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(name='path', dtype=DataType.VARCHAR, description='path to image', max_length=500, is_primary=True, auto_id=False),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='image embedding vectors', dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description='reverse image search')
        self.collection = Collection(name=collection_name, schema=schema)

        # 创建索引，注意使用的类型
        index_params = {
            'metric_type': MilvusDB.METRIC_TYPE,
            'index_type': MilvusDB.INDEX_TYPE,
            'params': {"nlist": 128} # 128簇
        }
        self.collection.create_index(field_name='embedding', index_params=index_params)
        return self.collection
    
    """连接表，返回表内数据条数"""
    def connect_collection(self, collection_name):
        self.collection = Collection(collection_name)
        return self.collection.num_entities
    
    """插入数据（可以批量多条），返回表内数据条数"""
    def insert_data(self, user_data):
        self.collection.insert(data=user_data, partition_name='_default')
        self.collection.flush()
        return self.collection.num_entities

    def search_data(self, key_feature, topk=10) -> list:
        self.collection.load()
        res = self.collection.search(
            data=key_feature,
            limit=topk,                #返回记录数
            anns_field='embedding',    #查询字段
            param={'nprobe': 10, 'metric_type': MilvusDB.METRIC_TYPE}, #nprobe在最近的10个簇中搜索
            output_fields=['path'] #只能是标量字段，向量无法返回
        )
        image_files = []
        for hits in res:
            for hit in hits:
                # print(hit.id)
                # print(hit.distance)
                image_files.append(hit.id)
        return image_files

