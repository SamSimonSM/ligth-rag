from pymongo import MongoClient

class MongoRepository:
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection_name]

    def get_document_batch(self, campo_flag: str, batch_size: int, last_id=None):
        filtro = {campo_flag: {"$in": [None, False]}}
        if last_id:
            filtro["_id"] = {"$gt": last_id}
        return list(self.collection.find(filtro).sort("_id", 1).limit(batch_size))

    def get_document_active_batch(self, batch_size: int, last_id=None, campo_flag=None):
        filtros = [{"active": {"$in": [None, True]}}]
        
        if campo_flag:
            filtros.append({campo_flag: {"$in": [None, False]}})

        if last_id:
            filtros.append({"_id": {"$gt": last_id}})

        filtro = {"$and": filtros} if len(filtros) > 1 else filtros[0]

        return list(self.collection.find(filtro).sort("_id", 1).limit(batch_size))
    def mark_as_processed(self, doc_id, campo_flag: str):
        return self.collection.update_one(
            {"_id": doc_id},
            {"$set": {campo_flag: True}}
        )
