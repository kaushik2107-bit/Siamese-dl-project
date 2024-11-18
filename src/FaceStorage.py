from .Storage import Storage

class FaceStorage:
    def __init__(self, data_path = "data/facial_data.json") -> None:
        self.db = Storage(data_path)
        saved_data = []
        try:
            saved_data = self.db.get_all_data()
            print(f"Data loaded from DB with {len(saved_data)} entries!!")
        except Exception as e:
            print("DB not found")

    def add_facial_data(self, facial_data):
        self.db.add_data(facial_data)

    # def remove_facial_data(self, face_id: str = None):
    #     self.db.delete_data(face_id)

    def get_all_facial_data(self):
        return self.db.get_all_data()
    
