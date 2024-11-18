from typing import Dict, List
import os
import json

class Storage:
    def __init__(self, data_path = "./data/facial_data_db.json") -> None:
        self.db_path = data_path

    def add_data(self, face_data: Dict):
        data = []
        base_path, filename = os.path.split(self.db_path)

        if not os.path.exists(base_path):
            print("Creating DB!!")
            os.makedirs(base_path)
        if os.path.exists(self.db_path):
            data = self.get_all_data()

        try: 
            data.append(face_data)
            self.save_data(data=data)
            print("Data saved to DB!!")
        except Exception as e:
            raise e

    def get_all_data(self) -> List:
        if not os.path.exists(self.db_path):
            raise Exception("Database File not found")
        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)
                for d in data:
                    d["encoding"] = tuple(d["encoding"])
                return data
        except Exception as e:
            raise e
        
    # def delete_data(self, face_id: str) -> bool:
    #     all_data = self.get_all_data()
    #     num_entries = len(all_data)
    #     for idx, face_data in enumerate(all_data):
    #         for key_pair in face_data.items():
    #             if face_id in key_pair:
    #                 all_data.pop(idx)

    #     if num_entries != len(all_data):
    #         self.save_data(data=all_data)
    #         print(f"{num_entries - len(all_data)} faces deleted and update data saved to DB!!")
    #         return True
    #     return False
    
    def save_data(self, data:Dict = None):
        if data is not None:
            with open(self.db_path, "w") as f:
                json.dump(data, f)