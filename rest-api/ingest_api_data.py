import requests
import json
import os


class IngestAPI:
    def __init__(self):
        self.__page_index = 0
        self.__url = "http://{host}:{port}/api/v1/data".format(
            host=os.getenv("API_HOST", "0.0.0.0"), port=os.getenv("API_PORT", "5000")
        )

    def get_data(self):
        resp = requests.get(self.__url, params={"page": self.__page_index})
        data = json.loads(resp.content)
        self.__page_index += 1
        return data

    def get_page_index(self):
        return self.__page_index
