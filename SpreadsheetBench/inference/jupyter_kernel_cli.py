import json
import requests

class ClientJupyterKernel:
    def __init__(self, url, conv_id):
        self.url = url
        self.conv_id = conv_id
        print(f"ClientJupyterKernel initialized with url={url} and conv_id={conv_id}")

    def execute(self, code):
        payload = {"convid": self.conv_id, "code": code}
        response = requests.post(self.url, data=json.dumps(payload))
        response_data = response.json()
        if response_data["new_kernel_created"]:
            print(f"New kernel created for conversation {self.conv_id}")
        return response_data["result"]
