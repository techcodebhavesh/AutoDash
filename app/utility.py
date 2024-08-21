import os
import uuid
from werkzeug.utils import secure_filename
import numpy as np
dirname = os.path.dirname(__file__)

class FileUploader:
    def __init__(self, request):
        self.request = request

    def uploadFile(self):       
        if 'code' not in self.request.files:
            return None
        file = self.request.files['code']
        if file.filename == '':
            return None
        if file:
            _url = os.path.splitext(file.filename)
            filename = secure_filename(_url[0]+"_"+str(uuid.uuid4())+_url[1])
            filename=os.path.join(os.environ['NGINX_FOLDER'], filename)
            print(filename)
            file.save(filename)
            return filename

    def deleteFile(self, file_path):
        if os.path.exists(file_path):
            if os.path.basename(os.path.dirname(file_path)) == "res":
                os.remove(file_path)
            return True
        return False


def newestFilePath(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)