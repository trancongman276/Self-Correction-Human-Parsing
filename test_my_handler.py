import tarfile
tar = tarfile.open("./archive.tar.gz", "r:gz")
tar.extractall()
tar.close()

from test_handler import MyHandler

_services = MyHandler()

def handle(data, context):
    if not _services.initialized:
        _services.initialize(context)

    if data is None:
        return None
    
    data = _services.preprocess(data)
    data = _services.inference(data)
    data = _services.potprocess(data)

    return data
