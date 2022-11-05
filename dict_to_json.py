import torch
import json
from json import JSONEncoder
import numpy

file_path = "test_binaries/quant_io"

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

od1 = torch.load(file_path + ".pth")

with open(file_path + ".json", "w") as pf:
    json.dump(od1, pf, cls=NumpyArrayEncoder, indent=4)