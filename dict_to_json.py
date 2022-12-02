import os
import torch
import json
from json import JSONEncoder
import numpy

rootdir = "test_binaries/"
extensions = (".pth")

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in extensions:
            fn_wihout_ext = os.path.splitext(file)[0]
            filepath = os.path.join(subdir, file)
            print("Processing '" + os.path.join(subdir, file) + "'..")
            od1 = torch.load(filepath)

            with open(os.path.join(subdir, fn_wihout_ext) + ".json", "w") as pf:
                json.dump(od1, pf, cls=NumpyArrayEncoder, indent=4)
