import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        names_dir_param = params.get("names_directory", {})
        if isinstance(names_dir_param, dict):
            names_dir = names_dir_param.get("string_value", "/models/labels/names")
        else:
            names_dir = str(names_dir_param)

        self.names_dir = names_dir
        self.labels_cache = {}

        if os.path.isdir(self.names_dir):
            for fname in os.listdir(self.names_dir):
                if not fname.endswith(".txt"):
                    continue
                model_name = fname[:-4]
                path = os.path.join(self.names_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    self.labels_cache[model_name] = [
                        line.strip() for line in f if line.strip()
                    ]

    def execute(self, requests):
        responses = []
        for request in requests:
            model_name_tensor = pb_utils.get_input_tensor_by_name(request, "model_name")
            if model_name_tensor is None:
                out = np.array([], dtype=object)
            else:
                model_name = model_name_tensor.as_numpy().reshape(-1)[0]
                if isinstance(model_name, bytes):
                    model_name = model_name.decode("utf-8")

                labels = self.labels_cache.get(model_name, [])
                out = np.array([l.encode("utf-8") for l in labels], dtype=object)

            output_tensor = pb_utils.Tensor("labels", out)
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses

    def finalize(self):
        pass
