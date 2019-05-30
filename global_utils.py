import json


def flush_json_metrics(d: dict, step=None):
    """
    writes json dicts (from a python dict of metrics) to be parsed by FloydHub servers
    :param d: a dict whose keys are metric names, and values are corresponding values
    :param step: global step count, default is None
    :return: None
    """
    for key in d:
        if step is None:
            print(json.dumps({"metric": key, "value": float(d[key])}))
        else:
            print(json.dumps({"metric": key, "value": float(d[key]), "step": step}))
