import json
import os
import sys
from collections import OrderedDict
"""
    Useful commom tools
"""

if sys.version_info[0] == 3:
    PY3 = True
else:
    PY3 = False

if not PY3:

    def json_load_byteified(file_handle):
        return _byteify(json.load(file_handle, object_hook=_byteify),
                        ignore_dicts=True)

    def json_loads_byteified(json_text):
        return _byteify(json.loads(json_text, object_hook=_byteify),
                        ignore_dicts=True)

    def _byteify(data, ignore_dicts=False):
        # if this is a unicode string, return its string representation
        if isinstance(data, unicode):
            return data.encode('utf-8')
        # if this is a list of values, return list of byteified values
        if isinstance(data, list):
            return [_byteify(item, ignore_dicts=True) for item in data]
        # if this is a dictionary, return dictionary of byteified keys and values
        # but only if we haven't already byteified it
        if isinstance(data, dict) and not ignore_dicts:
            return {
                _byteify(key, ignore_dicts=True): _byteify(value,
                                                           ignore_dicts=True)
                for key, value in data.iteritems()
            }
        # if it's anything else, return it in its original form
        return data


def input_json(file_name, ordered=False):
    if os.path.isfile(file_name):
        inputf = open(file_name, mode="r")
        try:
            file_json = "".join(inputf.readlines()).replace("\r\n", "")
            if not PY3:
                file_obj = json_loads_byteified(file_json)
            else:
                if ordered:
                    file_obj = json.loads(file_json,
                                          object_pairs_hook=OrderedDict)
                else:
                    file_obj = json.loads(file_json)
            return file_obj
        finally:
            inputf.close()


def output_json(obj, file_name):
    outputf = open(file_name, 'w')
    try:
        outputf.write(json.dumps(obj))
    finally:
        outputf.close()
