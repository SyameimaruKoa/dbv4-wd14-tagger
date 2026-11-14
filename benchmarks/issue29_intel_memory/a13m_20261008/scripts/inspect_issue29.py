import json, platform, sys
import importlib.metadata
import onnxruntime as ort
import openvino as ov
core = ov.Core()
result = {'python': sys.version, 'os': platform.platform(), 'providers': ort.get_available_providers(), 'openvino_devices': core.available_devices, 'versions': {p: importlib.metadata.version(p) for p in ['numpy','openvino','onnxruntime-openvino']}}
try:
    result['gpu_name'] = core.get_property('GPU', 'FULL_DEVICE_NAME')
    result['gpu_supported_properties'] = str(core.get_property('GPU', 'SUPPORTED_PROPERTIES'))
except Exception as exc:
    result['gpu_error'] = str(exc)
print(json.dumps(result, indent=2))
