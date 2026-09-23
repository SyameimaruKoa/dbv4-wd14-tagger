"""List native WebGPU devices and Windows DXGI adapter indices without inference."""

import argparse
import json

import onnxruntime as ort
import onnxruntime_ep_webgpu as webgpu

from webgpu_vendor import vendor_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    ort.register_execution_provider_library("probe_webgpu", webgpu.get_library_path())
    devices = [item for item in ort.get_ep_devices()
               if item.ep_name == webgpu.get_ep_name()]
    records = []
    for index, item in enumerate(devices):
        hardware = item.device
        records.append({
            "webgpu_index": index,
            "vendor": vendor_name(hardware.vendor, hardware.vendor_id),
            "vendor_id": hardware.vendor_id,
            "device_id": hardware.device_id,
            "metadata": dict(hardware.metadata),
        })
    if args.json:
        print(json.dumps(records, ensure_ascii=False))
    else:
        for item in records:
            metadata = item["metadata"]
            print(f"WebGPU {item['webgpu_index']}: {item['vendor']} "
                  f"{metadata.get('Description', '')}; "
                  f"DirectML/DXGI {metadata.get('DxgiAdapterNumber', 'n/a')}")


if __name__ == "__main__":
    main()
