"""Resolve WebGPU vendor names when an EP reports only a PCI vendor ID."""

VENDORS = {0x10DE: "nvidia", 0x8086: "intel", 0x1002: "amd"}


def vendor_name(name, vendor_id):
    return VENDORS.get(vendor_id, (name or "").lower())
