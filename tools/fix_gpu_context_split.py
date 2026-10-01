#!/usr/bin/env python3
from pathlib import Path

path = Path("gpu.c")
source = path.read_text(encoding="utf-8")

for owner in ("frame", "work"):
    for field in (
        "descriptor_pool",
        "temporary_descriptors",
        "temporary_descriptor_num",
        "temporary_descriptor_cap",
        "temporary_buffers",
        "temporary_buffer_num",
        "temporary_buffer_cap",
    ):
        source = source.replace(f"{owner}->gpu->{field}", f"{owner}->{field}")

if "frame->gpu->" in source or "work->gpu->" in source:
    leftovers = [line.strip() for line in source.splitlines() if "frame->gpu->" in line or "work->gpu->" in line]
    raise SystemExit("unexpected context redirection remains:\n" + "\n".join(leftovers[:20]))

path.write_text(source, encoding="utf-8")
