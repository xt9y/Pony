from pathlib import Path
import re

path = Path('shader.hlsl')
text = path.read_text()
pattern = r"\n\[numthreads\(8, 8, 1\)\]\nvoid CS_DirectRadiance\(uint3 dispatch_id : SV_DispatchThreadID\) \{.*?\n\}\n\n// -----------------------------------------------------------------------------\n// Screen probes / compacted miss queue\."
replacement = "\n// -----------------------------------------------------------------------------\n// Screen probes / compacted miss queue."
text, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
if count != 1:
    raise SystemExit(f'expected one obsolete CS_DirectRadiance, found {count}')
path.write_text(text)
