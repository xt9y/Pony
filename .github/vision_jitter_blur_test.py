from pathlib import Path

vision = Path('shaders/vision_compute.hlsl').read_text()

def require(cond, message):
    if not cond:
        raise AssertionError(message)

require('float vision_jitter(' in vision, 'missing stable screen-space jitter helper')
require('uint2 pixel_id' in vision, 'probe integration must receive pixel id for deterministic jitter')
require('jitter - 0.5f' in vision, 'ray-march samples must be shifted within the step')
require('float4 edge_aware_volume_blur(' in vision, 'missing edge-aware volume blur')
require('int2 offsets[5]' in vision, 'blur must be a small 5-tap cross')
require('depth_similarity(center_depth, sample_depth)' in vision, 'blur must reject depth discontinuities')
require('edge_aware_volume_blur(uv, center_depth, fog)' in vision, 'compose path must apply blur after reconstruction')
print('vision jitter/blur contract: PASS')
