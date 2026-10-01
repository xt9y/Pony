#!/usr/bin/env python3
from pathlib import Path
import argparse


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return Path(path).read_text()


def red(task):
    s = read("sdf.c")
    h = read("shader.hlsl")
    r = read("render.c")
    g = read("game.h")
    b = read("build.c")
    if task == 1:
        require("uint64_t source_order;" not in s, "task1 unexpectedly implemented")
    elif task == 2:
        require("void CS_SelectWorldProbeUpdates" not in h, "task2 selector unexpectedly implemented")
        require("CS_SelectWorldProbeUpdates" not in b, "task2 build job unexpectedly implemented")
    elif task == 3:
        require("static void prepare_world_probe_updates" in r, "task3 CPU scheduler unexpectedly absent")
        require("world_radiance_select_pipeline" not in g, "task3 selector pipeline unexpectedly implemented")
    else:
        raise AssertionError(f"unknown red task {task}")


def task1():
    s = read("sdf.c")
    t = read("tests/stage10_world_cache.c")
    require("uint64_t source_order;" in s, "candidate source order missing")
    require("if (a->source_order < b->source_order) return -1;" in s, "source-order ascending comparator missing")
    require("if (a->source_order > b->source_order) return 1;" in s, "source-order comparator upper branch missing")
    surface = s.index("if (a->surface_id < b->surface_id) return -1;")
    order = s.index("if (a->source_order < b->source_order) return -1;")
    require(surface < order, "source order must be final tie-break after surface id")
    require("uint64_t source_order = 0u;" in s, "source traversal counter missing")
    require("candidate.source_order = source_order++;" in s, "candidate source-order assignment missing")
    require("for (uint32_t repeat = 0u; repeat < 8u; ++repeat)" in t, "permanent repeated determinism regression missing")


def task2():
    h = read("shader.hlsl")
    b = read("build.c")
    entry = h.index("void CS_SelectWorldProbeUpdates")
    prefix = h[max(0, entry - 80):entry]
    require("[numthreads(64, 1, 1)]" in prefix, "selector must use exactly 64 threads")
    require("uint base = Pass.dispatch.x % active_count;" in h, "rotating selector base missing")
    require("for (uint logical_offset = lane; logical_offset < active_count; logical_offset += 64u)" in h, "disjoint lane scan missing")
    require("uint probe_index = (base + logical_offset) % active_count;" in h, "rotated probe mapping missing")
    require("return (Pass.dispatch.x - state.state.x) & 0x00ffffffu;" in h, "24-bit frame age missing")
    require("return 3u;" in h and "return 2u;" in h and "return age_frames >= 2u * sweep_frames ? 1u : 0u;" in h, "priority classes missing")
    require("candidate.statistics.y > best.statistics.y" in h, "residual priority missing")
    require("candidate.statistics.x < best.statistics.x" in h, "confidence priority missing")
    require("candidate_age > best_age" in h, "age priority missing")
    require("return candidate_index < best_index;" in h, "index tie-break missing")
    require("InterlockedAdd(RayCounters[3], 1u, slot);" in h, "GPU update-list append missing")
    require("if (slot < Radiance.world_probe_config.y) RadianceUpdateList[slot] = best_index;" in h, "selector budget bound missing")
    require("float residual_sum = 0.0f;" in h, "residual accumulation missing")
    require("residual_sum += length(sample - old) / (1.0f + length(sample));" in h, "relative directional delta missing")
    require("float measured_residual = residual_sum / max((float)WorldProbeDirectionCount(), 1.0f);" in h, "mean residual missing")
    require("state.statistics.y = established ? lerp(state.statistics.y, measured_residual, 0.25f) : measured_residual;" in h, "residual smoothing missing")
    require(b.count('"CS_SelectWorldProbeUpdates", "compute", "radiance_world_select.cs.spv", NULL') == 1, "selector shader build job must exist exactly once")
    require("if (stable_previous && state.state.x == Pass.dispatch.x" in h, "Stage10 stable previous-bank behavior was lost")


def task3():
    g = read("game.h")
    r = read("render.c")
    h = read("shader.hlsl")
    world_struct = g[g.index("typedef struct RADIANCE_WORLD_RESOURCES"):g.index("} RADIANCE_WORLD_RESOURCES;")]
    require("cpu_update_list" not in world_struct, "CPU update-list ownership remains")
    require("update_count" not in world_struct, "CPU world update count remains")
    require("update_cursor" not in world_struct, "CPU round-robin cursor remains")
    require("NriPipeline *world_radiance_select_pipeline;" in g, "selector pipeline handle missing")
    require("static void prepare_world_probe_updates" not in r, "CPU round-robin function remains")
    require("world_update_data" not in r, "per-frame world update data remains")
    require("upload_world_updates" not in r, "per-frame world update upload remains")
    require("cpu_update_list" not in r, "CPU update-list usage remains")
    require('"build/shaders/radiance_world_select.cs.spv"' in r, "selector pipeline creation missing")
    require("&renderer->world_radiance_select_pipeline" in r, "selector pipeline ownership missing")
    require("renderer->world_radiance_select_pipeline," in r, "selector pipeline destruction missing")
    require("RayCounters[3] = 0u;" in h, "wavefront reset must clear world selection count")
    require("RayCounters[3] = Pass.range.x;" not in h, "old CPU-fed world update count remains")

    begin = r.index("if (renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_WORLD_CACHE)")
    end = r.index("bind_wavefront(renderer, command_buffer, renderer->wavefront_budget_pipeline);", begin)
    block = r[begin:end]
    select = block.index("renderer->world_radiance_select_pipeline")
    first_dispatch = block.index("CmdDispatch", select)
    first_barrier = block.index("barrier_wavefront_buffers", first_dispatch)
    update = block.index("renderer->world_radiance_pipeline", first_barrier)
    second_dispatch = block.index("CmdDispatch", update)
    world_barrier = block.index("barrier_world_radiance", second_dispatch)
    require(select < first_dispatch < first_barrier < update < second_dispatch < world_barrier, "selector/barrier/update ordering incorrect")
    dispatch_literal = "(NriDispatchDesc){.workGroupNumX = 1, .workGroupNumY = 1, .workGroupNumZ = 1}"
    require(block.count(dispatch_literal) == 2, "selector and world update must each dispatch exactly one workgroup")


def full():
    task1()
    task2()
    task3()
    b = read("build.c")
    require(b.count("CS_UpdateWorldRadianceCache") == 1, "world update shader job must remain unique")
    require(b.count("CS_SelectWorldProbeUpdates") == 1, "selector shader job must remain unique")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--red-task", type=int)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.red_task is not None:
        red(args.red_task)
    elif args.task == 1:
        task1()
    elif args.task == 2:
        task2()
    elif args.task == 3:
        task3()
    else:
        full()
    print("stage11 architecture PASS")


if __name__ == "__main__":
    main()
