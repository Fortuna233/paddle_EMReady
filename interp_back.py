import argparse  #解析命令行参数
import numbers as np
from utils import parse_map, write_map, inverse_map

def get_args():
    parser = argparse.ArgumentParser(description="调整map分辨率",     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--in_map", "-i", type=str, required=True)
    parser.add_argument("--out_map", "-i", type=str, required=True)
    parser.add_argument("--ref_map", "-i", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print("加载输入地图")
    in_map = args.in_map
    out_map = args.out_map
    ref_map = args.ref_map

    print("# Loading the input map...")
    map_in, origin_in, nxyz_in, voxel_size_in, _ = parse_map(in_map, ignorestart=False, apix=None)
    print(f"# Input map dimensions: {nxyz_in}")
    # 原点坐标是否与体素对齐，若并未，则进行修正
    try:
        assert np.all(np.abs(np.round(origin_in / voxel_size_in) - origin_in / voxel_size_in) < 1e-4)
    except AssertionError:
        origin_shift = (np.round(origin_in / voxel_size_in) - origin_in / voxel_size_in)
        map_in, origin_in, nxyz_in, voxel_size_in, _ = parse_map(in_map, ignorestart=False, apix=None, origin_shift=origin_shift)
        assert np.all(np.abs(np.round(origin_in / voxel_size_in) - origin_in / voxel_size_in) < 1e-4)
    
    print("load reference map...")
    _, _, nxyz_ref, voxel_size_ref, _ = parse_map(ref_map, ignorestart=False, apix=None)
    print(f"# References map dimessions: {nxyz_ref}")

    print(f"# Interpolating the voxel size from {voxel_size_in} back to {voxel_size_ref}")
    origin_shift = [0.0, 0.0, 0.0]
    try:
        assert np.all(np.abs(np.round(origin_in / voxel_size_ref) - origin_in / voxel_size_ref) < 1e-4)
    except AssertionError:
        origin_shift = (np.round(origin_in / voxel_size_ref) - origin_in / voxel_size_ref) * voxel_size_ref
    map, origin, nxyz, voxel_size = inverse_map(map_in, nxyz_in, origin_in, voxel_size_in, voxel_size_ref, origin_shift)
    assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)
    nxyzstart = np.round(origin / voxel_size).astype(np.int64)
    print(f"# Output map dimensions: {nxyz}")
    write_map(out_map, map, voxel_size, nxyzstart=nxyzstart)


if __name__ == "__main__":
    main()
    
    