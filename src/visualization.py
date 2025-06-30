import os
import imageio
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from utils.utils_dataprocessing import get_all_files
from utils.utils import mrc2map, parse_map
from constants import datasetsFolder, depoMapFolder, simuMapFolder


def combine_tensors_to_gif(
    tensors: Dict[str, np.ndarray],
    output_path: str = "combined_tensors.gif",
    fps: int = 2,
    cmap: str = "viridis",
    figsize: tuple = (12, 8),
    dpi: int = 100,
    show_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> None:

    # Check all tensors have the same depth
    depths = [tensor.shape[0] for tensor in tensors.values()]   
    depth = max(depths)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate frames for GIF
    frames = []
    
    for i in range(depth):
        # Create figure and axes
        fig, axes = plt.subplots(1, len(tensors), figsize=figsize, dpi=dpi, sharey=True)
        if len(tensors) == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Plot each tensor's current layer
        ims = []
        for j, (name, tensor) in enumerate(tensors.items()):
            im = axes[j].imshow(tensor[i % tensor.shape[0]], cmap=cmap, vmin=vmin, vmax=vmax)
            ims.append(im)
            axes[j].set_title(f"{name} - Layer {i % tensor.shape[0]}")
        
        # Add colorbar if specified
        if show_colorbar:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(ims[0], cax=cbar_ax)
        
        # Render figure to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        # Close figure to free memory
        plt.close(fig)
    
    # Save frames as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path}")


def align(depoMap, simuMap):
    padded_shape = [max(depoMap.shape[0], simuMap.shape[0]),
                    max(depoMap.shape[1], simuMap.shape[1]),
                    max(depoMap.shape[2], simuMap.shape[2])]
    # 对两个信号进行零填充（将原始数据放在左上角）
    depo_padded = np.zeros(padded_shape, dtype=depoMap.dtype)
    depo_padded[:depoMap.shape[0], :depoMap.shape[1], :depoMap.shape[2]] = depoMap
    simu_padded = np.zeros(padded_shape, dtype=simuMap.dtype)
    simu_padded[:simuMap.shape[0], :simuMap.shape[1], :simuMap.shape[2]] = simuMap
    # 3D FFT
    fft_depo = np.fft.fftn(depo_padded)
    fft_simu = np.fft.fftn(simu_padded)
    # calculate corr
    corr_freq = fft_depo * np.conj(fft_simu)
    # ifftn->real
    corr = np.fft.ifftn(corr_freq).real 

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    dx = peak_idx[0]
    dy = peak_idx[1]
    dz = peak_idx[2]
    print(dx, dy, dz)
    depo_padded = np.roll(depo_padded, shift=-dx, axis=0)
    depo_padded = np.roll(depo_padded, shift=-dy, axis=1)
    depo_padded = np.roll(depo_padded, shift=-dz, axis=2)
    return depo_padded, simu_padded




depo_map_list = get_all_files(depoMapFolder)
simu_map_list = get_all_files(simuMapFolder)
print(f"lenth of depo_map_list:{len(depo_map_list)}")
print(f"lenth of simu_map_list:{len(simu_map_list)}")

print(depo_map_list[188])
print(simu_map_list[188])
depo_data, depoMax = mrc2map(depo_map_list[188], 1.0)

#对齐
simu_data, simuMax = mrc2map(simu_map_list[188], 1.0)
print("normalization")
depo_data = depo_data.clip(min=0.0, max=depoMax) / depoMax
simu_data = simu_data.clip(min=0.0, max=simuMax) / simuMax
print("start aligning")
aligned_depo, aligned_simu = align(depo_data, simu_data)
print(f"original map shape: depo={depo_data.shape}, simu={simu_data.shape}")
print(f"aligned map shape: {aligned_depo.shape} (target map shape: {aligned_depo.shape})")
combine_tensors_to_gif({"aligned_depomap": aligned_depo, "simumap": aligned_simu},
                       fps=2,
                       cmap='grey')