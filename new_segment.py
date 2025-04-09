import os
import numpy as np
import dask_image.imread
import zarr
from cellpose.contrib.distributed_segmentation import distributed_eval

folders = [
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__0hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__4hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__8hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__12hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__16hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__24hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R0hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R4hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R8hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R12hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R16hr__20241213__182319",
    r"C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R24hr__20241213__182319"
]
labels = ["0hr", "4hr", "8hr", "12hr", "16hr", "24hr", "R0hr", "R4hr", "R8hr", "R12hr", "R16hr", "R24hr"]
postfix = r"\morphology_focus\morphology_focus_0001.ome.tif"

model_kwargs = {"gpu": False, "model_type": "cyto3"}
eval_kwargs = {"z_axis": 0, "diameter": 75, "channels": [2, 1], "do_3D": False}
cluster_kwargs = {"n_workers": 1, "ncpus": 1, "memory_limit": "8GB", "threads_per_worker": 1}
save_base = r"C:\Users\ishaa\Documents\Xenium\resegmenting_cytoplasm"


def stack_channels(image, crop):
    return np.stack((image[crop], input_zarr2[crop]), axis=1)

if __name__ == "__main__":
    for folder, label in zip(folders, labels):
        image_path = folder + postfix
        print(image_path)

        data = dask_image.imread.imread(image_path)  # shape: (C, Z, Y, X) or (Z, Y, X)

        out_dir = save_base + fr'{label}_cyto'
        os.makedirs(out_dir, exist_ok=True)

        zarr1_path = out_dir + r'\input.zarr'
        data[2].to_zarr(zarr1_path, overwrite=True, compute=True)
        input_zarr1 = zarr.open(zarr1_path, mode='r')

        zarr2_path = out_dir + r'\second.zarr'
        data[1].to_zarr(zarr2_path, overwrite=True, compute=True)

        global input_zarr2
        input_zarr2 = zarr.open(zarr2_path, mode='r')

        blocksize = (1, 256, 256) # because it stacks the 2nd channel with the preprocess function?
        output_zarr_path = out_dir + r'\segmentation.zarr'

        preprocessing_steps = [(stack_channels, {})]

        print(f'segmenting {label}...\n')

        segments, boxes = distributed_eval(
            input_zarr=input_zarr1,
            blocksize=blocksize,
            write_path=output_zarr_path,
            preprocessing_steps=preprocessing_steps,
            model_kwargs=model_kwargs,
            eval_kwargs=eval_kwargs,
            cluster_kwargs=cluster_kwargs
        )

        print(f'{label} segmentation saved at {output_zarr_path}\n')
