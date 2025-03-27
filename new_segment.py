import numpy as np
import dask_image.imread
from cellpose.contrib.distributed_segmentation import distributed_eval
import os
import zarr


folders = [
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__0hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__4hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__8hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__12hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__16hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029719__24hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R0hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R4hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R8hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R12hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R16hr__20241213__182319',
    r'C:\Users\ishaa\Documents\Xenium\Data\output-XETG00126__0029731__R24hr__20241213__182319'
]
labels = ['0hr', '4hr', '8hr', '12hr', '16hr', '24hr', 'R0hr', 'R4hr', 'R8hr', 'R12hr', 'R16hr', 'R24hr']
postfix = os.path.join(r'\morphology_focus\morphology_focus_0001.ome.tif')

# these are parameters for distributed_eval(), see the end of this script
model_kwargs = {'gpu': False, 'model_type': 'cyto3'}
eval_kwargs = {'diameter': 75, 'channels': [2, 1], 'do_3D': False}
cluster_kwargs = { # i am running on my laptop so i keep the parameters very mild
    'n_workers': 1,
    'ncpus': 1,
    'memory_limit': '8GB',
    'threads_per_worker': 1,
}
save_base = r'C:\Users\ishaa\Documents\Xenium\resegmenting_cytoplasm'

if __name__ == '__main__': # make sure to keep the __main__ part, otherwise multiprocessing gets mad
    for k, folder in enumerate(folders):
        label = labels[k]
        image_path = folder + postfix
        print(f'processing {label}: {image_path}')

        data = dask_image.imread.imread(image_path)
        out_dir = save_base + fr'{label}_cyto'
        os.makedirs(out_dir, exist_ok=True)

        # i am trying to do this basically exactly the same way that the example (particularly the last one) does
        # https://cellpose.readthedocs.io/en/latest/distributed.html
        zarr_input_path = out_dir + r'\input.zarr'
        zarr_second_path = out_dir + r'\second.zarr'
        print(f'writing zarr input to {zarr_input_path}...')

        # compute=True isn't an issue here, i thought it was but it is not
        data[2].to_zarr(zarr_input_path, overwrite=True, compute=True)
        data[1].to_zarr(zarr_second_path, overwrite=True, compute=True)
        input_zarr = zarr.open(zarr_input_path, mode='r')
        second_zarr = zarr.open(zarr_second_path, mode='r')

        def stack_channels(image, crop):
            return np.stack((image[crop], second_zarr[crop]), axis=1)

        preprocessing_steps = [(stack_channels, {})]
        blocksize = (input_zarr.shape[0], 256, 256)
        output_zarr_path = out_dir + r'\segmentation.zarr'

        print(f'running cyto3 segmentation for {label}')
        segments, boxes = distributed_eval(
            input_zarr=input_zarr, 
            blocksize=blocksize, 
            write_path=output_zarr_path, 
            preprocessing_steps=preprocessing_steps, 
            model_kwargs=model_kwargs, eval_kwargs=eval_kwargs, 
            cluster_kwargs=cluster_kwargs
        )
        print(f'{label} segmentation saved at {output_zarr_path}\n')
