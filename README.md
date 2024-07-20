# AutoMask

AutoMask lets one _naively_ label a raster image. AutoMask simply requires the
path to the raster and elevation (DEM) data (ideally in `tif` formats), and can spit out the labeled data in an
organized `train/validation/test` folder structure (suitable
for [`tensorflow.keras.preprocessing.image.ImageDataGenerator`](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory)).

<video src='https://github.com/user-attachments/assets/47ad7dad-ced3-42be-9876-21df0339d080'> </video>
<p align="center">How labeling with AutoMask works in a nutshell</p>

## Usage

The video above should highlight the main usage of AutoMask. One can instantiate an `AutoMask` instance by providing it
two required arguments: 1. path to raster file 2. path to elevation (DEM) raster file.

```python
from landslide_analysis_helpers import AutoMask

mask = AutoMask("raster_image.tif", "elevation_dem.tif")
```

### Plotting

`mask.show_comparison` or `mask.show` can be run with some tuning parameters to plot the labeled binary
mask. `mask.show` doesn't show any comparison with the original raster (i.e., computation is faster), but ideally, one
would find `mask.show_comparison` more convenient for labeling.

Both `mask.show_comparison` and `mask.show` supports the following arguments:

1. **`std_deviation_distance_factor`**: A required argument. Essentially, this parameter controls the binary threshold
   in intensity histogram distribution. `std_deviation_distance_factor == 1` implies the threshold is exactly at the
   maximum intensity value. As the factor value is increased, so will the minimum threshold for masking out intensity.
2. **`slope_angle`**: A required argument. `AutoMask` computes slope angles from elevation (DEM) raster. The higher the
   angle (in degrees), the less less-inclined regions will be masked.
3. **`band`**: The RGB band color to use for `std_deviation_distance_factor`. Default is "blue". Supports values: `0` ("
   red"),
   `1` ("green") and `2` ("blue") in integer or string.
4. **`noise_suppression_strategy`**: AutoMask can apply a combination of binary dilation, binary erosion and Gaussian
   blur filter for removing noise. This argument is optional. The following `noise_suppression_strategy` value would
   apply Gaussian blur with variance
   1, binary erosion with 9 iterations, and binary dilation with 15 iterations, in an ordered manner.

```python
noise_suppression_strategy = {
    "gaussian_blur": 1,
    "binary_erosion": 9,
    "binary_dilation": 15,
}
```

5. **`dpi`**: DPI value for plots. Default is 150.

### Raw image matrix

`mask.get_binary_mask` can be called with the above-explained parameters to get just the output image matrix instead of
a plot. This can be useful if one wishes to plot with their own customization.

### Export patches

Once a satisfactory label has been found, `mask.export_patches` can be run to export the binary mask image into smaller
images.

```python
from pathlib import Path

params = dict(std_deviation_distance_factor=2,
              slope_angle=30,
              band="blue",
              noise_suppression_strategy={
                  "gaussian_blur": 1,
                  "binary_erosion": 9,
                  "binary_dilation": 15,
              }
              )

export_dir = Path(".")  # Current directory
mask.export_patches(
    export_dir,
    mask.get_binary_mask(**params),
    width=512,
    height=512,
    name_suffix="main",
    convert_to_png=True,
    use_4d=False,
    use_patchify=True,
)
```

The first strictly positional argument `export_dir` defines the path to export the images which in this examples is the
current directory. `width` and `height` define the patch size. If `use_patchify` is `True`, a
non-overlapping `width x height` images will be produced using `[patchify](https://pypi.org/project/patchify/)`. If it
is `False`, then [`gdal_retile`](https://gdal.org/programs/gdal_retile.html) will be used to patch. `gdal_retile` may
not produce patches of uniform size, i.e., not all images will have `width x height` dimension. `name_suffix` defines
the suffix to add to the folder where everything will be exported. `name_suffix="main"` here would export everything
in `./automask_export_main` folder. If `use_4d` is `True`, then the original raster images will be patched as 4D
images where the 4th band is a raster slope image calculated from the elevation (DEM) raster. One has to make sure
though their image pre-processor knows how to handle 4D images.

## Features

There are a few features that `AutoMask` comes with that make certain things easier to work with.

### Cache

`AutoMask` utilizes `__dict__` to cache loaded raster images from their paths, and computation-heavy images (
like `mask.slope_image`). Thus, it avoids re-computation and re-reading the rasters
from the disk. The caches are deleted if the raster file paths are changed dynamically. This makes sure that a wrong
cache will not be used. The following properties are
cached: `mask.raster_rio`, `mask.elevation_rio`, `mask.slope_image`, `mask.raster_image_with_slope`.

### Convert to Tensorflow-friendly directory structure

Exported patches can are not immediately structured, so it can be fed
to [`tensorflow.keras.preprocessing.image.ImageDataGenerator`](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory).

```python
from landslide_analysis_helpers import transform_dir_to_tf_dir_structure

transform_dir_to_tf_dir_structure(export_dir / "automask_export_main")
# Notice, "automask_export_main" is the name of the folder where all raster image patches have been exported before.

```

## Remarks

This small library was made as part of my "Data Science Lab" course at my university. I mainly built it to automate some
of the processes involved in labeling. Unlike my other works, the codebase is not production-ready.
