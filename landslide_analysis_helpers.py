import os
import subprocess
from datetime import datetime
from functools import partial, cached_property
from pathlib import Path
from typing import Union, Type, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import patchify
import rasterio
import richdem
import scipy.ndimage as ndi
import splitfolders
from PIL import Image
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rio_color import operations
from skimage import segmentation

DEFAULT_CONTRAST = 10
# noinspection SpellCheckingInspection
PROJECTION_FILE_PATH: Path = (
    Path("./Leibniztal_Einzugsgebiet.prj").expanduser().absolute()
)
GDAL_TRANSLATE_PATH: Path = Path("./.conda/bin/gdal_translate")
GDAL_RETILE_PATH: Path = Path("./.conda/bin/gdal_retile.py")


# noinspection SpellCheckingInspection
def convert_tif_to_png(
    directory: Path, /, gdal_translate_path: Path = GDAL_TRANSLATE_PATH
):
    source_extension = "tif"
    target_extension = "png"
    GDAL_PAM_ENABLED_ENVIRON: str = "GDAL_PAM_ENABLED"
    os.environ[GDAL_PAM_ENABLED_ENVIRON] = "NO"
    for file in directory.expanduser().resolve().iterdir():
        if file.suffix == f".{source_extension}":
            subprocess.run(
                [
                    gdal_translate_path,
                    "-of",
                    f"{target_extension}",
                    file,
                    f"{file.parent / file.stem}.{target_extension}",
                ],
                capture_output=True,
                check=True,
            )
            file.unlink()
    os.environ.pop(GDAL_PAM_ENABLED_ENVIRON, None)


def transform_dir_to_tf_dir_structure(root_dir: Path, /):
    import shutil

    tf_train_img_path = root_dir / "train_images" / "train"
    tf_train_mask_path = root_dir / "train_masks" / "train"
    tf_val_img_path = root_dir / "val_images" / "val"
    tf_val_mask_path = root_dir / "val_masks" / "val"
    for _ in [
        tf_train_img_path,
        tf_train_mask_path,
        tf_val_img_path,
        tf_val_mask_path,
    ]:
        _.mkdir(exist_ok=True, parents=True)
    for src, target in (
        (root_dir / "train" / "images", tf_train_img_path),
        (root_dir / "train" / "masks", tf_train_mask_path),
        (root_dir / "val" / "images", tf_val_img_path),
        (root_dir / "val" / "masks", tf_val_mask_path),
    ):
        if not src.exists():
            raise ValueError(
                f"{src} doesn't exist! "
                f"Root dir {root_dir} should have the following structure:\n"
                f"- train\n"
                f"     - images\n"
                f"     - masks\n"
                f"- val\n"
                f"     - images\n"
                f"     - mask"
            )
        src.rename(target)
    shutil.rmtree(root_dir / "train")
    shutil.rmtree(root_dir / "val")


def add_contrast(image: np.ndarray, contrast: int) -> np.ndarray:
    if contrast is not None:
        if not isinstance(contrast, int):
            raise ValueError("Contrast must be an integer.")
        return operations.sigmoidal(image / 255, contrast, 0.5)


class AutoMask:
    CRS_VAL = 31255
    DEFAULT_EXTENSION: str = "tif"

    def __init__(
        self, raster_data_path: Union[str, Path], elevation_data_path: Union[str, Path]
    ):
        self.raster_data_path = raster_data_path
        self.elevation_data_path = elevation_data_path

    @property
    def raster_data_path(self):
        return self._raster_data_path

    @raster_data_path.setter
    def raster_data_path(self, value):
        if not isinstance(value, Path):
            try:
                value = Path(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid path for raster data path '{value}'") from e
        if value.expanduser().absolute() != getattr(self, "_raster_data_path", None):
            self.__dict__.pop("_raster_rio", None)
            self.__dict__.pop("_elevation_rio", None)
            self.__dict__.pop("slope_image", None)
            self.__dict__.pop("raster_image_with_slope", None)
        self._raster_data_path = value.expanduser().absolute()
        if not self._raster_data_path.exists():
            raise ValueError(
                f"Raster data in raster data path '{self._raster_data_path}' "
                f"must exist!"
            )

    @property
    def elevation_data_path(self):
        return self._elevation_data_path

    @elevation_data_path.setter
    def elevation_data_path(self, value):
        if not isinstance(value, Path):
            try:
                value = Path(value)
            except ValueError as e:
                raise ValueError(
                    f"Invalid path for elevation data path '{value}'"
                ) from e
        if value.expanduser().absolute() != getattr(self, "_elevation_data_path", None):
            self.__dict__.pop("_raster_rio", None)
            self.__dict__.pop("_elevation_rio", None)
            self.__dict__.pop("slope_image", None)
            self.__dict__.pop("raster_image_with_slope", None)
        self._elevation_data_path = value.expanduser().absolute()
        if not self._elevation_data_path.exists():
            raise ValueError(
                f"Elevation data in elevation data path '{self._elevation_data_path}' "
                f"must exist!"
            )

    @classmethod
    def get_crs(cls):
        return CRS().from_epsg(cls.CRS_VAL)

    @cached_property
    def _raster_rio(self) -> Tuple[np.ndarray, dict]:
        with rasterio.open(self.raster_data_path, mode="r+") as raster_rio:
            raster_rio.crs = self.get_crs()
            raster_meta = raster_rio.meta
            self._raster_unmodified_metadata = raster_rio.meta
            # For some reason, modifying raster_rio.meta directly doesn't make any effect!
            raster_meta.update({"transform": Affine(*raster_rio.get_transform())})
            return raster_rio.read(), raster_meta

    @property
    def raster_rio(self) -> Tuple[np.ndarray, dict]:
        return self._raster_rio

    @raster_rio.setter
    def raster_rio(self, value):
        raise ValueError("raster_rio cannot be modified!")

    @property
    def raster_image(self) -> np.ndarray:
        raster, _ = self.raster_rio
        return np.moveaxis(raster, 0, -1)

    @property
    def raster_image_metadata(self) -> dict:
        _, metadata = self.raster_rio
        return metadata

    @cached_property
    def _elevation_rio(self) -> Tuple[np.ndarray, dict]:
        metadata = self.raster_image_metadata
        with rasterio.open(self.elevation_data_path, mode="r+") as elevation_rio:
            elevation_rio.crs = self.get_crs()
            return elevation_rio.read(
                out_shape=(
                    elevation_rio.count,
                    metadata["height"],
                    metadata["width"],
                ),
                resampling=Resampling.cubic_spline,
            ), elevation_rio.meta

    @property
    def elevation_rio(self) -> Tuple[np.ndarray, dict]:
        return self._elevation_rio

    @raster_rio.setter
    def raster_rio(self, value):
        raise ValueError("elevation_rio cannot be modified!")

    @property
    def elevation_image(self) -> np.ndarray:
        elevation, _ = self.elevation_rio
        return np.squeeze(np.moveaxis(elevation, 0, -1))

    @property
    def elevation_metadata(self) -> dict:
        _, metadata = self.elevation_rio
        return metadata

    @staticmethod
    def get_band(
        band: Union[int, str], return_type: Union[Type[int], Type[str]] = int
    ) -> Union[int, str]:
        BANDS = {0: "red", 1: "green", 2: "blue"}
        if isinstance(band, str):
            try:
                band = {v: k for k, v in BANDS.items()}[band.lower()]
            except KeyError as e:
                raise ValueError(
                    "band can only be integer literals 0, 1 or 2, "
                    "or string literals 'red', 'green', or 'blue'."
                ) from e
            else:
                if issubclass(return_type, int):
                    return band
                elif issubclass(return_type, str):
                    return BANDS[band]
        elif isinstance(band, int):
            if band not in BANDS.keys():
                raise ValueError(
                    "band can only be integer literals 0, 1 or 2, "
                    "or string literals 'red', 'green', or 'blue'."
                )
            if issubclass(return_type, int):
                return band
            elif issubclass(return_type, str):
                return BANDS[band]
            else:
                raise ValueError("Return type must be integer or str type.")

    # noinspection SpellCheckingInspection
    def get_band_color(self, band: Union[int, str]):
        from collections import namedtuple

        ColorInfo = namedtuple("ColorInfo", ["red", "green", "blue", "hex"])
        indicators = {
            0: ColorInfo(255, 0, 123, "#ff007b"),
            1: ColorInfo(152, 231, 181, "#49d47c"),
            2: ColorInfo(46, 206, 255, "#2eceff"),
        }
        return indicators[self.get_band(band, return_type=int)]

    def get_single_band_from_raster(self, band: Union[int, str] = 2) -> np.ndarray:
        return self.raster_image[:, :, self.get_band(band)]

    def get_raster_histogram(self, band: Union[int, str] = 2) -> np.ndarray:
        return ndi.histogram(
            self.get_single_band_from_raster(band), min=0, max=255, bins=256
        )

    @cached_property
    def slope_image(self) -> np.ndarray:
        landslide_beau = richdem.rdarray(self.elevation_image, no_data=-9999)
        landslide_beau.geotransform = self.raster_image_metadata["transform"]
        return richdem.TerrainAttribute(landslide_beau, attrib="slope_degrees")

    @cached_property
    def raster_image_with_slope(self) -> np.ndarray:
        return np.dstack((self.raster_image, self.slope_image))

    def get_suitable_intensity(self, std_deviation_distance_factor: float) -> int:
        histogram_dist = self.get_raster_histogram()
        max_val_index = np.argmax(histogram_dist == np.max(histogram_dist))
        reasonable_intensity_window = histogram_dist[max_val_index + 1 :]
        reasonable_freq_val = reasonable_intensity_window[
            np.argmax(
                reasonable_intensity_window
                <= abs(
                    np.max(histogram_dist)
                    - std_deviation_distance_factor * np.std(histogram_dist)
                )
            )
        ]
        # noinspection PyAttributeOutsideInit
        self._latest_binary_threshold_value = np.argmax(
            histogram_dist == reasonable_freq_val
        )
        return self._latest_binary_threshold_value

    def get_masked_raster_image(
        self, std_deviation_distance_factor: float, band: Union[int, str] = 2
    ) -> np.ndarray:
        single_band = self.get_single_band_from_raster(band)
        return np.where(
            single_band < self.get_suitable_intensity(std_deviation_distance_factor),
            single_band,
            0,
        )

    def get_masked_raster_image_with_elevation(
        self,
        std_deviation_distance_factor: float,
        slope_angle: Union[int, float],
        band: Union[int, str] = 2,
        apply_binary_threshold: bool = True,
    ) -> np.ndarray:
        mask = np.where(
            self.slope_image < slope_angle,
            self.get_single_band_from_raster(band),
            self.get_masked_raster_image(std_deviation_distance_factor, band),
        ).astype(np.uint8)
        if not apply_binary_threshold:
            return mask
        return np.where(
            mask == 0,
            255,
            0,
        )

    def get_overlayed_masked_raster(
        self, mask: np.ndarray, mask_band: Union[int, str]
    ) -> np.ndarray:
        r_color_val, g_color_val, b_color_val, _ = self.get_band_color(mask_band)
        mask = np.where(mask != 255, -1, mask)
        r = np.where(mask == 255, r_color_val, mask)  # r: 255
        g = np.where(mask == 255, g_color_val, mask)  # g: 0
        b = np.where(mask == 255, b_color_val, mask)  # b: 123
        a = 255 * (np.dstack((r, g, b))[:, :, :3] != -1).any(axis=2)
        back = Image.fromarray(self.raster_image).convert("RGBA")
        fore = Image.fromarray(np.dstack((r, g, b, a)).astype(np.uint8))
        return np.array(Image.alpha_composite(back, fore))

    @staticmethod
    def suppress_noise(image: np.ndarray, strategy: dict) -> np.ndarray:
        original_image, preference = image.copy(), {}
        BINARY_EROSION = "binary_erosion"
        BINARY_DILATION = "binary_dilation"
        GAUSSIAN_BLUR = "gaussian_blur"
        methods = {
            BINARY_EROSION: ndi.binary_erosion,
            BINARY_DILATION: ndi.binary_dilation,
            GAUSSIAN_BLUR: ndi.gaussian_filter,
        }
        try:
            for key, val in strategy.items():
                if key in (BINARY_EROSION, BINARY_DILATION):
                    preference[key] = partial(methods[key], iterations=val)
                elif key == GAUSSIAN_BLUR:
                    preference[key] = partial(methods[key], sigma=val)
                else:
                    ValueError(
                        "Strategy contains unsupported noise suppression method."
                    )
        except KeyError as e:
            raise ValueError("Invalid strategy for suppress_noise method.") from e
        else:
            for name, func in preference.items():
                image = func(input=image)
            return segmentation.watershed(image, original_image, mask=image)

    def get_binary_mask(
        self,
        std_deviation_distance_factor: float,
        slope_angle: Union[int, float],
        band: Union[int, str] = 2,
        noise_suppression_strategy: Optional[dict] = None,
    ) -> np.ndarray:
        mask = self.get_masked_raster_image_with_elevation(
            std_deviation_distance_factor, slope_angle, band
        )
        if noise_suppression_strategy is None:
            return mask
        return self.suppress_noise(mask, strategy=noise_suppression_strategy)

    def show(
        self,
        std_deviation_distance_factor: float,
        slope_angle: Union[int, float],
        band: Union[int, str] = 2,
        noise_suppression_strategy: Optional[dict] = None,
        dpi: int = 150,
    ) -> None:
        latest_binary_threshold = getattr(
            self,
            "_latest_binary_threshold_value",
            self.get_suitable_intensity(std_deviation_distance_factor),
        )
        figure = plt.figure(figsize=(10, 10), dpi=dpi)  # noqa: F841
        ax = figure.add_subplot(1, 1, 1)  # noqa: F841
        custom_font = {"fontname": "Raleway"}
        plt.imshow(
            self.get_binary_mask(
                std_deviation_distance_factor,
                slope_angle,
                band,
                noise_suppression_strategy,
            ),
            cmap="gray",
        )
        plt.suptitle(
            f"{self.get_band(band, return_type=str).capitalize()} mask in binary"
            f"\nwith elevation consideration",
            fontsize=18,
            **custom_font,
        )
        noise_sup_str_text = (
            " →️ ".join(f"{k} ({v})" for k, v in noise_suppression_strategy.items())
            if noise_suppression_strategy
            else None
        )
        ax.set_title(
            f"Binary threshold: x={latest_binary_threshold}, "
            f"slope angle: {slope_angle}°,\n"
            f"noise suppression with: {noise_sup_str_text}",
            fontsize=12,
            pad=10,
        )

    def show_comparison(
        self,
        std_deviation_distance_factor: float,
        slope_angle: Union[int, float],
        band: Union[int, str] = 2,
        noise_suppression_strategy: Optional[dict] = None,
        dpi: int = 150,
    ) -> None:
        latest_binary_threshold = getattr(
            self,
            "_latest_binary_threshold_value",
            self.get_suitable_intensity(std_deviation_distance_factor),
        )
        mask = self.get_masked_raster_image_with_elevation(
            std_deviation_distance_factor, slope_angle, self.get_band(band)
        )
        figure = plt.figure(figsize=(12, 12), dpi=dpi)
        # Original image
        ax1 = figure.add_subplot(2, 2, 1)
        ax1.imshow(add_contrast(self.raster_image, contrast=DEFAULT_CONTRAST))
        # Histogram
        _, _, _, hex_val = self.get_band_color(band)
        ax2 = figure.add_subplot(2, 2, 2)
        ax2.plot(hist := self.get_raster_histogram(band), color="#242424")
        ax2.set_facecolor((0.961, 0.961, 0.961))
        ax2.set_xticks(np.arange(0, 256, 25))
        ax2.vlines(
            latest_binary_threshold,
            ymin=0,
            ymax=hist.max(),
            linewidth=2,
            color=hex_val,
            linestyles="dashed",
        )
        ax2.grid()
        ax2.set_ylabel("Frequency")
        ax2.set_xlabel("Pixel values (intensity)")
        # Overlay image
        ax3 = figure.add_subplot(2, 2, 3)
        if noise_suppression_strategy is not None:
            mask = self.suppress_noise(mask, noise_suppression_strategy)
        ax3.imshow(
            add_contrast(
                self.get_overlayed_masked_raster(mask, mask_band=band),
                contrast=DEFAULT_CONTRAST,
            )
        )
        # Binary image
        ax4 = figure.add_subplot(2, 2, 4)
        ax4.imshow(mask, cmap="gray")
        ax4.set_yticklabels([])
        custom_font = {"fontname": "Raleway"}
        ax1.set_title("Original image", fontsize=14, **custom_font, pad=10)
        ax2.set_title(
            f"Intensity distribution\nbinary threshold: x={latest_binary_threshold}",
            fontsize=14,
            **custom_font,
            pad=10,
        )
        noise_sup_str_text = (
            "\n"
            + " →️ ".join(f"{k} ({v})" for k, v in noise_suppression_strategy.items())
            if noise_suppression_strategy
            else None
        )
        ax3.set_title(
            "Original image overlayed with mask.\n"
            f"Slope angle: {slope_angle}°, noise suppression with: "
            f"{noise_sup_str_text}",
            fontsize=12,
            pad=10,
        )
        ax4.set_title("Binary mask", fontsize=14, **custom_font, pad=10)
        plt.subplots_adjust(wspace=0.25, hspace=0.1)
        plt.suptitle(
            f"{self.get_band(band, return_type=str).capitalize()} channel mask\n"
            f"with elevation consideration",
            fontsize=17,
            **custom_font,
        )

    # noinspection SpellCheckingInspection
    def export_patches(
        self,
        export_dir: Union[Path, str],
        /,
        binary_mask: np.ndarray,
        width: int,
        height: int,
        *,
        name_suffix: Optional[str] = None,
        split_ratio: tuple[float, float, ...] = (0.75, 0.25),
        convert_to_png: bool = False,
        use_patchify: bool = False,
        gdal_retile_path: Path = GDAL_RETILE_PATH,
        projection_file_path: Path = PROJECTION_FILE_PATH,
        use_4d: bool = False,
    ) -> None:
        if not isinstance(export_dir, Path):
            try:
                export_dir = Path(export_dir)
            except ValueError as e:
                raise ValueError(f"Invalid path for export path '{export_dir}'") from e
            else:
                if not export_dir.exists():
                    raise ValueError(
                        f"Export directory '{export_dir}' must already exist!"
                    )
        export_dir = export_dir.expanduser().resolve()
        if use_patchify is False:
            if gdal_retile_path is None:
                raise ValueError(
                    "gdal_retile_path must be passed if use_patchify is false."
                )
            if projection_file_path is None:
                raise ValueError(
                    "projection_file_path must be passed if use_patchify is false."
                )
        try:
            meta = self._raster_unmodified_metadata
        except AttributeError as e:
            raise AttributeError(
                "Unexpected error. Make sure method 'raster_image' is called first!"
            ) from e
        if len(binary_mask) != 1 and binary_mask.shape[0] != 1:
            binary_mask_proper = np.expand_dims(binary_mask, axis=0)
            meta.update({"count": 1})
        else:
            binary_mask_proper = binary_mask
        date = datetime.now()
        date_time_suffix = f'{date.strftime("%Y-%m-%d")}_{date.strftime("%H%M%S")}'
        if name_suffix is None:
            suffix: str = date_time_suffix
        else:
            if not isinstance(name_suffix, str) or name_suffix == "":
                raise ValueError("name_suffix can only be a non-empty string.")
            suffix = name_suffix
        export_dir = export_dir / f"automask_export_{suffix}"
        export_dir.mkdir(exist_ok=True)
        with rasterio.open(
            mask_export_path := (export_dir / f"mask_{date_time_suffix}.tif"),
            "w",
            **meta,
        ) as dest:
            dest.write(binary_mask_proper)
        masks_dir = export_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        if use_patchify:
            mask_patch_container = []
            mask_patches = patchify.patchify(binary_mask, (height, width), step=height)
            # height x width instead of width x height because that's what follows numpy.shape!
            # I.e., (row size, column size) ==  (height, width)
            for i in range(mask_patches.shape[0]):
                for j in range(mask_patches.shape[1]):
                    mask_patch_container.append(mask_patches[i, j, :, :])
            for i, img in enumerate(mask_patch_container, start=1):
                Image.fromarray(img.astype(np.uint8)).save(
                    masks_dir / f"mask_{self.raster_data_path.stem}_"
                    f"{i:0{len(str(len(mask_patch_container)))}d}.{self.DEFAULT_EXTENSION}",
                )
        else:
            subprocess.run(
                [
                    "python3",
                    gdal_retile_path,
                    "-ps",
                    f"{width}",
                    f"{height}",
                    "-targetDir",
                    masks_dir,
                    "-s_srs",
                    projection_file_path,
                    mask_export_path,
                ],
                capture_output=True,
                check=True,
            )
        original_image_patches_dir = export_dir / "images"
        original_image_patches_dir.mkdir(exist_ok=True)
        if use_patchify:
            original_image_patch_container = []
            original_image_patches = patchify.patchify(
                self.raster_image if not use_4d else self.raster_image_with_slope,
                (height, width, 3 if not use_4d else 4),
                step=height,
            )
            for i in range(original_image_patches.shape[0]):
                for j in range(original_image_patches.shape[1]):
                    original_image_patch_container.append(
                        original_image_patches[i, j, :, :][0]
                    )
            for i, img in enumerate(original_image_patch_container, start=1):
                Image.fromarray(img.astype(np.uint8)).save(
                    original_image_patches_dir / f"{self.raster_data_path.stem}_"
                    f"{i:0{len(str(len(original_image_patch_container)))}d}.{self.DEFAULT_EXTENSION}",
                )
        else:
            raster_data_path = self.raster_data_path
            if use_4d:
                meta.update({"count": 4})
                with rasterio.open(
                    raster_data_path := (
                        export_dir / f"rater_4d_with_elevation_{date_time_suffix}.tif"
                    ),
                    "w",
                    **meta,
                ) as dest:
                    dest.write(np.moveaxis(self.raster_image_with_slope, -1, 0))
            subprocess.run(
                [
                    "python3",
                    gdal_retile_path,
                    "-ps",
                    f"{width}",
                    f"{height}",
                    "-targetDir",
                    original_image_patches_dir,
                    "-s_srs",
                    projection_file_path,
                    raster_data_path,
                ],
                capture_output=True,
                check=True,
            )
        if convert_to_png is True:
            for main_dir in (original_image_patches_dir, masks_dir):
                convert_tif_to_png(main_dir)
        splitfolders.ratio(
            export_dir,
            output=export_dir,
            seed=1337,
            ratio=split_ratio,
            group_prefix=None,
            move=True,
        )
        masks_dir.rmdir()
        original_image_patches_dir.rmdir()
