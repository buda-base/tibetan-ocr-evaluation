import os
import cv2
import json
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum


class Encoding(Enum):
    Unicode = 0
    Wylie = 1

@dataclass
class CTCModelConfig:
    checkpoint: str
    model_file: str
    architecture: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    charset: list[str]


@dataclass
class TrOCRConfig:
    checkpoint: str


def show_image(
    image: np.array, cmap: str = "", axis="off", fig_x: int = 24, fix_y: int = 13
) -> None:
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)

    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)


def get_filename(file_path: str) -> str:
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def read_image(image_path: str) -> npt.NDArray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def read_label(gt_file: str) -> str:
    f = open(gt_file, "r", encoding="utf-8")
    label = f.readline()
    label = label.replace("_____", "")
    return label


def read_distribution(file_path: str):
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            content = dict(json.loads(content))

            if "train" in content and "validation" in content and "test" in content:
                train_samples = content["train"]
                valid_samples = content["validation"]
                test_samples = content["test"]
                return train_samples, valid_samples, test_samples
            else:
                print(
                    f"Data distribution is missing the required keys 'train' and 'validation' and 'test'."
                )
                return None, None, None

    else:
        print(f"Specified distribution file does not exist: {file_path}")
        return None, None, None


def read_ctc_model_config(config_file: str) -> CTCModelConfig:
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    checkpoint = f"{model_dir}/{json_content['checkpoint']}"
    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    architecture = json_content["architecture"]
    input_width = json_content["input_width"]
    input_height = json_content["input_height"]
    input_layer = json_content["input_layer"]
    output_layer = json_content["output_layer"]
    squeeze_channel_dim = (
        True if json_content["squeeze_channel_dim"] == "yes" else False
    )
    swap_hw = True if json_content["swap_hw"] == "yes" else False
    characters = json_content["charset"]

    config = CTCModelConfig(
        checkpoint,
        onnx_model_file,
        architecture,
        input_width,
        input_height,
        input_layer,
        output_layer,
        squeeze_channel_dim,
        swap_hw,
        characters,
    )

    return config


def build_vocabulary(charset: str) -> list[str]:
    vocab = [x for x in charset]
    vocab.insert(0, " ")

    return vocab


def resize_to_height(image, target_height: int) -> tuple[npt.NDArray, int]:
    ratio = target_height / image.shape[0]
    image = cv2.resize(image, (int(image.shape[1] * ratio), target_height))
    return image, ratio


def resize_to_width(image, target_width: int) -> tuple[npt.NDArray, int]:
    ratio = target_width / image.shape[1]
    image = cv2.resize(image, (target_width, int(image.shape[0] * ratio)))
    return image, ratio


def binarize(
    image: npt.NDArray, adaptive: bool = True, block_size: int = 51, c: int = 13
) -> npt.NDArray:
    line_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if adaptive:
        bw = cv2.adaptiveThreshold(
            line_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

    else:
        _, bw = cv2.threshold(line_img, 120, 255, cv2.THRESH_BINARY)

    # bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return bw


def pad_to_width(
    img: npt.NDArray, target_width: int, target_height: int, padding: str
) -> npt.NDArray:
    _, width = img.shape
    tmp_img, ratio = resize_to_width(img, target_width)

    height = tmp_img.shape[0]
    middle = (target_height - tmp_img.shape[0]) // 2

    if padding == "white":
        upper_stack = np.ones(shape=(middle, target_width), dtype=np.uint8)
        lower_stack = np.ones(
            shape=(target_height - height - middle, target_width),
            dtype=np.uint8,
        )

        upper_stack *= 255
        lower_stack *= 255
    else:
        upper_stack = np.zeros(shape=(middle, target_width), dtype=np.uint8)
        lower_stack = np.zeros(
            shape=(target_height - height - middle, target_width),
            dtype=np.uint8,
        )

    out_img = np.vstack([upper_stack, tmp_img, lower_stack])

    return out_img


def pad_to_height(
    img: npt.NDArray, target_width: int, target_height: int, padding: str
) -> npt.NDArray:
    height, _ = img.shape
    tmp_img, ratio = resize_to_height(img, target_height)

    width = tmp_img.shape[1]
    middle = (target_width - width) // 2

    if padding == "white":
        left_stack = np.ones(shape=(target_height, middle), dtype=np.uint8)
        right_stack = np.ones(
            shape=(target_height, target_width - width - middle),
            dtype=np.uint8,
        )

        left_stack *= 255
        right_stack *= 255

    else:
        left_stack = np.zeros(shape=(target_height, middle), dtype=np.uint8)
        right_stack = np.zeros(
            shape=(target_height, target_width - width - middle),
            dtype=np.uint8,
        )

    out_img = np.hstack([left_stack, tmp_img, right_stack])

    return out_img


def resize_n_pad(
    image: npt.NDArray, target_width: int, target_height: int, padding: str
) -> npt.NDArray:
    """
    Preliminary implementation of resizing and padding images.
    Args:
        - padding: "white" for padding the image with 255, otherwise the image will be padded with 0

    - TODO: using np.pad for an eventually more elegant/faster implementation
    """
    width_ratio = target_width / image.shape[1]
    height_ratio = target_height / image.shape[0]

    if width_ratio < height_ratio:  # maybe handle equality separately
        tmp_img, _ = resize_to_width(image, target_width)

        if padding == "white":
            v_stack = np.ones(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            v_stack = np.zeros(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        v_stack *= 255

        out_img = np.vstack([v_stack, tmp_img])

    elif width_ratio > height_ratio:
        tmp_img, _ = resize_to_height(image, target_height)

        if padding == "white":
            h_stack = np.ones(
                (tmp_img.shape[0], target_width - tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            h_stack = np.zeros(
                (tmp_img.shape[0], target_width - tmp_img.shape[1]), dtype=np.uint8
            )
        h_stack *= 255

        out_img = np.hstack([tmp_img, h_stack])
    else:
        tmp_img, _ = resize_to_width(image, target_width)

        if padding == "white":
            v_stack = np.ones(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            v_stack = np.zeros(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        v_stack *= 255

        out_img = np.vstack([v_stack, tmp_img])
        # print(f"Info -> equal ratio: {img.shape}, w_ratio: {width_ratio}, h_ratio: {height_ratio}")

    return cv2.resize(out_img, (target_width, target_height))


def preprare_ocr_line(
    img: npt.NDArray,
    input_width: int,
    input_height: int,
    padding: str = "black",
) -> npt.NDArray:

    width_ratio = input_width / img.shape[1]
    height_ratio = input_height / img.shape[0]

    if width_ratio < height_ratio:
        out_img = pad_to_width(img, input_width, input_height, padding)

    elif width_ratio > height_ratio:
        out_img = pad_to_height(img, input_width, input_height, padding)
    else:
        out_img = pad_to_width(img, input_width, input_height, padding)

    out_img = cv2.resize(
        out_img, (input_width, input_height), interpolation=cv2.INTER_LINEAR
    )
    out_img = out_img.astype(np.float32)
    out_img = (out_img / 127.5) - 1.0

    return out_img


def post_pad_image(image: np.array, pad_width: int = 100) -> np.array:
    pad_patch = np.ones((image.shape[0], pad_width), dtype=np.uint8)
    pad_patch *= 255
    padded_img = np.hstack([pad_patch, image, pad_patch])

    return padded_img
