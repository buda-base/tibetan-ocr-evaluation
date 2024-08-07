import cv2
import torch
import pyewts
import numpy as np
import numpy.typing as npt
from Models import Easter2, VanillaCRNN
from pyctcdecode import build_ctcdecoder
from Utils import (
    CTCModelConfig,
    Encoding,
    binarize,
    build_vocabulary,
    preprare_ocr_line
)
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    RobertaTokenizer,
)


class CRNNInference:
    def __init__(self, model_config: CTCModelConfig) -> None:
        self.config = model_config
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        self.model = VanillaCRNN(
            img_width=self.config.input_width,
            img_height=self.config.input_height,
            charset_size=len(self.config.charset) + 1
        )

        self.checkpoint = torch.load(self.config.checkpoint)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()
        self.model.to(self.device)

        self.converter = pyewts.pyewts()
        self.vocab = build_vocabulary(self.config.charset)

        self.ctc_decoder = build_ctcdecoder(
            self.vocab,
            alpha=0.5,
            beta=1.0,
        )

        print(f"Using Device: {self.device}")

    def predict(
        self,
        image: npt.NDArray,
        binarize_input: bool = True,
    ):
        if binarize_input:
            image = binarize(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if len(image.shape) > 2:
            # make sure the input shape to prepare_ocr_line is HxW
            image = image[:, :]
        image = preprare_ocr_line(
            image, self.config.input_width, self.config.input_height
        )
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        t_img = torch.FloatTensor(image).to(self.device)
        logits = self.model(t_img)
        logits = logits.cpu().detach().numpy()
        logits = np.squeeze(logits, axis=1)

        predicted_text = self.ctc_decoder.decode(logits)
        predicted_text = predicted_text.replace(" ", "")
        predicted_text = predicted_text.replace("§", " ")

        return predicted_text
    

class Easter2Inference:
    def __init__(self, model_config: CTCModelConfig) -> None:
        self.config = model_config
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        self.model = Easter2(
            self.config.input_width,
            self.config.input_height,
            vocab_size=len(self.config.charset) + 1,
            mean_pooling=False,
        )
        self.checkpoint = torch.load(self.config.checkpoint)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()
        self.model.to(self.device)

        self.converter = pyewts.pyewts()
        self.vocab = build_vocabulary(self.config.charset)

        self.ctc_decoder = build_ctcdecoder(
            self.vocab,
            alpha=0.5,
            beta=1.0,
        )

        print(f"Using Device: {self.device}")

    def predict(
        self,
        image: npt.NDArray,
        binarize_input: bool = True,
    ):
        if binarize_input:
            image = binarize(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if len(image.shape) > 2:
            # make sure the input shape to prepare_ocr_line is HxW
            image = image[:, :]
        image = preprare_ocr_line(
            image, self.config.input_width, self.config.input_height
        )
        image = np.expand_dims(image, axis=0)
        t_img = torch.FloatTensor(image).to(self.device)
        logits = self.model(t_img)
        logits = logits.cpu().detach().numpy()
        logits = np.squeeze(logits, axis=0)
        logits = np.transpose(logits, axes=[1, 0])
        predicted_text = self.ctc_decoder.decode(logits)
        predicted_text = predicted_text.replace(" ", "")
        predicted_text = predicted_text.replace("§", " ")

        return predicted_text

    def score(
        self, prediction: str, gt_label: str, scorer, output_encoding: Encoding.Wylie
    ):
        gt_label = self.converter.toWylie(gt_label)
        cer_score = scorer.compute(predictions=[prediction], references=[gt_label])

        return cer_score


class TrOCRInference:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        self.encoder = "google/vit-base-patch16-224-in21k"
        self.decoder = "spsither/tibetan_RoBERTa_S_e3"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.decoder)
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.encoder)
        self.processor = TrOCRProcessor(
            feature_extractor=self.feature_extractor, tokenizer=self.tokenizer
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(checkpoint).to(
            self.device
        )

        print(f"Using Device: {self.device}")

    def predict(self, image: npt.NDArray):
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values.to(self.device))
        predicted_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return predicted_text

    def score(
        self, prediction: str, gt_label: str, scorer, output_encoding: Encoding.Wylie
    ):
        cer_score = scorer.compute(predictions=[prediction], references=[gt_label])
        return cer_score
