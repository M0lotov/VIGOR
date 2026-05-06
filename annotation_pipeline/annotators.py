from __future__ import annotations

from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageFilter


ImageInput = Union[str, Path, Image.Image]


def _load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        converted = image.convert("RGB")
        if hasattr(image, "filename"):
            converted.filename = image.filename
        return converted
    converted = Image.open(image).convert("RGB")
    converted.filename = str(image)
    return converted


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {heatmap.shape}.")
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value <= min_value:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap - min_value) / (max_value - min_value)


def _threshold_heatmap(heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    normalized = _normalize_heatmap(heatmap)
    return (normalized >= threshold).astype(np.float32)

def _resize_heatmap_to_image(heatmap: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    normalized = _normalize_heatmap(heatmap)
    heatmap_image = Image.fromarray(normalized, mode="F")
    resized = heatmap_image.resize(image_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)

def _overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    image_alpha: float = 1,
    contour_width: int = 3,
) -> Image.Image:
    image_rgba = image.convert("RGBA")
    white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    faded_image = Image.blend(white_background, image_rgba, image_alpha)

    mask_array = np.asarray(mask, dtype=np.uint8)
    padded = np.pad(mask_array.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )
    contour = center & ~interior
    contour_uint8 = 255 * contour.astype(np.uint8)
    contour_image = Image.fromarray(contour_uint8)
    if contour_width > 1:
        contour_image = contour_image.filter(ImageFilter.MaxFilter(contour_width))

    contour_color = (0, 255, 255)
    contour_overlay = Image.new("RGBA", image.size, contour_color + (0,))
    contour_overlay.putalpha(contour_image.point(lambda value: 255 if value > 0 else 0))
    return Image.alpha_composite(faded_image, contour_overlay)

def factor_pair_closest_to_aspect(num_tokens: int, aspect_ratio: float) -> tuple[int, int]:
    if num_tokens <= 0:
        raise ValueError("Image token count must be positive.")

    best_h = 1
    best_w = num_tokens
    best_error = float("inf")
    for h in range(1, int(math.sqrt(num_tokens)) + 1):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        for candidate_h, candidate_w in ((h, w), (w, h)):
            error = abs((candidate_w / candidate_h) - aspect_ratio)
            if error < best_error:
                best_h = candidate_h
                best_w = candidate_w
                best_error = error
    return best_h, best_w


def image_aspect_ratio(image, inputs: dict) -> float:
    pixel_values = inputs.get("pixel_values")
    if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim >= 4:
        height = int(pixel_values.shape[-2])
        width = int(pixel_values.shape[-1])
        if height > 0 and width > 0:
            return width / height
    width, height = image.size
    return width / height

class Annotator(ABC):
    def __init__(self, overlay_alpha: float = 1, mask_threshold: float = 0.5) -> None:
        self.overlay_alpha = overlay_alpha
        self.mask_threshold = mask_threshold

    @abstractmethod
    def get_mask(self, image: ImageInput, concept: str) -> np.ndarray:
        raise NotImplementedError

    def generate(self, image: ImageInput, concept: str) -> Tuple[np.ndarray, Image.Image]:
        pil_image = _load_image(image)
        mask = self.get_mask(pil_image, concept)
        overlay = _overlay_mask(
            image=pil_image,
            mask=mask,
            image_alpha=self.overlay_alpha,
        )
        return mask, overlay



class Chefer(Annotator):
    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        overlay_alpha: float = 1,
        mask_threshold: float = 0.5,
    ) -> None:
        super().__init__(overlay_alpha=overlay_alpha, mask_threshold=mask_threshold)
        import CLIP.clip as clip

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clip = clip
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device, jit=False)

    def get_mask(self, image: ImageInput, concept: str) -> np.ndarray:
        pil_image = _load_image(image)
        image_tensor = self.preprocess(pil_image.resize((224, 224))).unsqueeze(0).to(self.device)
        text_tensor = self.clip.tokenize([concept]).to(self.device)
        _, image_relevance = self._interpret(image=image_tensor, texts=text_tensor)
        num_patches = int(np.sqrt(image_relevance.shape[-1]))
        heatmap = image_relevance[0].detach().cpu().numpy().reshape(num_patches, num_patches)
        resized_heatmap = _resize_heatmap_to_image(heatmap, pil_image.size)
        return _threshold_heatmap(resized_heatmap, threshold=self.mask_threshold)

    def _interpret(
        self,
        image: torch.Tensor,
        texts: torch.Tensor,
        start_layer: int = -1,
        start_layer_text: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, _ = self.model(images, texts)
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[np.arange(logits_per_image.shape[0]), np.arange(batch_size)] = 1
        one_hot = torch.from_numpy(one_hot).to(self.device)
        one_hot = torch.sum(one_hot * logits_per_image)
        self.model.zero_grad()

        image_attn_blocks = list(dict(self.model.visual.transformer.resblocks.named_children()).values())
        if start_layer == -1:
            start_layer = len(image_attn_blocks) - 1

        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        relevance = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype, device=self.device)
        relevance = relevance.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for layer_idx, block in enumerate(image_attn_blocks):
            if layer_idx < start_layer:
                continue
            grad = torch.autograd.grad(one_hot, [block.attn_probs], retain_graph=True)[0].detach()
            cam = block.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = (grad * cam).reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            relevance = relevance + torch.bmm(cam, relevance)
        image_relevance = relevance[:, 0, 1:]

        text_attn_blocks = list(dict(self.model.transformer.resblocks.named_children()).values())
        if start_layer_text == -1:
            start_layer_text = len(text_attn_blocks) - 1

        num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
        text_relevance = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype, device=self.device)
        text_relevance = text_relevance.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for layer_idx, block in enumerate(text_attn_blocks):
            if layer_idx < start_layer_text:
                continue
            grad = torch.autograd.grad(one_hot, [block.attn_probs], retain_graph=True)[0].detach()
            cam = block.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = (grad * cam).reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            text_relevance = text_relevance + torch.bmm(cam, text_relevance)

        return text_relevance, image_relevance


class Sam3(Annotator):
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        overlay_alpha: float = 1,
        mask_threshold: float = 0.5,
        instance_threshold: float = 0.5,
    ) -> None:
        super().__init__(overlay_alpha=overlay_alpha, mask_threshold=mask_threshold)
        from transformers import Sam3Model, Sam3Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.instance_threshold = instance_threshold
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def get_mask(self, image: ImageInput, concept: str) -> np.ndarray:
        pil_image = _load_image(image)
        inputs = self.processor(images=pil_image, text=concept, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.instance_threshold,
            mask_threshold=self.instance_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results.get("masks")
        if masks is None or len(masks) == 0:
            return np.zeros((pil_image.height, pil_image.width), dtype=np.float32)

        combined_mask = masks.to(torch.float32).amax(dim=0)
        return combined_mask.cpu().numpy()


class GroundedSAM(Annotator):
    def __init__(
        self,
        grounding_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        overlay_alpha: float = 1,
        mask_threshold: float = 0.5,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        instance_threshold: float = 0.5,
    ) -> None:
        super().__init__(overlay_alpha=overlay_alpha, mask_threshold=mask_threshold)
        from transformers import (
            GroundingDinoForObjectDetection,
            GroundingDinoProcessor,
            Sam3Model,
            Sam3Processor,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.instance_threshold = instance_threshold

        self.grounding_processor = GroundingDinoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = GroundingDinoForObjectDetection.from_pretrained(
            grounding_model_id
        ).to(self.device)
        self.grounding_model.eval()

        self.sam_processor = Sam3Processor.from_pretrained(sam_model_id)
        self.sam_model = Sam3Model.from_pretrained(sam_model_id).to(self.device)
        self.sam_model.eval()

    def get_mask(self, image: ImageInput, concept: str) -> np.ndarray:
        pil_image = _load_image(image)
        prompt = self._normalize_text_prompt(concept)

        grounding_inputs = self.grounding_processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            grounding_outputs = self.grounding_model(**grounding_inputs)

        detections = self.grounding_processor.post_process_grounded_object_detection(
            grounding_outputs,
            input_ids=grounding_inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(pil_image.height, pil_image.width)],
        )[0]

        boxes = detections.get("boxes")
        if boxes is None or len(boxes) == 0:
            return np.zeros((pil_image.height, pil_image.width), dtype=np.float32)

        sam_inputs = self.sam_processor(
            images=pil_image,
            input_boxes=[boxes.detach().cpu().tolist()],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            sam_outputs = self.sam_model(**sam_inputs)

        segmentation = self.sam_processor.post_process_instance_segmentation(
            sam_outputs,
            threshold=self.instance_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=sam_inputs.get("original_sizes").tolist(),
        )[0]

        masks = segmentation.get("masks")
        if masks is None or len(masks) == 0:
            return np.zeros((pil_image.height, pil_image.width), dtype=np.float32)

        combined_mask = masks.to(torch.float32).amax(dim=0)
        return (combined_mask >= self.mask_threshold).cpu().numpy().astype(np.float32)

    @staticmethod
    def _normalize_text_prompt(concept: str) -> str:
        prompt = concept.strip().lower()
        if not prompt:
            raise ValueError("Concept prompt must be non-empty.")
        if not prompt.endswith("."):
            prompt = f"{prompt}."
        return prompt


class Attention(Annotator):
    def __init__(
        self,
        device: Optional[str] = "cuda",
        prompt_template: str = "Describe the image.",
        overlay_alpha: float = 1,
        mask_threshold: float = 0.5,
    ) -> None:
        super().__init__(overlay_alpha=overlay_alpha, mask_threshold=mask_threshold)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_template = prompt_template
        self._auto_processor_cls = AutoProcessor
        self._auto_model_cls = AutoModelForImageTextToText
        self.backends = [self._load_backend(config) for config in self._backend_configs()]

    def get_mask(self, image: ImageInput, concept: str) -> np.ndarray:
        pil_image = _load_image(image)
        concept_text = concept.strip()
        if not concept_text:
            raise ValueError("Concept prompt must be non-empty.")

        resized_heatmaps = []
        for backend in self.backends:
            backend_image = self._prepare_image_for_backend(pil_image, backend)
            text_prompt = self._build_text_prompt(backend["processor"])
            prompt_inputs = backend["processor"](
                text=[text_prompt],
                images=[backend_image],
                padding=True,
                return_tensors="pt",
            ).to(backend['model'].device)
            forward_inputs = backend["processor"](
                text=[text_prompt + f"A {concept_text}"],
                images=[backend_image],
                padding=True,
                return_tensors="pt",
            ).to(backend['model'].device)

            prompt_len = int(prompt_inputs["input_ids"].shape[1])
            generated_ids = forward_inputs["input_ids"][:, prompt_len:]
            if generated_ids.shape[1] == 0:
                raise ValueError(
                    f"The attention annotator backend {backend['name']} produced no concept tokens."
                )

            prompt_image_token_mask = self._get_prompt_image_token_mask(
                backend["model"],
                prompt_inputs,
            )
            grid_h, grid_w = self._get_grid_shape(
                image=pil_image,
                inputs=prompt_inputs,
                image_token_count=prompt_image_token_mask.sum().item(),
            )

            with torch.inference_mode():
                outputs = backend["model"](
                    **forward_inputs,
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )

            layer_image_attentions = []
            for layer_attention in outputs.attentions:
                layer_attention = layer_attention[0, :, prompt_len:, :prompt_len]
                layer_image_attentions.append(layer_attention[:, :, prompt_image_token_mask])

            generated_image_attentions = torch.stack(layer_image_attentions).permute(2, 0, 1, 3)
            generated_image_attentions = generated_image_attentions[1:]
            if generated_image_attentions.shape[0] == 0:
                raise ValueError(
                    f"The attention annotator backend {backend['name']} has no usable concept tokens."
                )

            if backend["drop_topk_image_tokens"] > 0:
                topk_count = min(
                    backend["drop_topk_image_tokens"],
                    generated_image_attentions.shape[-1],
                )
                if topk_count > 0:
                    _, topk_indices = generated_image_attentions.mean((0, 1, 2)).topk(topk_count)
                    generated_image_attentions[:, :, :, topk_indices] = 0

            heatmap = generated_image_attentions.mean(dim=(0, 1, 2)).reshape(grid_h, grid_w).float()
            resized_heatmap = _resize_heatmap_to_image(
                heatmap.detach().cpu().numpy(),
                pil_image.size,
            )
            resized_heatmaps.append(resized_heatmap)

        ensemble_heatmap = np.mean(np.stack(resized_heatmaps, axis=0), axis=0)
        return _threshold_heatmap(ensemble_heatmap, threshold=self.mask_threshold)

    def _build_text_prompt(self, processor) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt_template},
                ],
            },
        ]
        return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _get_prompt_image_token_mask(self, model, prompt_inputs) -> torch.Tensor:
        mm_token_type_ids = prompt_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            return (mm_token_type_ids[0] == 1).to(model.device)

        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("Unable to identify image tokens for attention extraction.")
        return (prompt_inputs["input_ids"][0] == image_token_id).to(model.device)

    def _get_grid_shape(
        self,
        inputs,
        image,
        image_token_count: int,
    ) -> tuple[int, int]:
        square_size = int(math.sqrt(image_token_count))
        if square_size * square_size == image_token_count:
            return square_size, square_size
        return factor_pair_closest_to_aspect(image_token_count, image_aspect_ratio(image, inputs))

    def _prepare_image_for_backend(self, image: Image.Image, backend: dict) -> Image.Image:
        resize_to = backend.get("resize_to")
        if resize_to is None:
            return image
        return image.resize(resize_to, Image.Resampling.BILINEAR)

    def _backend_configs(self) -> list[dict]:
        return [
            {
                "name": "llava_1.5_7b",
                "model_id": "llava-hf/llava-1.5-7b-hf",
                "processor_kwargs": {},
                "model_kwargs": {},
                "resize_to": None,
                "drop_topk_image_tokens": 3,
                "device": "cuda:5"
            },
            {
                "name": "llava_ov_8b",
                "model_id": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
                "processor_kwargs": {"trust_remote_code": True},
                "model_kwargs": {"trust_remote_code": True},
                "resize_to": None,
                "drop_topk_image_tokens": 3,
                "device": "cuda:6"
            },
            {
                "name": "glm_4.6v_flash",
                "model_id": "zai-org/GLM-4.6V-Flash",
                "processor_kwargs": {"trust_remote_code": True},
                "model_kwargs": {"trust_remote_code": True},
                "resize_to": None,
                "drop_topk_image_tokens": 3,
                "device": "cuda:7"
            },
        ]

    def _load_backend(self, config: dict) -> dict:
        processor = self._auto_processor_cls.from_pretrained(
            config["model_id"],
            **config["processor_kwargs"],
        )
        model = self._auto_model_cls.from_pretrained(
            config["model_id"],
            dtype=torch.bfloat16,
            attn_implementation="eager",
            **config["model_kwargs"],
        ).to(config['device'])
        model.eval()
        return {
            **config,
            "processor": processor,
            "model": model,
        }


class ClipSeg(Annotator):
    def __init__(
        self,
        model_id: str = "CIDAS/clipseg-rd64-refined",
        device: Optional[str] = None,
        overlay_alpha: float = 1,
        mask_threshold: float = 0.5,
    ) -> None:
        super().__init__(overlay_alpha=overlay_alpha, mask_threshold=mask_threshold)
        from transformers import AutoProcessor, CLIPSegForImageSegmentation

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def get_mask(self, image: ImageInput, concept: str) -> np.ndarray:
        pil_image = _load_image(image)
        inputs = self.processor(
            text=[concept],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        probabilities = torch.sigmoid(logits).to(torch.float32)
        probabilities = torch.nn.functional.interpolate(probabilities[None, None, :, :], size=(pil_image.size[1], pil_image.size[0]), mode="bilinear", align_corners=False).squeeze()
        return (probabilities >= self.mask_threshold).cpu().numpy().astype(np.float32)
