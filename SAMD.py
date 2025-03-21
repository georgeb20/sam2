from abc import ABC, abstractmethod
import torch
from PIL import Image, ImageDraw
import numpy as np
import gradio as gr
import cv2
# -----------------------------------------------------------------------------
# 1. Model Loading Functions
# -----------------------------------------------------------------------------
def load_sam_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """
    Load the SAM2 model and return a predictor.
    """
    from sam2.build_sam import build_sam2 
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam_model = build_sam2(config_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam_model)
    return predictor

def load_stable_diffusion_inpaint(model_name: str = "stabilityai/stable-diffusion-2-inpainting", device: str = "cuda"):
    """
    Load the Stable Diffusion inpainting pipeline.
    """
    from diffusers import StableDiffusionInpaintPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_name, 
        torch_dtype=dtype
    )
    pipe = pipe.to(device)
    return pipe

# Global model loading
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "checkpoints/sam2.1_hiera_large.pt"
config = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = load_sam_model(checkpoint, config, device=device)
inpaint_pipe = load_stable_diffusion_inpaint(device=device)

# -----------------------------------------------------------------------------
# 2. Helper Functions to Draw Points and Overlays
# -----------------------------------------------------------------------------
def draw_points_on_image(img, points):
    """
    Draw the selected points on the image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    radius = 5
    for point in points:
        x, y, label = point["x"], point["y"], point["label"]
        color = "green" if label == 1 else "red"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=color, outline="white", width=2)
    return img_copy

def overlay_mask_on_image(img, mask, color=(30, 144, 255, 150)):
    """
    Overlay a binary mask on the image.
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    mask_arr = np.array(mask, dtype=bool)
    overlay_arr = np.array(overlay)
    overlay_arr[mask_arr] = color
    overlay = Image.fromarray(overlay_arr, 'RGBA')
    return Image.alpha_composite(img, overlay)

# -----------------------------------------------------------------------------
# 3. Helper to Set Original Image Only Once
# -----------------------------------------------------------------------------
def set_original(img, current_original):
    """
    Store the uploaded image as original (only once).
    """
    if current_original is None and img is not None:
        return img, img
    else:
        return current_original, gr.update()

# -----------------------------------------------------------------------------
# 4. Helper to Get Binary Mask from Points (using SAM2)
# -----------------------------------------------------------------------------
def get_binary_mask(original_img, points_state):
    """
    Run segmentation using the current points and return a binary mask image.
    """
    if not points_state:
        return Image.new("L", original_img.size, 0)
    
    point_coords = np.array([[pt["x"], pt["y"]] for pt in points_state])
    point_labels = np.array([pt["label"] for pt in points_state])
    predictor.set_image(original_img)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    best_mask = masks[0]
    binary_mask = (best_mask.astype(np.uint8)) * 255
    mask_img = Image.fromarray(binary_mask, mode="L")
    return mask_img

# -----------------------------------------------------------------------------
# 5. Segmentation Function Using Collected Points
# -----------------------------------------------------------------------------
def segment_image_from_points(original_img, points_state):
    """
    Run SAM2 segmentation and return the image with mask overlay.
    """
    if not points_state:
        return original_img
    
    point_coords = np.array([[pt["x"], pt["y"]] for pt in points_state])
    point_labels = np.array([pt["label"] for pt in points_state])
    predictor.set_image(original_img)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    best_mask = masks[0]
    overlayed_img = overlay_mask_on_image(original_img, best_mask)
    return overlayed_img

# -----------------------------------------------------------------------------
# 6. Combined Update Points and Segmentation Callback
# -----------------------------------------------------------------------------
def update_and_segment(points_state, original_img, evt: gr.SelectData, point_mode):
    """
    Update points from the select event, run segmentation, and return both
    a composite overlay image and the binary mask.
    """
    if points_state is None:
        points_state = []
    
    if evt is None:
        comp_img = draw_points_on_image(original_img, points_state)
        bin_mask = get_binary_mask(original_img, points_state)
        return points_state, comp_img, bin_mask

    x = evt.index[0]
    y = evt.index[1]
    label = 1 if point_mode == "Positive" else 0
    threshold = 10
    remove_index = None
    for i, pt in enumerate(points_state):
        if abs(pt["x"] - x) < threshold and abs(pt["y"] - y) < threshold:
            remove_index = i
            break
    if remove_index is not None:
        points_state.pop(remove_index)
    else:
        points_state.append({"x": x, "y": y, "label": label})
    
    segmented_img = segment_image_from_points(original_img, points_state)
    comp_img = draw_points_on_image(segmented_img, points_state)
    bin_mask = get_binary_mask(original_img, points_state)
    return points_state, comp_img, bin_mask

# -----------------------------------------------------------------------------
# 7. Inpainting Interface and Implementations
# -----------------------------------------------------------------------------
class Inpainter(ABC):
    @abstractmethod
    def inpaint(self, original_img: Image.Image, points_state: list, prompt: str) -> Image.Image:
        """
        Perform inpainting on the given image using the provided points and prompt.
        """
        pass

class StableDiffusionInpainter(Inpainter):
    def __init__(self, pipeline, predictor):
        self.pipeline = pipeline
        self.predictor = predictor

    def inpaint(self, original_img: Image.Image, points_state: list, prompt: str) -> Image.Image:
        if not points_state:
            return original_img
        
        # Use the predictor to generate a segmentation mask from points.
        point_coords = np.array([[pt["x"], pt["y"]] for pt in points_state])
        point_labels = np.array([pt["label"] for pt in points_state])
        self.predictor.set_image(original_img)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        best_mask = masks[0]  # Boolean array
        
        # Create binary mask image.
        binary_mask = (best_mask.astype(np.uint8)) * 255
        mask_img = Image.fromarray(binary_mask, mode="L")
        
        # Resize images to the pipeline's expected size.
        original_resized = original_img.resize((512, 512))
        mask_resized = mask_img.resize((512, 512))
        
        # Perform inpainting.
        result = self.pipeline(
            prompt=prompt,
            image=original_resized,
            mask_image=mask_resized,
            guidance_scale=12
        ).images[0]
        return result
def cv2_inpaint_image(original_img: Image.Image, mask_img: Image.Image, radius: int = 3) -> Image.Image:
    """
    Perform inpainting using OpenCV's Telea algorithm.

    Args:
        original_img (PIL.Image): The original image.
        mask_img (PIL.Image): Binary mask where the region to inpaint is white (255) and the rest is black (0).
        radius (int): Inpainting radius.

    Returns:
        PIL.Image: The inpainted image.
    """
    # Convert PIL images to OpenCV (BGR for the original, grayscale for the mask).
    original_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    mask_cv = np.array(mask_img)
    # Ensure mask is binary
    mask_cv = (mask_cv > 0).astype(np.uint8) * 255
    # Use Telea algorithm for inpainting
    inpainted_cv = cv2.inpaint(original_cv, mask_cv, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    # Convert back to PIL image (RGB)
    inpainted_img = Image.fromarray(cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB))
    return inpainted_img

class CV2Inpainter(Inpainter):
    def __init__(self, predictor):
        self.predictor = predictor

    def inpaint(self, original_img: Image.Image, points_state: list, prompt: str = "") -> Image.Image:
        """
        Inpaint the selected region using OpenCV's inpainting method.
        The prompt parameter is unused here but is part of the interface.
        """
        if not points_state:
            return original_img
        
        # Generate a binary mask from points (using your existing segmentation approach)
        point_coords = np.array([[pt["x"], pt["y"]] for pt in points_state])
        point_labels = np.array([pt["label"] for pt in points_state])
        self.predictor.set_image(original_img)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        best_mask = masks[0]  # Boolean array
        binary_mask = (best_mask.astype(np.uint8)) * 255
        mask_img = Image.fromarray(binary_mask, mode="L")
        
        # Optionally, you might want to adjust or extend the mask horizontally here.
        # For example, you could apply a dilation or custom mask manipulation.
        
        # Inpaint using OpenCV
        inpainted_result = cv2_inpaint_image(original_img, mask_img, radius=3)
        return inpainted_result
# -----------------------------------------------------------------------------
# 8. Gradio Interface
# -----------------------------------------------------------------------------
def interactive_demo():
    # Instantiate the inpainter using Stable Diffusion.
    #inpainter = StableDiffusionInpainter(inpaint_pipe, predictor)
    inpainter = CV2Inpainter(predictor)

    with gr.Blocks() as demo:
        gr.Markdown("# Interactive Layer Segmentation and Inpainting")
        gr.Markdown("**Note:** Upload your image once. Then click/select to add points. The segmentation overlay and binary mask update automatically.")
        
        with gr.Row():
            annotated_image = gr.Image(type="pil", label="Annotated Image (Overlay)", interactive=True)
            binary_mask_display = gr.Image(type="pil", label="Binary Mask", interactive=False)
        
        original_image = gr.State(None)
        points_state = gr.State([])
        point_mode = gr.Radio(choices=["Positive", "Negative"], value="Positive", label="Annotation Mode")
        
        annotated_image.change(
            fn=set_original,
            inputs=[annotated_image, original_image],
            outputs=[original_image, annotated_image]
        )
        
        annotated_image.select(
            fn=update_and_segment,
            inputs=[points_state, original_image, point_mode],
            outputs=[points_state, annotated_image, binary_mask_display]
        )
        
        sd_prompt = gr.Textbox(
            label="Inpainting Prompt",
            placeholder="Enter your prompt for inpainting..."
        )
        inpaint_button = gr.Button("Inpaint Selected Region")
        inpainted_image = gr.Image(type="pil", label="Inpainted Image")
        
        # Use the interface's inpaint method.
        inpaint_button.click(
            fn=inpainter.inpaint,
            inputs=[original_image, points_state, sd_prompt],
            outputs=inpainted_image
        )
    
    return demo

if __name__ == "__main__":
    app = interactive_demo()
    app.launch()
