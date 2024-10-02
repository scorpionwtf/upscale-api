from PIL import Image
import io
import torch
import logging
import gc
import os

from RealESRGAN import RealESRGAN

# Set environment variables for better CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def resize_image(image, max_dimension):
    """Resize the image to a maximum dimension (width or height)."""
    width, height = image.size
    scaling_factor = max_dimension / max(width, height)
    if scaling_factor < 1:
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def upscale(imageData, upscale_factor):
    try:
        # Clear CUDA memory and garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

        # Convert bytes data into an image
        original_image = Image.open(io.BytesIO(imageData))

        # Convert the image to RGB if it is not already in RGB mode
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')

        # Resize the image to reduce memory usage if needed
        max_dimension = 1024  # Set this value based on memory constraints
        original_image = resize_image(original_image, max_dimension)

        # Check if CUDA is available and set the device accordingly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the RealESRGAN model and the corresponding weights
        model = RealESRGAN(device, scale=int(upscale_factor))
        model.load_weights(f'weights/RealESRGAN_x{upscale_factor}.pth')

        # Upscale the image with mixed precision
        with torch.cuda.amp.autocast():
            sr_image = model.predict(original_image)

        # Handle potential large output image
        max_pixel_limit = 178956970  # Limit set by PIL to prevent decompression bomb attacks
        if sr_image.size[0] * sr_image.size[1] > max_pixel_limit:
            scaling_factor = (max_pixel_limit / (sr_image.size[0] * sr_image.size[1])) ** 0.5
            new_width = int(sr_image.size[0] * scaling_factor)
            new_height = int(sr_image.size[1] * scaling_factor)
            sr_image = sr_image.resize((new_width, new_height), Image.LANCZOS)

        # Save the upscaled image to a BytesIO object
        image_buffer = io.BytesIO()
        sr_image.save(image_buffer, format='PNG')

        # Reset buffer's current position to the beginning
        image_buffer.seek(0)
        image_object = Image.open(image_buffer)

        # Clear CUDA memory and garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return image_object

    except Exception as e:
        logging.exception("Error during the upscaling process")
        raise ValueError(f"Upscaling failed: {str(e)}")
