import PIL.Image
import os
import io
from ..utils.imports import vertexai, genai_new
from google.genai.types import (
    GenerateContentConfig,
    HarmCategory,
    HarmBlockThreshold,
    HttpOptions,
    SafetySetting,
)
safetysettings = [SafetySetting(**{"category": "HARM_CATEGORY_HARASSMENT",
                  "threshold": "BLOCK_NONE",
                  }),
SafetySetting(**{"category": "HARM_CATEGORY_HATE_SPEECH",
                 "threshold": "BLOCK_NONE",
                 }),
SafetySetting(**{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                 "threshold": "BLOCK_NONE",
                 }),
SafetySetting(**{"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                 "threshold": "BLOCK_NONE",
                 }),       
] 

def make_llm_call_gemini(image_path: str, system_message: str, model: str = "gemini-2.0-flash", response_schema: dict = None, max_tokens: int = 4000, temperature: float = 0.5) -> str:
    # With the newer Google GenAI SDK, we need to create a client
    client = genai_new.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Create generation config with the correct GenerateContentConfig type
    config = genai_new.types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json"
    )

    # Add response schema if provided
    if response_schema is not None:
        config.response_schema = response_schema

    try:
        # Open and compress the image
        image = PIL.Image.open(image_path)
        compressed_image_bytes, _ = compress_image(image) # Quality is returned but not used here

        # Close the original image object now that compression is done
        if image:
            image.close()
            # The 'image' variable still exists and will be handled by the finally block,
            # PIL's close() is typically safe to call multiple times.

        # Create content parts using bytes
        image_part = genai_new.types.Part.from_bytes(data=compressed_image_bytes, mime_type='image/jpeg')
        content_parts = [image_part, system_message]

        # For Gemini 2.5 models, disable thinking
        if model.startswith("gemini-2.5"):
            # Create a new config with thinking disabled by setting thinking_config
            gemini25_config = genai_new.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                thinking_config=genai_new.types.ThinkingConfig(thinking_budget=0)
            )

            # Add response schema if provided
            if response_schema is not None:
                gemini25_config.response_schema = response_schema

            # Generate content with thinking disabled
            response = client.models.generate_content(
                model=model,
                contents=content_parts,
                config=gemini25_config
            )
        else:
            # Standard call for other Gemini models
            response = client.models.generate_content(
                model=model,
                contents=content_parts,
                config=config
            )

        return response.text
    finally:
        # Ensure image is closed even if an error occurs
        if 'image' in locals() and image: # Check if image was defined and not None
            try:
                image.close() # Attempt to close; safe if already closed
            except Exception:
                pass # Ignore errors if it fails (e.g., trying to close a None object or already closed and problematic)

def make_llm_call_vertex(image_path: str, system_message: str, model: str, project_id: str, location: str, response_schema: dict = None, max_tokens: int = 4000, temperature: float = 0.5) -> str:
    """
    This function calls the Vertex AI Gemini API (not to be confused with the Gemini API) with an image and a system message and returns the response text.
    """
   # With the newer Google GenAI SDK, we need to create a client
    client = genai_new.Client(vertexai =True,project=project_id, location=location)

    # Create generation config with the correct GenerateContentConfig type
    config = genai_new.types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json"
    )

    # Add response schema if provided
    if response_schema is not None:
        config.response_schema = response_schema

    try:
        # Open and compress the image
        image = PIL.Image.open(image_path)
        compressed_image_bytes, _ = compress_image(image) # Quality is returned but not used here

        # Close the original image object now that compression is done
        if image:
            image.close()
            # The 'image' variable still exists and will be handled by the finally block,
            # PIL's close() is typically safe to call multiple times.

        # Create content parts using bytes
        image_part = genai_new.types.Part.from_bytes(data=compressed_image_bytes, mime_type='image/jpeg')
        content_parts = [image_part, system_message]

        # For Gemini 2.5 models, disable thinking
        if model.startswith("gemini-2.5"):
            thinking_budget = 0 if "flash" in model else 1
            gemini25_config = genai_new.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                thinking_config=genai_new.types.ThinkingConfig(thinking_budget=thinking_budget),
                safety_settings=safetysettings
            )

            # Add response schema if provided
            if response_schema is not None:
                gemini25_config.response_schema = response_schema

            # Generate content with thinking disabled
            response = client.models.generate_content(
                model=model,
                contents=content_parts,
                config=gemini25_config
            )
        else:
            # Standard call for other Gemini models
            response = client.models.generate_content(
                model=model,
                contents=content_parts,
                config=config
            )

        return response.text
    finally:
        # Ensure image is closed even if an error occurs
        if 'image' in locals() and image: # Check if image was defined and not None
            try:
                image.close() # Attempt to close; safe if already closed
            except Exception:
                pass # Ignore errors if it fails (e.g., trying to close a None object or already closed and problematic)


def compress_image(image: PIL.Image.Image, max_size_bytes: int = 1097152, quality: int = 95) -> tuple[bytes, int]:
    """
    Compress image if it exceeds file size while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_size_bytes: Maximum file size in bytes (default ~1MB)
        quality: Initial JPEG quality (0-100)
    
    Returns:
        Tuple of (compressed image bytes, final quality used)
    """
    output = io.BytesIO()
    
    # Initial compression
    image.save(output, format='JPEG', quality=quality)
    
    # Reduce quality if file is too large
    while output.tell() > max_size_bytes and quality > 10:
        output = io.BytesIO()
        quality -= 5
        image.save(output, format='JPEG', quality=quality)
    
    # If reducing quality didn't work, reduce dimensions
    if output.tell() > max_size_bytes:
        while output.tell() > max_size_bytes:
            width, height = image.size
            image = image.resize((int(width*0.9), int(height*0.9)), PIL.Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=quality)
    
    # Return the bytes directly
    output.seek(0)
    return output.getvalue(), quality