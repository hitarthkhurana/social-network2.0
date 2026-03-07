"""
reconstruct.py — Use NanoBanana 2 (Gemini Flash Image) to reconstruct
visuals for memories that have no stored keyframes.
"""

import os
import io
from PIL import Image
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, NANOBANANA_MODEL, RECONSTRUCTED_DIR

client = genai.Client(api_key=GEMINI_API_KEY)


def reconstruct_image(memory: dict) -> str | None:
    """Generate an image for a memory using NanoBanana 2."""
    os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)

    prompt = f"""Create a photorealistic scene depicting this memory:

{memory['summary']}

Activity: {memory.get('activity', 'unknown')}
People: {', '.join(memory.get('people', [])) or 'none specified'}
Tags: {', '.join(memory.get('tags', []))}

Style: photorealistic, natural lighting, cinematic"""

    try:
        response = client.models.generate_content(
            model=NANOBANANA_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["image", "text"]
            )
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                img = Image.open(io.BytesIO(part.inline_data.data))
                out_path = os.path.join(RECONSTRUCTED_DIR, f"memory_{memory['id']}.jpg")
                img.save(out_path, "JPEG", quality=85)
                return out_path

    except Exception as e:
        print(f"  [NanoBanana reconstruction error] {e}")
        return None


def reconstruct_sequence(memories: list[dict]) -> list[str]:
    """Reconstruct images for a sequence of memories."""
    paths = []
    for memory in memories:
        existing = memory.get("keyframe_paths", [])
        if existing:
            paths.extend(existing)
        else:
            path = reconstruct_image(memory)
            if path:
                paths.append(path)
    return paths
