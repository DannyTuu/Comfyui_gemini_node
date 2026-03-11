from .google_api import (
    GoogleGeminiImageAnalysis,
    GoogleGeminiImageGeneration,
    GoogleGeminiImageEditing,
    GoogleGeminiConfigManager,
)


NODE_CLASS_MAPPINGS = {
    "GoogleGeminiImageAnalysis": GoogleGeminiImageAnalysis,
    "GoogleGeminiImageGeneration": GoogleGeminiImageGeneration,
    "GoogleGeminiImageEditing": GoogleGeminiImageEditing,
    "GoogleGeminiConfigManager": GoogleGeminiConfigManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleGeminiImageAnalysis": "Google Gemini - Image Analysis",
    "GoogleGeminiImageGeneration": "Google Gemini - Image Generation",
    "GoogleGeminiImageEditing": "Google Gemini - Image Editing",
    "GoogleGeminiConfigManager": "Google Gemini - Config Manager",
}

