import os
import json
import base64
import requests
from io import BytesIO
from typing import Optional, List, Dict, Any
from PIL import Image
import torch
import numpy as np
from typing import List, Union


def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """
    Convert PIL image(s) to tensor, matching ComfyUI's implementation.
    
    Args:
        image: Single PIL Image or list of PIL Images
        
    Returns:
        torch.Tensor: Image tensor with values normalized to [0, 1]
    """
    if isinstance(image, list):
        if len(image) == 0:
            return torch.empty(0)
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image).astype(np.float32) / 255.0

    return torch.from_numpy(img_array)[None,]


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    """
    Convert tensor to PIL image(s), matching ComfyUI's implementation.
    
    Args:
        image: Tensor with shape [B, H, W, 3] or [H, W, 3], values in range [0, 1]
        
    Returns:
        List[Image.Image]: List of PIL Images
    """
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

    return [Image.fromarray(numpy_image)]


# ==================== 配置管理 ====================
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "gemini_config.json")
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def normalize_base_url(base_url: str) -> str:
    """Normalize base_url for consistent storage and request building."""
    if not isinstance(base_url, str):
        return ""
    return base_url.strip().rstrip("/")


def describe_base_url(base_url: str) -> str:
    """Return a generic label for status output without exposing the raw URL."""
    normalized_base_url = normalize_base_url(base_url)
    default_base_url = normalize_base_url(DEFAULT_BASE_URL)
    if not normalized_base_url or normalized_base_url == default_base_url:
        return "default"
    return "custom"


def get_config() -> Dict[str, Any]:
    """读取配置文件"""
    default_config = {
        "api_key": "",
        "base_url": DEFAULT_BASE_URL,
        "key_base_url_map": {},
    }

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                if isinstance(loaded_config, dict):
                    config = {**default_config, **loaded_config}
                    if not isinstance(config.get("key_base_url_map"), dict):
                        config["key_base_url_map"] = {}
                    config["base_url"] = normalize_base_url(config.get("base_url", "")) or DEFAULT_BASE_URL
                    return config
        except Exception as e:
            print(f"[GoogleAPI] 读取配置失败: {e}")
    return default_config.copy()


def save_config(config: Dict[str, Any]):
    """保存配置文件"""
    try:
        config_to_save = dict(config)
        config_to_save["base_url"] = normalize_base_url(config_to_save.get("base_url", "")) or DEFAULT_BASE_URL

        key_base_url_map = config_to_save.get("key_base_url_map", {})
        if not isinstance(key_base_url_map, dict):
            key_base_url_map = {}

        config_to_save["key_base_url_map"] = {
            str(key).strip(): normalize_base_url(value)
            for key, value in key_base_url_map.items()
            if str(key).strip() and normalize_base_url(value)
        }

        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_to_save, f, indent=2)
    except Exception as e:
        print(f"[GoogleAPI] 保存配置失败: {e}")


# ==================== Google API 客户端 ====================
class GoogleGeminiClient:
    """Google Gemini API 客户端"""

    def __init__(self, api_key: str = "", base_url: str = ""):
        config = get_config()
        self.api_key = api_key.strip() or config.get('api_key', '').strip()
        # self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.base_url = self._resolve_base_url(base_url, config)
        self.timeout = 30000  # 请求超时时间，单位秒

    def _resolve_base_url(self, base_url: str = "", config: Optional[Dict[str, Any]] = None) -> str:
        normalized_runtime_url = normalize_base_url(base_url)
        if normalized_runtime_url:
            return normalized_runtime_url

        config = config or get_config()
        if self.api_key:
            key_base_url_map = config.get("key_base_url_map", {})
            if isinstance(key_base_url_map, dict):
                mapped_url = normalize_base_url(key_base_url_map.get(self.api_key, ""))
                if mapped_url:
                    return mapped_url

        return normalize_base_url(config.get("base_url", "")) or DEFAULT_BASE_URL

    def _save_credentials(self):
        if not self.api_key:
            return

        config = get_config()
        config['api_key'] = self.api_key
        config['base_url'] = self.base_url

        key_base_url_map = config.get("key_base_url_map", {})
        if not isinstance(key_base_url_map, dict):
            key_base_url_map = {}
        key_base_url_map[self.api_key] = self.base_url
        config["key_base_url_map"] = key_base_url_map
        save_config(config)

    def set_api_key(self, api_key: str, base_url: str = ""):
        """设置API密钥"""
        self.api_key = api_key.strip()
        if not self.api_key:
            return

        self.base_url = self._resolve_base_url(base_url)
        self._save_credentials()

    def set_base_url(self, base_url: str):
        """设置base_url"""
        normalized_base_url = normalize_base_url(base_url)
        if not normalized_base_url:
            return

        self.base_url = normalized_base_url

        if self.api_key:
            self._save_credentials()
        else:
            config = get_config()
            config['base_url'] = self.base_url
            save_config(config)

    def apply_runtime_config(self, api_key: str = "", base_url: str = ""):
        """Apply runtime api_key/base_url and persist key-url pair when key is provided."""
        normalized_api_key = api_key.strip()
        normalized_base_url = normalize_base_url(base_url)

        if normalized_api_key:
            self.set_api_key(normalized_api_key, normalized_base_url)
        elif normalized_base_url:
            self.set_base_url(normalized_base_url)

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_url(self, model: str, endpoint: str) -> str:
        """构建API URL"""
        return f"{self.base_url}/models/{model}:{endpoint}?key={self.api_key}"

    def generate_content(self, model: str, contents: List[Dict],
                        generation_config: Optional[Dict] = None,
                        system_instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        调用 generateContent API

        Args:
            model: 模型名称 (e.g., "gemini-2.0-flash")
            contents: 内容列表
            generation_config: 生成配置
            system_instruction: 系统指令

        Returns:
            API响应
        """
        if not self.api_key:
            raise ValueError("API key not set")

        payload = {
            "contents": contents
        }

        if generation_config:
            payload["generationConfig"] = generation_config

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        url = self._build_url(model, "generateContent")

        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")


# ==================== ComfyUI 节点 ====================

class GoogleGeminiImageAnalysis:
    """Google Gemini 图像分析节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gemini-3.1-pro-preview"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 65536}),
                "system_instruction": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis",)
    FUNCTION = "analyze"
    CATEGORY = "Google/Gemini"

    def __init__(self):
        self.client = GoogleGeminiClient()

    def _tensor_to_base64(self, image_tensor: torch.Tensor) -> str:
        """将张量转换为base64"""
        pil_images = tensor2pil(image_tensor)
        pil_image = pil_images[0]

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze(self, prompt: str, model: str, temperature: float, top_p: float,
               max_tokens: int, api_key: str = "", base_url: str = "", system_instruction: str = "",
               image1=None, image2=None, image3=None, image4=None) -> tuple:
        """分析多个图像"""
        try:
            if api_key.strip() or base_url.strip():
                self.client.apply_runtime_config(api_key=api_key, base_url=base_url)

            if not self.client.api_key:
                return ("Error: API key not provided",)

            # 构建parts
            parts = []

            # 添加所有图像
            for img in [image1, image2, image3, image4]:
                if img is not None:
                    image_base64 = self._tensor_to_base64(img)
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_base64
                        }
                    })

            # 添加提示词
            parts.append({"text": prompt})

            # 构建内容
            contents = [{
                "role": "user",
                "parts": parts
            }]

            # 生成配置
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
            }

            # 调用API
            response = self.client.generate_content(
                model=model,
                contents=contents,
                generation_config=generation_config,
                system_instruction=system_instruction if system_instruction else None
            )

            # 提取文本
            if "candidates" in response and response["candidates"]:
                candidate = response["candidates"][0]
                response_parts = candidate.get("content", {}).get("parts") or []
                for part in response_parts:
                    # 跳过thought部分
                    if part.get("thought", False):
                        print("[GoogleAPI] 跳过thought部分")
                        continue

                    if "text" in part:
                        text = part["text"]
                        return (text,)
            else:
                return ("Error: No response from API",)

        except Exception as e:
            return (f"Error: {str(e)}",)


class GoogleGeminiImageGeneration:
    """Google Gemini 图像生成节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gemini-2.5-flash-image", "gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"], {"default": "gemini-2.5-flash-image"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 65536}),
                "system_instruction": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["auto", "1K", "2K", "4K"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "description", "response_info")
    FUNCTION = "generate"
    CATEGORY = "Google/Gemini"

    def __init__(self):
        self.client = GoogleGeminiClient()

    def _base64_to_tensor(self, base64_str: str) -> torch.Tensor:
        """将base64转换为张量"""
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            return pil2tensor(image)
        except Exception as e:
            print(f"[GoogleAPI] Base64转换失败: {e}")
            return None

    def generate(self, prompt: str, model: str, temperature: float, top_p: float,
                max_tokens: int, api_key: str = "", base_url: str = "", system_instruction: str = "",
                aspect_ratio: str = "auto", image_size: str = "auto", seed: int = 0) -> tuple:
        """生成图像"""
        try:
            if api_key.strip() or base_url.strip():
                self.client.apply_runtime_config(api_key=api_key, base_url=base_url)

            if not self.client.api_key:
                return (None, "Error: API key not provided", "")

            # 构建内容 - 请求图像生成
            contents = [{
                "role": "user",
                "parts": [{"text": prompt}]
            }]

            # 生成配置 - 指定响应模式包含图像
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
                "responseModalities": ["TEXT", "IMAGE"],  # 请求图像响应
            }

            # 添加图像配置
            image_config = {}
            if aspect_ratio != "auto":
                image_config["aspectRatio"] = aspect_ratio
            if image_size != "auto" and "gemini-2.5-flash-image" not in model:
                image_config["imageSize"] = image_size

            if image_config:
                generation_config["imageConfig"] = image_config

            # 添加seed
            if seed > 0:
                generation_config["seed"] = seed

            # 构建响应信息
            response_info = {
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
                "seed": seed,
            }

            # 调用API
            response = self.client.generate_content(
                model=model,
                contents=contents,
                generation_config=generation_config,
                system_instruction=system_instruction if system_instruction else None
            )

            # 提取图像和文本
            if "candidates" in response and response["candidates"]:
                parts = response["candidates"][0].get("content", {}).get("parts") or []

                image_tensor = None
                description = ""

                for part in parts:
                    # 提取文本描述
                    if "text" in part:
                        description = part["text"]

                    # 提取图像数据
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        if inline_data.get("mimeType", "").startswith("image/"):
                            base64_data = inline_data.get("data", "")
                            image_tensor = self._base64_to_tensor(base64_data)

                # 添加生成信息到响应
                response_info["status"] = "success"
                response_info["image_generated"] = image_tensor is not None
                response_info_str = json.dumps(response_info, ensure_ascii=False, indent=2)

                if image_tensor is not None:
                    return (image_tensor, description, response_info_str)
                else:
                    response_info["status"] = "error"
                    response_info["error"] = "No image generated"
                    response_info_str = json.dumps(response_info, ensure_ascii=False, indent=2)
                    return (None, f"Error: No image generated. Response: {description}", response_info_str)
            else:
                response_info["status"] = "error"
                response_info["error"] = "No response from API"
                response_info_str = json.dumps(response_info, ensure_ascii=False, indent=2)
                return (None, "Error: No response from API", response_info_str)

        except Exception as e:
            response_info = {
                "status": "error",
                "error": str(e),
                "model": model,
            }
            response_info_str = json.dumps(response_info, ensure_ascii=False, indent=2)
            return (None, f"Error: {str(e)}", response_info_str)


class GoogleGeminiImageEditing:
    """Google Gemini 图像编辑节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edit_prompt": ("STRING", {"multiline": True}),
                "model": (["gemini-2.5-flash-image", "gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"], {"default": "gemini-2.5-flash-image"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 65536}),
                "system_instruction": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["auto", "1K", "2K", "4K"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("edited_image", "result", "response_info")
    FUNCTION = "edit"
    CATEGORY = "Google/Gemini"

    def __init__(self):
        self.client = GoogleGeminiClient()

    def _tensor_to_base64(self, image_tensor: torch.Tensor) -> str:
        """将张量转换为base64"""
        pil_images = tensor2pil(image_tensor)
        pil_image = pil_images[0]

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _base64_to_tensor(self, base64_str: str) -> torch.Tensor:
        """将base64转换为张量"""
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data)).convert('RGB')

            #save
            buffered = BytesIO()
            image.save(buffered, format="PNG")

            return pil2tensor(image)
        except Exception as e:
            print(f"[GoogleAPI] Base64转换失败: {e}")
            return None

    def edit(self, edit_prompt: str, model: str, temperature: float, top_p: float,
            max_tokens: int, api_key: str = "", base_url: str = "", system_instruction: str = "",
            aspect_ratio: str = "auto", image_size: str = "auto", seed: int = 0,
            image1=None, image2=None, image3=None, image4=None) -> tuple:
        """编辑多张图像"""
        try:
            if api_key.strip() or base_url.strip():
                self.client.apply_runtime_config(api_key=api_key, base_url=base_url)

            if not self.client.api_key:
                return (None, "Error: API key not provided", "")

            # 收集输入的图像
            input_images = [image1, image2, image3, image4]
            image_count = sum(1 for img in input_images if img is not None)

            if image_count == 0:
                return (None, "Error: No input images provided", "")

            # 构建parts - 包含所有输入图像和编辑指令
            parts = []

            # 添加所有图像
            for i, img in enumerate(input_images):
                if img is not None:
                    image_base64 = self._tensor_to_base64(img)
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_base64
                        }
                    })

            # 添加编辑指令
            parts.append({"text": edit_prompt})

            # 构建内容
            contents = [{
                "role": "user",
                "parts": parts
            }]

            # 生成配置 - 指定响应模式包含图像
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
                "responseModalities": ["TEXT", "IMAGE"],  # 请求图像响应
            }

            # 添加图像配置
            image_config = {}
            if aspect_ratio != "auto":
                image_config["aspectRatio"] = aspect_ratio
            if image_size != "auto" and "gemini-2.5-flash-image" not in model:
                image_config["imageSize"] = image_size

            if image_config:
                generation_config["imageConfig"] = image_config

            # 添加seed
            if seed > 0:
                generation_config["seed"] = seed

            # 构建响应信息
            response_info = {
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
                "seed": seed,
                "input_image_count": image_count,
                "edit_prompt": edit_prompt[:100] + "..." if len(edit_prompt) > 100 else edit_prompt,
            }

            # 调用API
            response = self.client.generate_content(
                model=model,
                contents=contents,
                generation_config=generation_config,
                system_instruction=system_instruction if system_instruction else None
            )

            # 提取编辑后的图像和结果
            if "candidates" in response and response["candidates"]:
                parts = response["candidates"][0].get("content", {}).get("parts") or []

                edited_image = None
                result_text = ""

                for part in parts:
                    # 提取文本结果
                    if "text" in part:
                        result_text = part["text"]

                    # 提取编辑后的图像
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        print(f"[GoogleAPI] 检测到inlineData部分，mimeType={inline_data.get('mimeType', '')}")
                        if inline_data.get("mimeType", "").startswith("image/"):
                            base64_data = inline_data.get("data", "")
                            edited_image = self._base64_to_tensor(base64_data)
                            print(f"[GoogleAPI] 成功提取编辑后的图像")
                            
                            #save_tensor_as_image(edited_image, "edited_image.png")  # 调试用：保存编辑后的图像


                # 添加编辑信息到响应
                response_info["status"] = "success"
                response_info["image_edited"] = edited_image is not None
                response_info_str = json.dumps(response_info, ensure_ascii=False, indent=2)

                if edited_image is not None:
                    return (edited_image, result_text, response_info_str)
                else:
                    response_info["status"] = "error"
                    response_info["error"] = "No edited image generated"
                    response_info_str = json.dumps(response_info, ensure_ascii=False, indent=2)
                    return (None, f"Error: No edited image generated. Response: {result_text}", response_info_str)
            else:
                response_info["status"] = "error"
                response_info["error"] = "No response from API"
                response_info_str = json.dumps(response_info, ensure_ascii=False, indent=2)
                return (None, "Error: No response from API", response_info_str)

        except Exception as e:
            response_info = {
                "status": "error",
                "error": str(e),
                "model": model,
            }
            response_info_str = json.dumps(response_info, ensure_ascii=False, indent=2)
            return (None, f"Error: {str(e)}", response_info_str)


class GoogleGeminiConfigManager:
    """Google Gemini 配置管理节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "action": (["set", "get", "clear"], {"default": "set"}),
                "base_url": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "manage"
    CATEGORY = "Google/Gemini"

    def manage(self, api_key: str, action: str = "set", base_url: str = "") -> tuple:
        """管理配置"""
        try:
            if action == "set":
                api_key_value = api_key.strip()
                if not api_key_value:
                    return ("Error: API key cannot be empty",)

                client = GoogleGeminiClient()
                client.apply_runtime_config(api_key=api_key_value, base_url=base_url)
                return (f"API key saved successfully (base_url: {describe_base_url(client.base_url)})",)

            elif action == "get":
                config = get_config()
                api_key_value = config.get('api_key', '').strip()
                if not api_key_value:
                    current_base_url = normalize_base_url(config.get("base_url", "")) or DEFAULT_BASE_URL
                    return (f"No API key configured; base_url: {describe_base_url(current_base_url)}",)
                if api_key_value:
                    masked_key = api_key_value[:10] + "..." + api_key_value[-4:]
                    key_base_url_map = config.get("key_base_url_map", {})
                    mapped_base_url = ""
                    if isinstance(key_base_url_map, dict):
                        mapped_base_url = normalize_base_url(key_base_url_map.get(api_key_value, ""))
                    current_base_url = mapped_base_url or normalize_base_url(config.get("base_url", "")) or DEFAULT_BASE_URL
                    return (f"API key found: {masked_key}; base_url: {describe_base_url(current_base_url)}",)

            elif action == "clear":
                config = get_config()
                current_key = config.get('api_key', '').strip()
                key_base_url_map = config.get("key_base_url_map", {})
                if isinstance(key_base_url_map, dict) and current_key in key_base_url_map:
                    key_base_url_map.pop(current_key, None)
                    config["key_base_url_map"] = key_base_url_map
                config['api_key'] = ""
                config['base_url'] = DEFAULT_BASE_URL
                save_config(config)
                return ("API key cleared; base_url reset to default",)

            else:
                return ("Error: Unknown action",)

        except Exception as e:
            return (f"Error: {str(e)}",)


# ==================== 节点注册 ====================
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
