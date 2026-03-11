"""
Google Gemini API ComfyUI 节点测试示例
"""

import json
import torch
from google_api import (
    GoogleGeminiClient,
    GoogleGeminiImageAnalysis,
    GoogleGeminiImageGeneration,
    GoogleGeminiImageEditing,
    GoogleGeminiConfigManager,
    get_config,
    save_config,
)


TEST_BASE_URL = "https://yunwu.ai/v1beta"

def create_test_image_tensor(width: int = 512, height: int = 512) -> torch.Tensor:
    """创建用于视觉分析和图像编辑的测试图像"""
    x_gradient = torch.linspace(0, 1, width, dtype=torch.float32).view(1, 1, width, 1)
    x_gradient = x_gradient.expand(1, height, width, 1)

    y_gradient = torch.linspace(0, 1, height, dtype=torch.float32).view(1, height, 1, 1)
    y_gradient = y_gradient.expand(1, height, width, 1)

    blue_channel = torch.full((1, height, width, 1), 0.35, dtype=torch.float32)
    return torch.cat([x_gradient, y_gradient, blue_channel], dim=-1)


def _is_api_key_configured() -> bool:
    """检查是否已经配置API密钥"""
    api_key = get_config().get('api_key', '').strip()
    if not api_key:
        print("   错误: 未检测到API密钥，请先在 gemini_config.json 中配置 api_key")
        return False
    return True


def test_config_manager():
    """测试配置管理"""
    print("\n=== 测试配置管理 ===")

    manager = GoogleGeminiConfigManager()

    # 测试设置API密钥
    print("\n1. 设置API密钥...")
    status = manager.manage("test-api-key-12345", action="set", base_url=TEST_BASE_URL)
    print(f"   结果: {status[0]}")

    # 测试获取API密钥
    print("\n2. 获取API密钥...")
    status = manager.manage("", action="get")
    print(f"   结果: {status[0]}")

    # 测试清除API密钥
    print("\n3. 清除API密钥...")
    status = manager.manage("", action="clear")
    print(f"   结果: {status[0]}")

def test_vision_analysis_nodes():
    """测试图像分析节点（单图与多图）"""
    print("=== 测试图像分析节点 ===")

    # if not _is_api_key_configured():
        # return

    image1 = create_test_image_tensor()
    image2 = torch.flip(image1, dims=[2])

    print("\n2. 图像对比分析...")
    multi_vision = GoogleGeminiImageAnalysis()
    multi_result = multi_vision.analyze(
        prompt="对比两张图像在颜色和构图上的差异。",
        model="gemini-3.1-pro-preview",
        temperature=1.0,
        top_p=0.95,
        max_tokens=512,
        api_key="",
        base_url=TEST_BASE_URL,
        system_instruction="请给出简短清晰的对比结论。",
        image1=image1,
        image2=image2,
    )
    print(f"   对比结果: {multi_result[0][:150]}...")


def test_image_generation_node():
    """测试图像生成节点"""
    print("\n=== 测试图像生成节点 ===")

    if not _is_api_key_configured():
        return

    generator = GoogleGeminiImageGeneration()

    generated_image, description, response_info = generator.generate(
        prompt="A futuristic city skyline at sunset, highly detailed, cinematic lighting",
        model="gemini-2.5-flash-image",
        temperature=0.9,
        top_p=0.95,
        max_tokens=65536,
        api_key="",
        base_url=TEST_BASE_URL,
        system_instruction="生成风格统一、细节清晰的图像。",
        aspect_ratio="16:9",
        image_size="1K",
        seed=42,
    )

    if generated_image is not None:
        print(f"   图像生成成功，Tensor shape: {tuple(generated_image.shape)}")
    else:
        print("   图像生成失败")

    print(f"   文本描述: {description[:150]}...")
    print(f"   响应信息: {response_info[:200]}...")


def test_image_editing_node():
    """测试图像编辑节点"""
    print("\n=== 测试图像编辑节点 ===")

    if not _is_api_key_configured():
        return

    editor = GoogleGeminiImageEditing()
    image1 = create_test_image_tensor()
    image2 = torch.flip(image1, dims=[2])

    edited_image, result_text, response_info = editor.edit(
        image1=image1,
        image2=image2,
        edit_prompt="把图像1与图2融合，增强对比度并保留原始几何构图。",
        model="gemini-2.5-flash-image",
        temperature=0.8,
        top_p=0.95,
        max_tokens=2048,
        api_key="",
        system_instruction="输出一张高质量编辑图像并简要说明修改内容。",
        aspect_ratio="1:1",
        base_url=TEST_BASE_URL,
        image_size="1K",
        seed=99,
    )

    if edited_image is not None:
        print(f"   图像编辑成功，Tensor shape: {tuple(edited_image.shape)}")
    else:
        print("   图像编辑失败")

    print(f"   编辑结果: {result_text[:150]}...")
    print(f"   响应信息: {response_info[:200]}...")

def test_client():
    """测试Google Gemini客户端"""
    print("\n=== 测试Google Gemini客户端 ===")

    # 创建客户端
    client = GoogleGeminiClient(api_key="", base_url=TEST_BASE_URL)

    print("\n1. 测试URL构建...")
    url = client._build_url("gemini-3-pro-preview", "generateContent")
    print(f"   URL: {url}")

    print("\n2. 测试请求头...")
    headers = client._get_headers()
    print(f"   请求头: {headers}")

    print("\n3. 测试API调用...")
    try:
        response = client.generate_content(
            model="gemini-3-pro-preview",
            contents=[{
                "role": "user",
                "parts": [{"text": "Hello"}]
            }],
            generation_config={
                "temperature": 0.7,
                "maxOutputTokens": 256
            }
        )
        print(f"   响应状态: 成功")
        print(f"   响应内容: {json.dumps(response, indent=2)[:200]}...")
    except Exception as e:
        print(f"   响应状态: 失败")
        print(f"   错误信息: {str(e)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Google Gemini API ComfyUI 节点测试")
    print("=" * 60)

    # 注意：以下测试需要有效的API密钥
    # 请在运行前替换 "your-api-key-here" 为实际的API密钥

    print("\n提示: 请在运行测试前设置有效的API密钥")
    print("      替换代码中的 'your-api-key-here' 为实际的API密钥")

    # 取消注释以下行来运行测试
    # test_config_manager()
    # test_vision_analysis_nodes()
    # test_image_generation_node()
    test_image_editing_node()
    

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
