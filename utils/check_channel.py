from PIL import Image


def check_image_channels(image_path):
    """
    检查指定图片的通道数。

    参数:
    image_path (str): 图片的文件路径。

    返回:
    int: 图片的通道数。
    """
    with Image.open(image_path) as img:
        # 获取图片的模式，例如'RGB', 'RGBA', 'L'等
        mode = img.mode
        # 根据模式获取通道数
        if mode == 'RGB':
            channels = 3
        elif mode == 'RGBA':
            channels = 4
        elif mode == 'L':  # 灰度图
            channels = 1
        else:
            channels = len(img.getbands())
        print(f"图片 '{image_path}' 的通道数为: {channels}")
        return channels


# 使用示例
image_path = r'D:\Edeg\github\Blind2Unblind-main\data\train\sgy\label7.png'  # 替换为你的图片路径
check_image_channels(image_path)