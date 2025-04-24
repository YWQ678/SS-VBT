from PIL import Image
import os

def convert_images(data_dir, out_dir, file_type=['png', 'tif', 'tiff', 'jpg']):
    # 获取指定目录下所有文件
    for maindir, subdir, file_name_list in os.walk(data_dir):
        for filename in file_name_list:
            if filename.split('.')[-1] in file_type:
                apath = os.path.join(maindir, filename)
                # 打开图片
                image = Image.open(apath)
                # 分离通道
                r, g, b, a = image.split()
                # 合并为RGB
                image = Image.merge("RGB", (r, g, b))
                # 保存转换后的图片
                out_path = os.path.join(out_dir, filename)
                image.save(out_path)

# 设置原始路径和保存路径
data_dir = r'D:\Edeg\github\Blind2Unblind-main\data\train\sgy4'  # 替换为你的图片所在目录
out_dir = r'D:\Edeg\github\Blind2Unblind-main\data\train\sgy'  # 替换为你想要保存转换后图片的目录

# 调用函数
convert_images(data_dir, out_dir)