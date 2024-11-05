from PIL import Image
import os
import tifffile
# 打开 TIFF 文件
#input_file = "/home/amhs/toolid/AMADI01/to_ai/FrontSideADRImg_333574.tiff"
#output_folder = "/home/amhs/toolid/img_classify/defectlink/tiff"
input_file = r"D:\Downloads(Edge)\tiff\A003348-01_T0111A_P1_ASI.000-000-0001.tif"
output_folder = r"D:\Downloads(Edge)\tiff"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 使用 tifffile 读取多页 TIFF 文件
with tifffile.TiffFile(input_file) as tif:
    for page_num, page in enumerate(tif.pages):
        # 将每一页保存为 JPEG 格式
        image = Image.fromarray(page.asarray())
        output_path = os.path.join(output_folder, f"page_{page_num + 1}.JPEG")
        image.convert("RGB").save(output_path, "JPEG")
        print(f"Saved {output_path}")

print(f"TIFF 文件分解完成，共解析出 {page_num + 1} 张图像。")
