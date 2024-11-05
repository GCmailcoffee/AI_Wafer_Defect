import os
from PIL import Image
import torch
from torchvision import transforms, models

# 设置基础目录路径，包含Klarf文件和图像文件
BASE_DIR = "/home/amhs/toolid/AYSEV01/to_ai"
OUTPUT_DIR = "/home/amhs/toolid/img_classify/defectlink/AYSEV01"
MODEL_PATH = '/home/amhs/recog/code/Result/2024-09-11_22-54-38/checkpoint_best.pth'

# 如果输出目录不存在，创建该目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def initialize_model(model_path):
    # 初始化ResNet50模型，并加载预训练权重
    model = models.resnet50(pretrained=True)
    # 修改模型的最后一层，全连接层输出大小为73个类别
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=73)
    # 加载训练好的模型权重
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置模型为评估模式
    return model

def preprocess_image(image_path):
    # 图像预处理，包括调整大小、转换为RGB、转换为张量并归一化
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小到224x224
        transforms.Lambda(lambda img: img.convert('RGB')),  # 确保图像为RGB格式
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0)  # 增加一个维度以匹配批处理输入格式

def read_klarf_file(file_path):
    # 读取Klarf文件内容
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.readlines()
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在，请检查路径是否正确！")
        return None

def update_defect_record_spec(defect_record_spec):
    # 更新缺陷记录规范，移除不需要的字段
    return [field for field in defect_record_spec if field not in ['IMAGECOUNT', 'IMAGELIST']]

def process_klarf_file(klarf_file, model):
    # 处理Klarf文件
    klarf_path = os.path.join(BASE_DIR, klarf_file)
    klarf_content = read_klarf_file(klarf_path)
    if klarf_content is None:
        return

    new_klarf_content = []
    defect_record_spec = None
    defect_data_lines = []
    tiff_file_dict = {}
    defect_list_started = False
    current_tiff_file_name = None  # 初始化为None

    # 遍历Klarf文件的每一行，分析内容
    for line in klarf_content:
        stripped_line = line.strip()
        if 'DefectRecordSpec' in line:
            # 找到DefectRecordSpec，保存字段信息
            defect_record_spec = stripped_line.replace(';', '').split()[1:]
            new_klarf_content.append(line)
        elif 'TiffFileName' in line:
            # 获取当前缺陷的图像文件名
            current_tiff_file_name = line.split()[-1].strip(';')
        elif 'DefectList' in line:
            # 标记缺陷列表的开始
            defect_list_started = True
            new_klarf_content.append(line)
        elif defect_list_started and stripped_line:
            # 处理缺陷列表中的数据行
            defect_data = line.replace(';', '').strip().split()
            defect_id = defect_data[0]
            if current_tiff_file_name is not None:
                tiff_file_dict[defect_id] = current_tiff_file_name  # 关联缺陷ID和图像文件名
            defect_data_lines.append((line, defect_data))
        else:
            new_klarf_content.append(line)

    # 如果缺陷记录规范或缺陷列表不存在，则跳过文件
    if not defect_record_spec or not defect_list_started:
        print(f"文件 {klarf_file} 缺少 DefectRecordSpec 或 DefectList，跳过处理。")
        return

    # 更新缺陷记录规范，去除不需要的字段
    defect_record_spec = update_defect_record_spec(defect_record_spec)
    for line, defect_data in defect_data_lines:
        if len(defect_data) < 10:
            continue

        defect_id = defect_data[0]
        tiff_file_name = tiff_file_dict.get(defect_id, "")
        if not tiff_file_name:
            continue

        # 获取图像路径并进行分类预测
        image_path = os.path.join(BASE_DIR, tiff_file_name)
        if not os.path.isfile(image_path):
            continue

        try:
            image = preprocess_image(image_path)  # 预处理图像
            with torch.no_grad():
                output = model(image)  # 通过模型进行前向传播
                _, predicted = torch.max(output, 1)  # 获取预测的类别
                defect_data[9] = str(predicted.item())  # 更新缺陷数据中的类别字段
                print(f"更新缺陷ID {defect_id} 的类别为 {predicted.item()}")
        except Exception as e:
            print(f"处理图像 {tiff_file_name} 时出错: {e}")
            continue

        new_klarf_content.append(' '.join(defect_data) + ';')

    # 移除原有的TiffFileName行
    new_klarf_content = [line for line in new_klarf_content if 'TiffFileName' not in line]
    defect_list_start = next((i + 1 for i, line in enumerate(new_klarf_content) if 'DefectList' in line), None)
    if defect_list_start is not None:
        # 在缺陷列表开始位置插入新的缺陷数据行
        new_klarf_content = new_klarf_content[:defect_list_start] + [line for _, line in defect_data_lines]
    else:
        print(f"未找到 DefectList，跳过文件 {klarf_file}")
        return

    save_updated_klarf(klarf_file, new_klarf_content)

def save_updated_klarf(klarf_file, new_content):
    # 保存更新后的Klarf文件
    updated_klarf_path = os.path.join(OUTPUT_DIR, klarf_file)
    try:
        with open(updated_klarf_path, 'w', encoding='utf-8') as file:
            # 将每一行写入文件，确保内容为字符串
            file.writelines([line if isinstance(line, str) else ' '.join(line) + '\n' for line in new_content])
        print(f"已将更新后的 Klarf 文件保存到: {updated_klarf_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

def main():
    # 初始化模型并处理每个Klarf文件
    model = initialize_model(MODEL_PATH)
    klarf_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.001')]
    for klarf_file in klarf_files:
        process_klarf_file(klarf_file, model)

if __name__ == "__main__":
    main()