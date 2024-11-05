import os
from PIL import Image
import torch
from torchvision import transforms, models

# 设置基础目录路径，包含 Klarf 文件和图像文件
BASE_DIR = "/home/amhs/toolid/AYSEV01/to_ai"
OUTPUT_DIR = "/home/amhs/toolid/img_classify/defectlink/AYSEV01"
MODEL_PATH = '/home/amhs/recog/code/Result/2024-09-11_22-54-38/checkpoint_best.pth'

# 如果输出目录不存在，则创建该目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化模型
def initialize_model(model_path):
    # 加载预训练的 ResNet50 模型，并设置最后的全连接层输出为 73 类
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=73)
    # 加载保存的模型权重文件
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 将模型设置为评估模式
    return model

# 图像预处理函数
def preprocess_image(image_path):
    # 定义图像预处理步骤：调整大小、转为 RGB、转为张量、并进行标准化
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    # 打开图像并应用预处理
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0)  # 返回带有 batch 维度的张量

# 读取 Klarf 文件内容
def read_klarf_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.readlines()  # 返回文件内容的行列表
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在，请检查路径是否正确！")
        return None

# 更新 DefectRecordSpec 字段
def update_defect_record_spec(defect_record_spec):
    updated_spec = []
    for field in defect_record_spec:
        updated_spec.append(field)
        # 在 TEST 字段后插入 AUTOONSEMCLASS 字段
        if field == "TEST":
            updated_spec.append("AUTOONSEMCLASS")
            break
    # 删除 AUTOONSEMCLASS 后的所有字段，并设置第2个字段值为 '12'
    if "AUTOONSEMCLASS" in updated_spec:
        auto_onsem_index = updated_spec.index("AUTOONSEMCLASS")
        updated_spec = updated_spec[:auto_onsem_index + 1]
        updated_spec[0] = '12' #确实设置为0能达到要求
    return updated_spec

# 处理 Klarf 文件的主要函数
def process_klarf_file(klarf_file, model):
    klarf_path = os.path.join(BASE_DIR, klarf_file)
    klarf_content = read_klarf_file(klarf_path)
    if klarf_content is None:
        return

    # 初始化变量以存储 Klarf 文件的不同部分内容
    new_klarf_content = []
    defect_record_spec = None
    defect_data_lines = []
    tiff_file_dict = {}
    defect_list_started = False
    current_tiff_file_name = None
    summary_spec_content = []
    summary_list_content = []
    eof_content = []
    is_summary_section = False
    is_summary_list_section = False

    # 逐行解析 Klarf 文件内容
    for line in klarf_content:
        stripped_line = line.strip()
        if 'DefectRecordSpec' in line:
            # 更新 DefectRecordSpec 字段，并加入新内容
            defect_record_spec = stripped_line.replace(';', '').split()[1:]
            defect_record_spec = update_defect_record_spec(defect_record_spec)
            new_klarf_content.append('DefectRecordSpec ' + ' '.join(defect_record_spec) + ';\n')
        elif 'SummarySpec' in line:
            # 识别 SummarySpec 部分的开始
            is_summary_section = True
            summary_spec_content.append(line)
        elif is_summary_section and len(summary_spec_content) == 1:
            # SummarySpec 部分第二行内容
            summary_spec_content.append(line)
            is_summary_section = False
            is_summary_list_section = True
        elif is_summary_list_section:
            if 'EndOfFile' in line:
                eof_content.append(line)
                is_summary_list_section = False
            else:
                summary_list_content.append(line)
        elif 'TiffFileName' in line:
            # 记录 TIFF 文件名
            current_tiff_file_name = line.split()[-1].strip(';')
        elif 'DefectList' in line:
            # 检测到 DefectList 部分
            defect_list_started = True
            new_klarf_content.append(line.replace(';', ''))
        elif defect_list_started and stripped_line:
            # 解析缺陷数据行
            defect_data = line.replace(';', '').strip().split()[:12]  # 仅保留前12个值
            defect_id = defect_data[0]
            if current_tiff_file_name is not None:
                tiff_file_dict[defect_id] = current_tiff_file_name
            defect_data_lines.append((line, defect_data))
        else:
            new_klarf_content.append(line)

    # 检查必要的字段
    if not defect_record_spec or not defect_list_started:
        print(f"文件 {klarf_file} 缺少 DefectRecordSpec 或 DefectList，跳过处理。")
        return

    # 使用模型预测每个缺陷数据的分类
    for line, defect_data in defect_data_lines:
        if len(defect_data) < 10:
            continue

        defect_id = defect_data[0]
        tiff_file_name = tiff_file_dict.get(defect_id, "")
        if not tiff_file_name:
            continue

        image_path = os.path.join(BASE_DIR, tiff_file_name)
        if not os.path.isfile(image_path):
            continue

        try:
            # 预处理图像并使用模型进行预测
            image = preprocess_image(image_path)
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                auto_onsem_class_value = str(predicted.item())
                print(f"缺陷ID {defect_id} 的AUTOONSEMCLASS值为 {auto_onsem_class_value}")

                # 将预测结果填入缺陷数据的第12个字段
                if len(defect_data) >= 12:
                    defect_data[11] = auto_onsem_class_value
                else:
                    defect_data += [''] * (12 - len(defect_data))
                    defect_data[11] = auto_onsem_class_value
        except Exception as e:
            print(f"处理图像 {tiff_file_name} 时出错: {e}")
            continue

        # 将缺陷数据加入新内容，确保每行末尾有分号
        new_klarf_content.append(' '.join(defect_data) + ';')

    # 删除 TIFF 文件名信息
    new_klarf_content = [line for line in new_klarf_content if 'TiffFileName' not in line]
    defect_list_start = next((i + 1 for i, line in enumerate(new_klarf_content) if 'DefectList' in line), None)
    if defect_list_start is not None:
        # 将缺陷数据插入到 DefectList 部分中
        new_klarf_content = new_klarf_content[:defect_list_start] + [line for _, line in defect_data_lines]
    else:
        print(f"未找到 DefectList，跳过文件 {klarf_file}")
        return

    # 合并各部分内容
    new_klarf_content.extend(summary_spec_content)
    new_klarf_content.extend(summary_list_content)
    new_klarf_content.extend(eof_content)

    # 保存更新后的 Klarf 文件
    save_updated_klarf(klarf_file, new_klarf_content)

# 保存 Klarf 文件并确保 SummarySpec 前一行末尾有分号
def save_updated_klarf(klarf_file, new_content):
    # 查找 SummarySpec 行的索引
    summary_index = next((i for i, line in enumerate(new_content) if 'SummarySpec' in line), None)
    if summary_index is not None and summary_index > 0:
        # 确保 SummarySpec 前一行以分号结尾
        previous_line = new_content[summary_index - 1]
        if isinstance(previous_line, list):
            previous_line = ' '.join(previous_line) 
        previous_line = previous_line.strip()
        
        if not previous_line.endswith(';'):
            new_content[summary_index - 1] = previous_line + ';\n'

    # 保存文件
    updated_klarf_path = os.path.join(OUTPUT_DIR, klarf_file)
    try:
        with open(updated_klarf_path, 'w', encoding='utf-8') as file:
            file.writelines([line if isinstance(line, str) else ' '.join(line) + '\n' for line in new_content])
        print(f"已将更新后的 Klarf 文件保存到: {updated_klarf_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

# 主程序入口
def main():
    model = initialize_model(MODEL_PATH)
    # 查找 BASE_DIR 目录中的所有 .001 结尾的 Klarf 文件
    klarf_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.001')]
    for klarf_file in klarf_files:
        process_klarf_file(klarf_file, model)

if __name__ == "__main__":
    main()
