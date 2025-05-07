import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import os
import shutil

def classify_images(model_path, src_dir, target_dir, class_names):
    

    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 修改最后一层为4分类
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 数据预处理（需与训练时一致）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建目标文件夹（如果不存在）
    for cls in class_names:
        os.makedirs(os.path.join(target_dir, cls), exist_ok=True)

    # 处理图片
    for filename in os.listdir(src_dir):
        # 过滤图片文件
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        src_path = os.path.join(src_dir, filename)
        
        try:
            # 加载并预处理图片
            img = Image.open(src_path).convert('RGB')
            inputs = transform(img).unsqueeze(0).to(device)
            
            # 推理
            with torch.no_grad():
                outputs = model(inputs)
            
            # 获取预测结果
            _, pred = torch.max(outputs, 1)
            class_idx = pred.item()
            
            # 目标路径
            dest_dir = os.path.join(target_dir, str(class_idx))
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest_path = os.path.join(dest_dir, filename)
            
            # 复制文件
            shutil.copy(src_path, dest_path)
            print(f'Copied {filename} to {dest_dir}')
            
        except Exception as e:
            print(f'Error processing {filename}: {str(e)}')

    print("Classification completed!")

if __name__ == "__main__":
    # 配置参数
    model_path = './best_model.pth'  # 需修改为你的模型路径
    src_dir = 'your photo\'s path to classify'                        # 待分类图片目录
    target_dir = 'your target path'                  # 目标分类目录
    class_names = ['0', '1', '2', '3']     # 类别文件夹名称
    # 调用分类函数
    classify_images(model_path, src_dir, target_dir, class_names)