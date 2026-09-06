import os
import pandas as pd
import torch


def load_pseudo_labels_from_csv(csv_path, dataroot):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"伪标签CSV文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    csv_paths = df['image_path'].tolist()
    pseudo_labels = torch.tensor(df['pseudo_label'].values, dtype=torch.long)
    ground_truths = torch.tensor(df['ground_truth'].values, dtype=torch.long)
    
    
    corrected_paths = []
    valid_count = 0
    
    dataroot_parts = dataroot.replace('\\', '/').split('/')
    dataset_name = dataroot_parts[-1] if dataroot_parts else None
    
    for csv_path_str in csv_paths:
        if 'dataset' in csv_path_str.lower():
            parts = csv_path_str.replace('\\', '/').split('/')
            try:
                dataset_idx = [i for i, p in enumerate(parts) if p.lower() == 'dataset'][0]
                relative_parts = parts[dataset_idx + 1:]  # ['Office-Home', 'Clipart', 'Alarm_Clock', '00001.jpg']
                
                if dataset_name and len(relative_parts) > 0 and relative_parts[0] == dataset_name:
                    relative_parts = relative_parts[1:]  # ['Clipart', 'Alarm_Clock', '00001.jpg']
                
                relative_path = os.path.join(*relative_parts) if relative_parts else ''
                
                corrected_path = os.path.join(dataroot, relative_path) if relative_path else dataroot
                corrected_paths.append(corrected_path)
                
                if os.path.exists(corrected_path):
                    valid_count += 1
            except (IndexError, ValueError):
                corrected_paths.append(csv_path_str)
        else:
            corrected_paths.append(csv_path_str)
    
    print(f"  📊 从CSV加载了 {len(corrected_paths)} 个样本")
    print(f"  ✅ 路径验证: {valid_count}/{len(corrected_paths)} 个样本路径有效")
    
    unique_labels = torch.unique(pseudo_labels)
    print(f"  📈 类别分布: {len(unique_labels)} 个类别")
    
    return {
        'data': corrected_paths,
        'label': pseudo_labels,
        'gt': ground_truths
    }


def organize_samples_by_class(samples_dict, classnames):
    paths_by_class = {}
    
    for classname in classnames:
        paths_by_class[classname] = []
    
    for path, label in zip(samples_dict['data'], samples_dict['label']):
        label_idx = label.item() if hasattr(label, 'item') else label
        classname = classnames[label_idx]
        paths_by_class[classname].append(path)
    
    return paths_by_class


def get_valid_classes(samples_dict, classnames, min_samples=3):
    unique_labels = torch.unique(samples_dict['label'])
    valid_classes = [classnames[label.item()] for label in unique_labels]
    
    print(f"  ✅ 有效类别: {len(valid_classes)}/{len(classnames)} (CSV中存在的类别)")
    
    return valid_classes


def load_unlabeled_source_pseudo_labels(unlabeled_source, labeled_source, results_dir, dataroot, classnames, target_name):
    task_dir = f"{labeled_source}_to_{target_name}"
    
    task_path = os.path.join(results_dir, task_dir)
    
    csv_path = None
    if os.path.exists(task_path):
        for filename in os.listdir(task_path):
            if filename.endswith(f"_{labeled_source}_to_{unlabeled_source}"):
                csv_file = os.path.join(task_path, filename, 'pseudo_labels_filtered.csv')
                if os.path.exists(csv_file):
                    csv_path = csv_file
                    break
    
    if not csv_path:
        raise FileNotFoundError(
            f"未找到无标签源域 {unlabeled_source} 的CSV文件\n"
            f"任务目录: {task_path}\n"
            f"期望的子目录格式: step*_{labeled_source}_to_{unlabeled_source}"
        )
    
    print(f"\n{'='*70}")
    print(f"📂 加载无标签源域 {unlabeled_source} 的伪标签")
    print(f"   任务: {labeled_source} -> {target_name}")
    print(f"   CSV路径: {csv_path}")
    print(f"{'='*70}")
    
    samples_dict = load_pseudo_labels_from_csv(csv_path, dataroot)
    
    paths_by_class = organize_samples_by_class(samples_dict, classnames)
    
    valid_classes = get_valid_classes(samples_dict, classnames, min_samples=3)
    
    return samples_dict, paths_by_class, valid_classes
