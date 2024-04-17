import os

dataset = 'mm-Fi'

if dataset == 'mRI':
    script_content = ""

    script_file = 'script.sh'
    python_file = "visualize_test.py"

    with open(script_file, 'w') as file:
        data_dir = "/home/naver/Documents/Kien/mRI/Dataset/dataset_release/features/radar"
        anno_dir = "/home/naver/Documents/Kien/mRI/Dataset/dataset_release/aligned_data/pose_labels"

        list_data = sorted(os.listdir(data_dir), key=lambda x: int(x.split("_")[0][7:]))
        list_anno = sorted(os.listdir(anno_dir), key=lambda x: int(x.split("_")[0][7:]))

        for anno_file, data_file in zip(list_anno, list_data):
            anno_path = os.path.join(anno_dir, anno_file)
            data_path = os.path.join(data_dir, data_file)
            script_content += f"python {python_file} {anno_path} {data_path} \n"
        file.write(script_content)
elif dataset == 'mm-Fi':
    script_content = ""

    script_file = 'script_mmFi.sh'
    python_file = "visualize_test.py"

    with open(script_file, 'w') as file:
        for i in range(215):
            script_content += f"python {python_file} --idx {i} \n"
        file.write(script_content)
