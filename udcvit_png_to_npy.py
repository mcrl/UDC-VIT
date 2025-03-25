import os
import numpy as np
from PIL import Image


def image_to_numpy(input_path, output_path):
    img = Image.open(input_path)
    img_array = np.array(img)
    np.save(output_path, img_array)


def process_dir(input_dir, output_dir):
    count = 0
    total_count = sum([len(files) for r, d, files in os.walk(input_dir) if any(f.endswith('.png') for f in files)])

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, input_dir)
                npy_path = os.path.join(output_dir, relative_path[:-4] + ".npy")

                count += 1
                npy_folder = os.path.dirname(npy_path)
                if not os.path.exists(npy_folder):
                    os.makedirs(npy_folder)
                image_to_numpy(file_path, npy_path)
                print(f"[{count}/{total_count}]", file_path, "===>", npy_path)
    return count


def main():
    base_dirs = ['training', 'validation', 'test']
    sub_dirs = ['Input', 'GT']
    
    total_count = 0
    for base_dir in base_dirs:
        for sub_dir in sub_dirs:
            input_dir = os.path.join("./UDC-VIT", base_dir, sub_dir)
            output_dir = os.path.join("./UDC-VIT_npy", base_dir, sub_dir)
            count = process_dir(input_dir, output_dir)
            total_count += count
            print(f"Total for {base_dir}-{sub_dir}: {count}")
    
    print("Overall Total:", total_count)


if __name__ == "__main__":
    main()
