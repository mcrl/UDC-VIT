import os
import datetime

def find_nearest_folder(min_val, max_val):
    # Folder path
    base_dir = 'UDC-VIT-Work/alignment/yuv'

    # Get folder list
    folders = [int(folder) for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
    
    # Exception handling for non-existing folders
    if not folders:
        print("Folders do not exist.")
        return None, None

    # Sort folder list and find the nearest folders to min_val and max_val
    folders.sort()
    nearest_min = min(folders, key=lambda x: abs(x - min_val))
    nearest_max = min(folders, key=lambda x: abs(x - max_val))

    # Change min_val and max_val to the nearest folder values and print messages
    print(f"min_val is modified from {min_val} to {nearest_min}")
    print(f"max_val is modified from {max_val} to {nearest_max}")
    min_val = nearest_min
    max_val = nearest_max

    return min_val, max_val


def check_fps(min_dir, max_dir):
    if not os.path.exists('logs/check_fps'):
        os.makedirs('logs/check_fps')

    # Folder path
    base_dir = 'UDC-VIT-Work/alignment/yuv'

    # Dictionary for storing results
    weird_fps = {}

    # Check folders from min_val to max_val
    for folder_num in range(min_dir, max_dir + 1):
        folder_path = os.path.join(base_dir, str(folder_num))
        print("Checking the directory named", str(folder_num) + ".")

        # Dictionary to record weird fps values for each camera
        camera_fps = {}

        # Check log files in the folder
        for log_file in ['log_0.txt', 'log_1.txt']:
            log_path = os.path.join(folder_path, log_file)

            # Check if file exists
            if os.path.exists(log_path):
                # Open and read the file
                with open(log_path, 'r') as f:
                    lines = f.readlines()

                    # Ignore the first 30 lines (when focus is not yet adjusted)
                    for line in lines[30:]:
                        # Get fps value
                        fps_value = float(line.split()[1].split('(')[1])

                        # Record fps values that are not 60 for each camera
                        if round(fps_value, 0) != 60.0:
                            camera_fps[log_file] = fps_value

        # Add folder with weird fps values to the dictionary
        if camera_fps:
            weird_fps[folder_num] = camera_fps

    # Write results to a log file
    log_file_path = f'./logs/check_fps/log_check_fps_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.txt'
    with open(log_file_path, 'w') as f:
        f.write(f'checked_dir_min: {min_dir}\n')
        f.write(f'checked_dir_max: {max_dir}\n')

        # Record folders with weird fps values if any
        if weird_fps:
            for folder_num, cameras in weird_fps.items():
                f.write(f'{folder_num}: ')
                for camera, fps_value in cameras.items():
                    f.write(f'{camera}({fps_value:.2f}), ')
                f.write('\n')
        else:
            f.write('No weird fps values found.\n')