import os
import csv

def write_aligned_list(input_dir, output_dir):
    
    file_list_cam_0 = []
    file_list_cam_1 = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            file_number = int(filename.split("_f")[1].split(".")[0])
            if file_number >= 0: # 30
                if filename.startswith("cam_0"):
                    file_list_cam_0.append(filename)
                elif filename.startswith("cam_1"):
                    file_list_cam_1.append(filename)
    
    file_list_cam_0.sort()
    file_list_cam_1.sort()
    
    with open(output_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'species', 'individual_id', 'box'])
   
        individual_id = 0
        for filename_cam_0, filename_cam_1 in zip(file_list_cam_0, file_list_cam_1):

            species = "udc"
            individual_id += 1
            box = "0 1 2 3"

            writer.writerow([filename_cam_0, species, individual_id, box])
            writer.writerow([filename_cam_1, species, individual_id, box])
    
    print("Done writing file list for calculating PCK:", output_dir)

