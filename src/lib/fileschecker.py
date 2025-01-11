import shutil
import os
import sys
import fnmatch

def file_save(file_path: str, folder_name: str) -> str:
    destination_folder = os.path.join(os.getcwd(), folder_name)
    base_file_name = os.path.splitext(os.path.basename(file_path))[0]
    destination_path = os.path.join(destination_folder, os.path.basename(file_path))

    os.makedirs(destination_folder, exist_ok=True)
    
    if not os.path.isabs(file_path):
        files_moved = []
        for file in os.listdir(os.getcwd()):
            pattern = base_file_name + '*'
            if fnmatch.fnmatch(file, pattern):
                src_path = os.path.join(os.getcwd(), file)
                dst_path = os.path.join(destination_folder, file)
                shutil.move(src_path, dst_path)
                files_moved.append(file)
                print(f"File {file} successfully moved to {destination_folder}")

        if not files_moved:
            print(f'No files with base name {base_file_name} found to move.')
            sys.exit(1)
        
    elif os.path.exists(file_path) and os.path.isabs(file_path):
        shutil.copy(file_path, destination_path)
        print(f"File successfully copied to {destination_path}")
    else:
        print(f'File not found: {file_path}')
        sys.exit(1)

    return destination_path
    


