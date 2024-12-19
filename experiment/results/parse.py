import os
counter = 0
for directory in os.listdir():
    if os.path.isdir(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if file.endswith(".html"):
                with open(file_path, 'r', encoding="utf-8") as f:
                    content = f.read()
                if content == 'None':
                    counter += 1
                    try:
                        os.remove(file_path)
                        print(f"Deleted file: {os.path.relpath(file_path)}")
                    except Exception as e:
                        print(f"Error deleting file: {os.path.relpath(file_path)}")
                        print(e)


print(f"Deleted {counter} files")