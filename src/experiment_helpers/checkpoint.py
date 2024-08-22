import os

def find_latest_checkpoint(directory,pattern):
    files = os.listdir(directory)
    max_e = -1
    latest_checkpoint = None

    for file in files:
        match = pattern.match(file)
        if match:
            e = int(match.group(1))
            if e > max_e:
                max_e = e
                latest_checkpoint = file

    return latest_checkpoint,max_e