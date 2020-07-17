import os, time, json

def test_and_make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def currentTime():
    return time.strftime("%H_%M_%S", time.localtime())

def test_postfix_dir(root):
    seplen = len(os.sep)
    if root[-seplen:] != os.sep:
        return root + os.sep
    return root

def save_opt(root, opt):
    json_dump(opt._get_kwargs(), root + "config.json")

def json_dump(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f)