import sys
sys.path.append('/data/codes/ty/evaluation/clipscore')
#from .clipscore import main  # 根据需要导入函数或类


from clipscore import main as clipscore_main

if __name__ == "__main__":
    sys.argv = ["clipscore.py", "/data/codes/ty/evaluation/clipscore/example/good_captions.json", "/data/codes/ty/evaluation/clipscore/example/images/"]
    clipscore_main()