import os
try:
    import git  
    git.Git(".").clone("https://github.com/XLabs-AI/x-flux")
except:
    
    os.system("git clone https://github.com/XLabs-AI/x-flux" )
#os.rename("x-flux", "xflux")
cur_dir = os.path.dirname(os.path.abspath(__file__))
run = f"mv x-flux {cur_dir}/xflux"
os.system(run)
os.system(f"pip install -r {cur_dir}/requirements.txt")
os.system(f"pip install -r {cur_dir}/xflux/requirements.txt")
print("Succesfully installed")
