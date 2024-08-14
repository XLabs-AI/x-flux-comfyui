import os
try:
    import git  
    git.Git(".").clone("https://github.com/XLabs-AI/x-flux")
except:
    os.system("git clone https://github.com/XLabs-AI/x-flux" )
#os.rename("x-flux", "xflux")
os.system("mv x-flux xflux")
os.system("pip install -r requirements.txt")
os.system("pip install -r xflux/requirements.txt")
print("Succesfully installed")
