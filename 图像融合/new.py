import os

path = 'E:\zhanglefu\save\labels\\watch\\'
for file in os.listdir(path):
    nfile = file.replace('.jpg','')
    print(nfile)
    os.rename(path+file,path+nfile)