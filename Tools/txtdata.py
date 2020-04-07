import os
def generate(dir,label):
    files = os.listdir(dir)
    files.sort()
    print '****************'
    print 'input :',dir
    print 'start...'
    listText = open(dir+'\\'+'list.txt','w')
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue        
        name = './images/test/9/' + file + ' ' + str(int(label)) +'\n'
        listText.write(name)
    listText.close()
    print 'down!'
    print '****************'    

if __name__ == '__main__':  
    generate('./9',9)   

