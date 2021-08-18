def main():
    f = '/data/proli/data/sse-label/bc40_label/valid/1ka8A.label'
    f = open(f)
    a = f.readlines()
    a = list(map(lambda x:int(x.strip()), a))
    str=''
    for i in a:
        if i == 0:
            str+='H'
        elif i == 1:
            str+='E'
        else:
            str+='C'
    print(str)

if __name__ == '__main__':
    main()


