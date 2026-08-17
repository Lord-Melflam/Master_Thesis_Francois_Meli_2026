def transpose(image):
    f=0
    result=[]
    im=len(image)-1
    for f in range(im):
        colonne=[]
        for i in range(im):
            colonne.append(f"{str(image[i][f])}")
        result.append(colonne)

    return result
