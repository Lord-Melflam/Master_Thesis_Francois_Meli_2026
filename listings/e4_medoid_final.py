def longest_word(nom_fichier):
    try :
        mot = ""
        ligne = 0
        n_ieme = 0
        with open(nom_fichier,"r") as f :
            ligne_prov = -1
            for line in f:
                ligne_prov += 1
                chaine=line.strip()
                if not chaine:
                    continue
                numero = -1
                l = chaine.split(" ")
                for word in l :
                    if word.strip().isalpha():
                        numero += 1
                    if len(word.strip()) > len(mot) :
                        mot = word
                        ligne = ligne_prov
                        n_ieme = numero
        return (ligne, n_ieme, mot)

    except FileNotFoundError :
        return "FileNotFoundError : Le fichier n'existe pas."
