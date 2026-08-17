def conway(lst) :
    """
    Calcule le terme suivant de la suite de Conway a partir de ``lst``.
    @pre:  ``lst`` est une liste non-vide d'entiers positifs
    @post: retourne une liste representant le terme suivant ``lst``
    dans la suite de Conway. Pour chaque sequence de ``n`` nombres identiques ``x``
    consecutifs dans ``lst`` (n  1),  ``conway(lst)`` contient les nombres ``n`` et ``x``.

    Exemple: ``conway([1,1,1,2,2,1]) == [3,1,2,2,1,1]`` (trois fois 1, deux fois 2, une fois 1)
    """
    l=[]