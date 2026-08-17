def affecter_tables(tables, reservations):
    """
    Assigne des tables a des reservations selon les regles evoquees dans l'enonce.
    @pre: ``tables`` est une liste de tuples ``(numero_table, capacite)``
          ``reservations`` est une liste de tuples ``(nom, taille_groupe)``
    @post: retourne un dictionnaire ``{nom: numero_table ou None, ...}`` qui fait correspondre chaque
           reservation au numero de table affecte, ou ``None`` si aucune table disponible ne peut
           accueillir le groupe, selon les regles deterministes de l'enonce.
    """
    tables = [numero_table, capacite]
    numero_tables >= 1
    if capacite = 0:
        tables = 0