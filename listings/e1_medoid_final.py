class Employe:

    def __init__(self, nom, taux_horaire, heures_travaillees):
        """ Initialise un employe
        pre: `nom` est une chaine de caracteres
             `taux_horaire` est un float positif
             `heures_travaillees` est un entier positif
        post: initialise un employe avec `taux_horaire` et `heures_travaillees`
        """
        self.nom = nom
        self.taux_horaire = taux_horaire
        self.heures_travaillees = heures_travaillees

    def salaire(self):
        """ Calcule le salaire de l'employe
        pre:  -
        post: retourne le salaire de cet employe calcule comme le
              produit du taux_horaire et des heures_travaillees
        """
        return self.taux_horaire * self.heures_travaillees