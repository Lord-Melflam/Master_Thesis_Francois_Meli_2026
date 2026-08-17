class Commercial(Employe) :
    def __init__(self, nom, taux_horaire, heures_travaillees, produits_vendus, bonus_par_produit ):
        self.nom = nom
        self.taux_horaire = taux_horaire
        self.heures_travaillees = heures_travaillees
        self.produits_vendus = produits_vendus
        self.bonus_par_produit = bonus_par_produit
    def __bonus__(self):
        return self.produits_vendus * self.bonus_par_produit

    def salaire(self, ):
        salire = super().salaire() + __bonus__(self)
        return salaire
    def __str(self):
        return salaire

