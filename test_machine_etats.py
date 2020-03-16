"""
Test de l'implémentation de la machine d'état
"""
# TODO: ajouter contrainte de temps (0.70 sec entre 2 chgmt d'état)

# Definition des variables d'état
ETAT_A = 0
ETAT_B = 1

# Défnition des fonctions d'état
def etatA(m):
    print('  état A : m = %d' % m)
    if m > 2:
        return ETAT_B
    else:
        return ETAT_A

def etatB(m):
    print('  état B : m = %d' % m)
    return ETAT_B

# Définition du dictionnaire des états
etats = {ETAT_A : etatA,
         ETAT_B : etatB}

# Boucle lecture du flux entrant, frame par frame
etat = 0
iFrame = 0
mesure = -1
while True:
    # simulation de l'évolution des inputs du programme
    mesure = mesure + 1

    # Séparation des cas
    try:
        etat = etats[etat](mesure)
    except:
        print("Cet état n'existe pas. Arrêt du programme. ")
        break
    print('etat = %d' % etat)
    input()

"""
Problèmes :
- le passage d'un résultat d'un état à l'autre, sans savoir quelle variable il faut faire passer
Solutions :
- orienté objet
- créer une super-objet qui référence les objets qui viennent d'être calculés.
"""
