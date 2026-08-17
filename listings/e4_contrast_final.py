def moyenne_ponderee(notes_etudiant, credits_cours):
    current_notes = notes_etudiant.head
    current_credit = credits_cours.head
    moy = 0
    div = 0
    if credits_cours.length > notes_etudiant.length:
        while current_credit != None:

            if current_credit.cargo[0] == current_notes.cargo[0]:

                moy += current_notes.cargo[1] * current_credit.cargo[1]
                div += current_credit.cargo[1]
                if current_notes.next != None:
                    current_notes = current_notes.next
            current_credit = current_credit.next
    else:
        while current_notes != None:

            if current_credit.cargo[0] == current_notes.cargo[0]:

                moy += current_notes.cargo[1] * current_credit.cargo[1]
                div += current_credit.cargo[1]
                if current_credit.next != None:
                    current_credit = current_credit.next
            current_notes = current_notes.next
    return moy / div
