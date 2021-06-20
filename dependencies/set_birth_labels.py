def set_birth_labels(model, labels):

    L_birth = len(model.L_birth)

    # Set new labels
    if len(labels) == 0:
        label_max = 0
        avail_labels = []
        L_avail = 0
    else:
        label_max = max(labels)
        avail_labels = list(set(range(label_max)) - set(labels))
        L_avail = len(avail_labels)

    if L_avail >= L_birth:
        l_birth = avail_labels[:L_birth]
    else:
        l_birth = avail_labels + [label_max + i for i in range(L_birth-L_avail)]

    # Set new labels in the model
    model.l_birth = l_birth