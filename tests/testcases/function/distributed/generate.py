def generate(array):
    print("{", end="")
    for (i, item) in enumerate(array):
        if i != len(array) - 1:
            print(f"{item}, ", end="")
        else:
            print(f"{item}", end="")
    print("}")
