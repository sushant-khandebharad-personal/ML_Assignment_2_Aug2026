from sklearn.tree import DecisionTreeClassifier


def create_model():
    return DecisionTreeClassifier(
        random_state=42
    )
