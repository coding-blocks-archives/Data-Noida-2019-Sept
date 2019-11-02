import pickle

with open("vector.pkl", "rb") as vec_file:
    vector = pickle.load(vec_file)
    
with open("model.pkl", "rb") as mod_file:
    model = pickle.load(mod_file)

# type(model)

def predict(sent):
    
    X_test = vector.transform([sent])
    return model.predict(X_test)[0]



