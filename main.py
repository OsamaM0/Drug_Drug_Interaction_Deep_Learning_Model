import numpy as np
import pandas as pd
from keras.models import load_model
from flask import Flask, request
import json
import time


def load_npz(file):
    loaded_dict = np.load(file)
    # Convert the loaded dictionary to a regular Python dictionary
    dict_file = dict(loaded_dict)
    return dict_file


d_feature = load_npz(r"D:\Git Hub\Git_Hub\Drug_Drug_Interaction_Deep_Learning_Model\Drug_Smile_Features.npz")
d_event = load_npz(r"D:\Git Hub\Git_Hub\Drug_Drug_Interaction_Deep_Learning_Model\Event_Number.npz")


def predict_event(drug1: str, drug2: str):
    drugs_feature = np.hstack((d_feature[drug1], d_feature[drug2]))
    drugs_feature = drugs_feature.reshape((1,1144))

    new_model = load_model(r"D:\Git Hub\Git_Hub\Drug_Drug_Interaction_Deep_Learning_Model\model_fold_0.h5")
    event = d_event[str(np.argmax(new_model.predict(drugs_feature), axis=1)[0])]
    event = str(event)
    # Replace the first occurrence of "name" with drug1
    new_event = event.replace("name", drug1, 1)
    # Replace the second occurrence of "name" with drug2
    new_event = new_event.replace("name", drug2, 1)
    return new_event




app = Flask('__name__')


@app.route("/", methods=['GET'])
def home_page():
    data_set =  {"Message": "Welcom To Drug Drug Interaction API Model "}
    json_dump = json.dumps(data_set)

    return json_dump


@app.route("/drugs", methods=['GET'])
def request_page():
    user_query = str(request.args.get("drugs"))  # /user/?user=
    drug1, drug2 = user_query.split("-")
    print(drug1)
    print(drug2)
    print(predict_event(drug1, drug2))
    data_set = {"Event": predict_event(drug1, drug2)}
    json_dump = json.dumps(data_set)

    return json_dump


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=0000)



# Example http://127.0.0.1:61952/drugs?drugs=Glucosamine-Anagrelide

