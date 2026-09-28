import pandas as pd
import numpy as np
import  joblib
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



MODEL_FILE = "model.pkl"
PIPLINE_FILE = "pipline.pkl"

# create a function for pipline

def build_pipline(num_attributes,cat_attributes):
    num_attribute_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())]
    )

    # 5  cereate pipeline for the cat_attribute 

    cat_attribute_pipline = Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    #  6  mearg the two pipline for the full pipline 

    full_pipline = ColumnTransformer([
        ("num",num_attribute_pipeline,num_attributes),
        ("cat",cat_attribute_pipline,cat_attributes)
    ])

    return full_pipline


# for the training the model

if not os.path.exists(MODEL_FILE):
    data = pd.read_csv(r"D:\New data science\Practice ml using sikit-learn\California\housing.csv\housing.csv")

        # 3  cerate a new data column for the test splite
    data["income_cat"] = pd.cut(data["median_income"],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])

    splite = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in splite.split(data,data["income_cat"]):
        train_data_set = data.loc[train_index].drop("income_cat",axis=1)
        test_data_set = data.loc[test_index].drop("income_cat",axis=1)

    df = test_data_set.copy()


    featuer = df["median_house_value"].copy()
    df = df.drop("median_house_value",axis=1)

    cat_attribute = ["ocean_proximity"]
    num_attribute = df.drop("ocean_proximity",axis=1).columns.tolist()

    pipline = build_pipline(num_attributes=num_attribute,cat_attributes=cat_attribute)


    houseing_pre = pipline.fit_transform(df)

    #pripare model

    model = RandomForestRegressor(random_state=42)
    model.fit(houseing_pre,featuer)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipline,PIPLINE_FILE)


    print("The model is train")

else:
    modell = joblib.load("model.pkl")
    piplinee = joblib.load("pipline.pkl")

    input_file = pd.read_csv("input_c.csv")

    transform_input = piplinee.transform(input_file)
    predict = modell.predict(transform_input)
    input_file["median_house_value"] = predict

    input_file.to_csv("output_c.csv",index=False)
    print("the output file is genrate")
    