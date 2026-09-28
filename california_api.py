import joblib
import pandas as pd
from fastapi import FastAPI , HTTPException ,UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel , Field
import io

model = joblib.load("model.pkl")
pipeline = joblib.load("pipline.pkl")

class house_prediction(BaseModel):
    longitude : float = Field(ge=-180, le=180, description="longitude = -118.11" ) ,
    latitude : float = Field(ge=-90, le=90 , description="Latitude = 34.01") ,
    housing_median_age : float = Field(gt=0 , description="housing_median_age = 22.0"),
    total_rooms : float = Field(gt=0,description=" Total room = 1141.0"),
    total_bedrooms : float = Field(gt=0 , description="Total bedroom = 332.0"),
    population : float = Field(gt=0, description="Population = 1189.0"),
    households : float = Field(gt=0 , description="Households = 321.0"),
    median_income : float = Field(gt=0,description="Media income = 2.2042"),
    ocean_proximity : str = Field(description="Location of the house [1H OCEAN,INLAND,NEAR BAY,NEAR OCEAN]")

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": " house prediction api",
        "status":"Running api",
        "endpoint": "Send post request to /predict"
    }
@app.get("/health")
def Health_chack():
    return {"status":"running",
            "model":"Randomregressor",
            }


@app.post("/predict")
def house_prdict(house : house_prediction ):
    try:
        input_data = pd.DataFrame([{
            "longitude": house.longitude,
            "latitude": house.latitude,
            "housing_median_age": house.housing_median_age,
            "total_rooms": house.total_rooms,
            "total_bedrooms":house.total_bedrooms,
            "population": house.population,
            "households": house.households,
            "median_income": house.median_income,
            "ocean_proximity": house.ocean_proximity
        }])
        pip = pipeline.transform(input_data)
        predicted = model.predict(pip)

        return{
            "prediction":predicted.tolist()
        }

    except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"the {e} error are came"
                )


@app.post("/predict_file")
async def predict_file(file:UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please upload csv file only"
        )
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    require_columns = [
        "latitude","longitude",
        "total_rooms","total_bedrooms",
        "housing_median_age","median_income",
        "population",
        "households","ocean_proximity"
    ]

    missing_columns = [
        col for col in require_columns
        if col not in df.columns
    ]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Those columns are missing in the content{missing_columns}"
        )
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="The file must  contain any data"
        )

    try:
        processed_data = pipeline.transform(df)
        prediction = model.predict(processed_data)
        df["prediction"] = prediction
        output = io.BytesIO()
        df.to_excel(output,index = False,engine="openpyxl")
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition":"attachment; filename = prediction.xlsx"}

        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction fail this error come {e}"
        )


