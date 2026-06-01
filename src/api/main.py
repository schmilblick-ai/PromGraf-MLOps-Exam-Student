#imports → constantes → état global → prometheus → pydantic → fonctions → lifespan → app → endpoints
# ═══════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import MAE, RMSE, R2Score
from evidently.presets import DataDriftPreset, RegressionPreset
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import requests
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
import zipfile
import pandas as pd
from datetime import datetime , date, time
import io, typing
from typing import Optional
import joblib, pickle
from pathlib import Path
from pydantic import BaseModel
from typing import List, Any, Dict
import asyncio

# ═══════════════════════════════════════
# 2. CONFIGURATION & CONSTANTES
# ═══════════════════════════════════════
# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Global Variables for Model and Data ---
target               = TARGET = 'cnt'
prediction           = PREDICTION = 'prediction'
numerical_features   = NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
categorical_features = CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']
all_model_feats      = ALL_MODEL_FEATS = NUM_FEATS + CAT_FEATS
my_data_loc  :Path   = Path("../../data/bike.zip")
my_model_loc :Path   = Path("../../models/RFRegressor.pkl")
my_refer_loc :Path   = Path("../../models/Rerefences.csv")

lstPeriods = []
calendar = { 
      "jan11"  : ['2011-01-01 00:00:00' , '2011-01-28 23:00:00'] 
    , "feb11"  : ['2011-01-29 00:00:00' , '2011-02-28 23:00:00']          
    , "week 1" : ['2011-01-29 00:00:00' , '2011-02-07 23:59:59']
    , "week 2" : ['2011-02-07 00:00:00' , '2011-02-14 23:59:59']
    , "week 3" : ['2011-02-15 00:00:00' , '2011-02-21 23:59:59']
}

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
API_EVALUATE_URL = "http://bike-api:8080/evaluate"
API_PREDICT_URL = "http://bike-api:8080/predict"

WEEKLY_PERIODS = {
    'week1_february': ('2011-01-29 00:00:00', '2011-02-07 23:00:00'),
    'week2_february': ('2011-02-08 00:00:00', '2011-02-14 23:00:00'),
    'week3_february': ('2011-02-15 00:00:00', '2011-02-21 23:00:00')
}

DEFAULT_EVAL_PERIOD =   calendar['week 1']
DEFAULT_PERIOD_NAME = 'week 1'

NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']
ALL_MODEL_FEATS = NUM_FEATS + CAT_FEATS
TARGET = 'cnt'

DTEDAY_COL_NAME = 'dteday'
COLUMNS_FOR_EVALUATION_PAYLOAD = ALL_MODEL_FEATS + [TARGET, DTEDAY_COL_NAME]








# état partagé — un simple dict évite le global
app_state = {}


# ═══════════════════════════════════════
# 3. ÉTAT GLOBAL (lecture seule après startup)
# ═══════════════════════════════════════
@dataclass
class AppState:
    model: RandomForestRegressor | None = None  # type cecicela ou None = Valeur initial None
    reference: pd.DataFrame      | None = None
    ToogleTrafic : bool          | None = False

app_state    = AppState()
metrics_lock = asyncio.Lock()




# ═══════════════════════════════════════
# 4. PROMETHEUS MÉTRIQUES
# ═══════════════════════════════════════
# --- Prometheus Metrics Definitions ---
PROM_registry = CollectorRegistry()

# Counter 'api_requests_total', label par endpoint, method, et status code
#mise à jour via l'endpoint /predict).
PROM_api_requests_total           = Counter('bike_requests_total','Total number of API requests',['endpoint', 'method', 'status_code'], registry=PROM_registry)
PROM_api_request_duration_seconds = Histogram('bike_request_duration_seconds','duration of API requests',['endpoint', 'method', 'status_code'], registry=PROM_registry)
#Une métrique de votre choix : Implémentez une métrique supplémentaire jugée pertinente pour le monitoring de ce modèle de régression
#et des dérives (par exemple, model_mape_score, 
PROM_data_drift_detected_status =Gauge("bike_data_drift_detected_status","Glissement de donnée",["??"], registry=PROM_registry)
# Counter 'predictions_ok'
PROM_predictions_ok = Counter('bike_predictions_ok','Number of predictions ok',['confidence_score'],registry=PROM_registry)

#mise à jour via l'endpoint /evaluate).
PROM_model_rmse_score             = Gauge('bike_model_rmse_score','Total number of API requests',['endpoint', 'method', 'status_code'], registry=PROM_registry)
PROM_model_mae_score              = Gauge('bike_model_mae_score','Total number of API requests',['endpoint', 'method', 'status_code'], registry=PROM_registry)
PROM_model_r2_score               = Gauge('bike_model_r2_score','Total number of API requests',['endpoint', 'method', 'status_code'], registry=PROM_registry)
# ....







# ═══════════════════════════════════════
# 5. MODÈLES PYDANTIC
# ═══════════════════════════════════════
# --- Pydantic Models for API Input/Output ---
class Period(BaseModel):
    start: datetime
    end: datetime

class Periods(BaseModel):
    items: Dict[str, Period]

class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: date = Field(..., example="2011-01-01", description="Date of the record in YYYY-MM-DD format.")

class WeekEvaluationInput(BaseModel):
    data: List[BikeSharingInput] 

class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)

class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="List of data points, each containing features and the true target ('cnt').")
    evaluation_period_name: str = Field("unknown_period", description="Name of the period being evaluated (e.g., 'week1_february').")
    model_config = {'arbitrary_types_allowed': True}

class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int

class apiButton(BaseModel):
    tag: str
    switch: bool

# Butons.item("toogleTraffic")
# Button :apiButtons
class apiButtons(BaseModel):
    item:dict[str, bool]


# ═══════════════════════════════════════
# 6. FONCTIONS MÉTIER
# ═══════════════════════════════════════
# --- Data Ingestion and Preparation Functions ---
def _fetch_data0() -> pd.DataFrame:
    #content = requests.get("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", verify=False).content
    content = Path(my_data_loc).read_bytes()
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday']) 
    return raw_data

def _process_data0(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(lambda row: datetime.combine(row.dteday.date(), time(row.hr)), axis=1)
    return raw_data


def _fetch_data() -> pd.DataFrame:
    """Fetches the bike sharing dataset and returns a DataFrame."""
    print("Fetching data from UCI archive...")
    try:
        #content = requests.get(DATASET_URL, verify=False, timeout=60).content
        content = Path(my_data_loc).read_bytes()
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            df = pd.read_csv(z.open("hour.csv"), header=0, sep=',', parse_dates=[DTEDAY_COL_NAME])
        print("Data fetched successfully.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}. Check URL or network connection.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing fetched data: {e}")
        sys.exit(1)

def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Processes raw data, setting a DatetimeIndex as in the exam script."""
    print("Processing raw data...")
    raw_data['hr'] = raw_data['hr'].astype(int)
    raw_data.index = raw_data.apply(
        lambda row: datetime.combine(row[DTEDAY_COL_NAME].date(), time(row.hr)),
        axis=1
    )
    raw_data = raw_data.sort_index()
    print("Data processed successfully.")
    return raw_data




# Gateway pour l'entrainement du modèle
def get_subset(raw_data: pd.DataFrame,period):
    # Reference and current data split
    return raw_data.loc[period[0]:period[1]]

# sample train RandomForestRegressor (_train_and_predict_reference_model) sur les données de janvier 2011
def _train_and_predict_reference_model(reference_data) -> RandomForestRegressor:
    # Train test split ONLY on reference_jan11
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        reference_data[numerical_features + categorical_features],
        reference_data[target], #cnt
        test_size=0.3
    )

    # Model training
    regressor = RandomForestRegressor(random_state = 0, n_estimators = 50)
    regressor.fit(X_train, y_train)
    
    # Predictions
    preds_train = regressor.predict(X_train)
    preds_test  = regressor.predict(X_test)

    # Gateway validation du modèle
    #-----------------------------
    # Add actual target and prediction columns to the training data for later performance analysis
    X_train['target']     = y_train
    X_train['prediction'] = preds_train

    # Add actual target and prediction columns to the test data for later performance analysis
    X_test['target']      = y_test
    X_test['prediction']  = preds_test
    
    return regressor


# ═══════════════════════════════════════
# 7. LIFESPAN (startup / shutdown)
# ═══════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ← tout ce qui est AVANT le yield = startup
    logger.info("🚀 Démarrage...")
    
    #app_state["model"]          = joblib.load("models/model.pkl")
    #app_state["reference_data"] = pd.read_csv("data/reference.csv")

    try:
        # load data
        RFRegressor :RandomForestRegressor

        logger.info( f"my_model_loc.is_file() : {my_model_loc.is_file()}" ) 

        if my_model_loc.is_file():
            raw_data = _process_data(_fetch_data())
            with open(my_model_loc, "rb") as f:
                RFRegressor = pickle.load(f)

            logger.info("Random forest Model loaded successfully")
        else:    
            #let's not train as first time but do use a train endpoint instead
            if False:
                reference_jan11 = get_subset(raw_data,calendar["jan11"])
                RFRegressor :RandomForestRegressor = _train_and_predict_reference_model(reference_jan11)
                pickle.dump(RFRegressor, my_model_loc)
        logger.info("Random forest Model not yet loaded run the train")
    

    except Exception as e:
        logger.error(f"Error loading random Forest model: {e}")
        raise RuntimeError("Failed to load ML model, application cannot start.") from e




    yield  # ← l'API tourne ici
    
    # ← tout ce qui est APRÈS le yield = shutdown
    logger.info("🔴 Arrêt propre de l'API")
    app_state.clear()

## STARTUP AREA
logger.info("🚀 Démarrage de l'API...")



# ═══════════════════════════════════════
# 8. APP FASTAPI
# ═══════════════════════════════════════
# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0",
    lifespan=lifespan
)

# ═══════════════════════════════════════
# 9. ENDPOINTS
# ═══════════════════════════════════════
# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."}

@app.get("/train")
async def train_and_save():
    start_time = time.time() 
    status_code = "200"
    try:
    
        raw_data = _process_data(_fetch_data())
        reference_jan11 = get_subset(raw_data,calendar["jan11"])
        RFRegressor :RandomForestRegressor = _train_and_predict_reference_model(reference_jan11)
        with open(my_model_loc, "wb") as f:
            pickle.dump(RFRegressor, f)
        return {"message": "Success :🚀 To run a bike, take the train"}
    
    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    except Exception as e:
        logger.error(f"Error during train_and_save : {my_model_loc}... Error: {e}")
        status_code = "500"
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")
    finally:
        end_time = time.time()
        # Durée de la requête
        duration = end_time - start_time
        PROM_api_request_duration_seconds.labels(endpoint="/train", method="GET", status_code=status_code).observe(duration)
        PROM_api_requests_total.labels(endpoint="/train", method="GET", status_code=status_code).inc()
    

@app.post("/predict", response_model=PredictionOutput)
async def predict(BikeSharing: BikeSharingInput):
    """
    predict count count of  bike based on BikeSharing details.
    """
    start_time = time.time() # Début du timer pour la durée de la requête

    status_code = "200"

    try:
        if app_state.RFRegressor == None:
            logger.warning("Model not loaded")
            status_code = "500"
            raise HTTPException(status_code=500, detail="Model was not yet loaded, please train before predicting")

        results = app_state.RFRegressor.fit(BikeSharing) #will load the trained model

        if not results:

            logger.error(f"RFRegressor returned empty results for text: {BikeSharing}...")
            status_code = "500"
            raise HTTPException(status_code=500, detail="Model could not regress the input.")

        predicted_target = results
        try:
            confidence_score = results[0]['score']
        except Exception as e:
            logger.error(f"Error during score retrieval at results : {results}")
        
        # Incrementation du counter pour la 
        PROM_predictions_ok.labels(endpoint="/predict", method="POST",confidence_score=confidence_score).inc()


        logger.info(f"Predicted: '{BikeSharing}...' into category: '{results}' with score: {confidence_score:.4f}")
        return PredictionOutput(score=confidence_score)

    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    except Exception as e:
        logger.error(f"Error during prediction for text: {BikeSharing}... Error: {e}")
        status_code = "500"
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")
    finally:
        end_time = time.time()
        # Durée de la requête
        duration = end_time - start_time
        PROM_api_request_duration_seconds.labels(endpoint="/predict", method="POST", status_code=status_code).observe(duration)
        PROM_api_requests_total.labels(endpoint="/predict", method="POST", status_code=status_code).inc()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/toogle")
def toogle(button: apiButton):
    AppState.ToogleTrafic
    return(AppState.ToogleTrafic)

@app.post("/evaluate/period")
def evaluate_period(periods: Periods):
    for name, period in Periods.periods.items():
        logger.info(f"Période {name} : {period.start} → {period.end}")
    return {"received": len(calendar.periods)}

# ✅ en sortie (response)
@app.get("/getPeriods", response_model=Periods)
def get_periods():
    return Periods(items={
        "jan11"  : Period(start="2011-01-01 00:00:00", end="2011-01-28 23:59:59"),
        "feb11"  : Period(start="2011-02-01 00:00:00", end="2011-02-28 23:59:59"),
        "week 1" : Period(start="2011-01-29 00:00:00", end="2011-02-07 23:59:59"),
        "week 2" : Period(start="2011-02-07 00:00:00", end="2011-02-14 23:59:59"),
        "week 3" : Period(start="2011-02-15 00:00:00", end="2011-02-21 23:59:59"),
    })

calendar = { 
      "jan11"  : ['2011-01-01 00:00:00' , '2011-01-31 23:59:59'] 
    , "feb11"  : ['2011-02-01 00:00:00' , '2011-02-28 23:59:59']          
    , "week 1" : ['2011-02-01 00:00:00' , '2011-02-08 23:59:59']
    , "week 2" : ['2011-02-09 00:00:00' , '2011-02-15 23:59:59']
    , "week 3" : ['2011-02-16 00:00:00' , '2011-02-22 23:59:59']
}


@app.post("/setPeriods")
def setWeeks(payload: Periods):
    global lstPeriods
    lstPeriods=payload.items
    return (lstPeriods)


@app.post("/evaluate", response_model=EvaluationReportOutput)
def evaluate(payload: WeekEvaluationInput):
    start_time = time.time() # Début du timer pour la durée de la requête
    status_code = "200"

    # 1. Convertir le payload en DataFrame
    current_df = pd.DataFrame([row.to_dict() for row in payload.data])
    nbitems=current_df.shape[0]

    # 2. Prédictions avec le modèle gelé
    current_df[prediction] = RFRegressor.predict(current_df[numerical_features + categorical_features])

    # 3. Dataset Evidently wrappé
    data_definition = DataDefinition(
        regression=[Regression(target=target, prediction=prediction)],
        numerical_columns=numerical_features + [target, prediction]
    )

    #benchmark
    reference_jan11 = get_subset(raw_data,calendar["jan11"])
    reference_dataset = Dataset.from_pandas(reference_jan11, data_definition=data_definition)
    current_dataset   = Dataset.from_pandas(current_df,     data_definition=data_definition)

    # 4. Rapport Evidently
    report = Report(metrics=[RegressionPreset(), DataDriftPreset()])
    snapshot = report.run(reference_data=reference_dataset, current_data=current_dataset)

    # 5. Extraire les métriques de Evidently
    results = snapshot.metric_results
    rmse  = results[RMSE()].current.value
    mae   = results[MAE()].current.value
    r2    = results[R2Score()].current.value
    drift = results[DataDriftPreset()].drift_detected

    # 6. Mettre à jour Prometheus
    labels = {"endpoint": "/evaluate", "method": "POST", "status_code": "200"}
    PROM_model_rmse_score.labels(**labels).set(rmse)
    PROM_model_mae_score.labels(**labels).set(mae)
    PROM_model_r2_score.labels(**labels).set(r2)


    end_time = time.time()
    # Durée de la requête
    duration = end_time - start_time
    return EvaluationReportOutput(message="evaluation terminated in {duration}"
            , rmse=rmse, mae=mae, r2=r2, drift_detected=drift
            , evaluated_items= nbitems)

@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(PROM_registry), media_type="text/plain")
