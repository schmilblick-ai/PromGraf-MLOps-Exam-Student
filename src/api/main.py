#imports → constantes → état global → prometheus → pydantic → fonctions → lifespan → app → endpoints
# ═══════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import MAE, RMSE, R2Score, MAPE
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
from datetime import datetime , date, time as dt_time
import time
import io, typing
from typing import Optional
import joblib, pickle
from pathlib import Path
from typing import List, Any, Dict
import asyncio
import sys, json
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
      "jan11"  : ['2011-01-01 00:00:00' , '2011-01-28 23:59:59'] 
    , "feb11"  : ['2011-01-29 00:00:00' , '2011-02-28 23:59:59']          
    , "week1_february" : ['2011-01-29 00:00:00' , '2011-02-07 23:59:59']
    , "week2_february" : ['2011-02-07 00:00:00' , '2011-02-14 23:59:59']
    , "week3_february" : ['2011-02-15 00:00:00' , '2011-02-21 23:59:59']
}

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
API_EVALUATE_URL = "http://bike-api:8080/evaluate"
API_PREDICT_URL = "http://bike-api:8080/predict"

WEEKLY_PERIODS = {
    'week1_february': ('2011-01-29 00:00:00', '2011-02-07 23:00:00'),
    'week2_february': ('2011-02-08 00:00:00', '2011-02-14 23:00:00'),
    'week3_february': ('2011-02-15 00:00:00', '2011-02-21 23:00:00')
}

DEFAULT_EVAL_PERIOD =   calendar['week1_february']
DEFAULT_PERIOD_NAME = 'week1_february'

NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']
ALL_MODEL_FEATS = NUM_FEATS + CAT_FEATS
TARGET = 'cnt'

DTEDAY_COL_NAME = 'dteday'
COLUMNS_FOR_EVALUATION_PAYLOAD = ALL_MODEL_FEATS + [TARGET, DTEDAY_COL_NAME]



# état partagé — un simple dict évite le global

# ═══════════════════════════════════════
# 3. ÉTAT GLOBAL (lecture seule après startup)
# ═══════════════════════════════════════
@dataclass
class AppState:
    RFRegressor:   RandomForestRegressor | None = None  # type cecicela ou None = Valeur initial None
    reference:     pd.DataFrame          | None = None
    ToogleTrafic : bool                  | None = False

app_state  = AppState()
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
#sur une régression on obtient juste une prédiction, pas de score
#on peut juste représenté dans un gauge, ce qui a été prédit (ou le min et max des prédictions ?)
#et des dérives (par exemple, model_mape_score, 
PROM_api_data_predict_level       = Gauge("bike_data_predict_level","Last data Prediction",['endpoint', 'method', 'status_code'], registry=PROM_registry)
# Counter 'predictions_ok'
PROM_api_predictions_ok           = Counter('bike_predictions_ok','Number of predictions ok',['endpoint', 'method', 'status_code'],registry=PROM_registry)

#mise à jour via l'endpoint /evaluate).
# pour mettre à jour cette section, cela se fait une fois les noms de métric découvert dans evidently du endpoint evaluate
PROM_model_rmse             = Gauge('bike_model_rmse','Model RMSE'            ,['endpoint', 'method', 'status_code'], registry=PROM_registry)
PROM_model_mape             = Gauge('bike_model_mape_mean' ,'model MAPE mean' ,['endpoint', 'method', 'status_code'], registry=PROM_registry)
PROM_model_mae              = Gauge('bike_model_mae_mean' ,'model MAE mean'   ,['endpoint', 'method', 'status_code'], registry=PROM_registry)
PROM_model_mae_std          = Gauge('bike_model_mae_std' ,'model MAE std'     ,['endpoint', 'method', 'status_code'], registry=PROM_registry)
PROM_model_r2               = Gauge('bike_model_r2'  ,'Model R2'              ,['endpoint', 'method', 'status_code'], registry=PROM_registry)
PROM_model_drift_count      = Gauge('bike_model_drift_count'  ,'Model col count drift' ,['endpoint', 'method', 'status_code'], registry=PROM_registry)
PROM_model_drift_share      = Gauge('bike_model_drift_share'  ,'Model col share drift' ,['endpoint', 'method', 'status_code'], registry=PROM_registry)

# ....




# ═══════════════════════════════════════
# 5. MODÈLES PYDANTIC
# ═══════════════════════════════════════
# --- Pydantic Models for API Input/Output ---
class Period(BaseModel):
    start: datetime
    end: datetime

class Periods(BaseModel):
    #key et value
    dict: Dict[str, Period]

class PeriodName(BaseModel):
    #'data': {'data': evaluation_data_payload, 'evaluation_period_name': period_name}
    evaluation_period_name: str

class BikeSharingInput(BaseModel):
    temp      : float = Field(..., example=0.24)
    atemp     : float = Field(..., example=0.2879)
    hum       : float = Field(..., example=0.81)
    windspeed : float = Field(..., example=0.0)
    mnth      : int = Field(..., example=1)
    hr        : int = Field(..., example=0)
    weekday   : int = Field(..., example=6)
    season    : int = Field(..., example=1)
    holiday   : int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday    : date | None = Field(..., example="2011-01-01", description="Date of the record in YYYY-MM-DD format.") 

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
    r2: Optional[float]
    drift_detected: int
    drift_shr_detected: float
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
    raw_data.index = raw_data.apply(lambda row: datetime.combine(row.dteday.date(), dt_time(row.hr)), axis=1)
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
        lambda row: datetime.combine(row[DTEDAY_COL_NAME].date(), dt_time(row.hr)),
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
    global app_state
    # ← tout ce qui est AVANT le yield = startup
    logger.info("🚀 Démarrage...")
    
    #app_stat e["model"]          = joblib.load("models/model.pkl")
    #app_stat e["reference_data"] = pd.read_csv("data/reference.csv")

    try:
        # load data

        logger.info( f"my_model_loc.is_file() : {my_model_loc.is_file()}" ) 


        if my_model_loc.is_file():
            raw_data = _process_data(_fetch_data())
            with open(my_model_loc, "rb") as f:
                app_state.RFRegressor = pickle.load(f)

            logger.info("Random forest Model loaded successfully")
        else:    
            #let's not train as first time but do use a train endpoint instead
            if False:
                reference_jan11 = get_subset(raw_data,calendar["jan11"])
                RFRegressor :RandomForestRegressor = _train_and_predict_reference_model(reference_jan11)
                pickle.dump(RFRegressor, my_model_loc)
            logger.info("Random forest Model not yet loaded : run the /train enpoint first")
    

    except Exception as e:
        logger.error(f"Error loading random Forest model: {e}")
        raise RuntimeError("Failed to load ML model, application cannot start.") from e


    yield  # ← l'API tourne ici
    
    # ← tout ce qui est APRÈS le yield = shutdown
    logger.info("🔴 Arrêt propre de l'API")
    del app_state

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
    start_time = time.perf_counter() 
    status_code = "200"
    try:
    
        raw_data = _process_data(_fetch_data())
        reference_jan11 = get_subset(raw_data,calendar["jan11"])
        app_state.RFRegressor = _train_and_predict_reference_model(reference_jan11)
        with open(my_model_loc, "wb") as f:
            pickle.dump(app_state.RFRegressor, f)
        return {"message": "Success :🚀 To run a bike, take the train"}
    
    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    except Exception as e:
        logger.error(f"Error during train_and_save : {my_model_loc}... Error: {e}")
        status_code = "500"
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")
    finally:
        end_time = time.perf_counter()
        # Durée de la requête
        duration = end_time - start_time
        PROM_api_request_duration_seconds.labels(endpoint="/train", method="GET", status_code=status_code).observe(duration)
        PROM_api_requests_total.labels(endpoint="/train", method="GET", status_code=status_code).inc()
    

@app.post("/predict", response_model=PredictionOutput)
async def predict(BikeSharing: BikeSharingInput):
    """
    predict count count of  bike based on BikeSharing details.
    """
    start_time = time.perf_counter() # Début du timer pour la durée de la requête

    status_code = "200"
    #time.sleep(0.3)
    try:
        if app_state.RFRegressor == None:
            logger.warning("Model not loaded")
            status_code = "500"
            raise HTTPException(status_code=500, detail="Model was not yet loaded, please train before predicting")
        #On converti l'input en df et on projete sur les features
        logger.info(BikeSharing)
        inputdf = pd.DataFrame([BikeSharing.model_dump()])[numerical_features + categorical_features]
        logger.info(inputdf)
        #inputdf=BikeSharing.model_dump(exclude={"dteday"})
        results = app_state.RFRegressor.predict(inputdf) #will predict with the trained model
        logger.info(results)

        if not results:

            logger.error(f"RFRegressor returned empty results for text: {BikeSharing}...")
            status_code = "500"
            raise HTTPException(status_code=500, detail="Model could not regress the input.")
                
        # Incrementation du counter pour les predictions ok 
        PROM_api_predictions_ok.labels(endpoint="/predict", method="POST",status_code=status_code).inc()

        logger.info(f"Predicted: '{BikeSharing}...' level count : '{results}'")
        #push le score sur la gauge
        PROM_api_data_predict_level.labels(endpoint="/predict", method="POST", status_code=status_code).set(results[0])

        return PredictionOutput(predicted_count=results[0])

    except HTTPException as e:
        status_code = str(e.status_code)
        logger.info(f"except HTTPException:")
        raise
    except Exception as e:
        logger.error(f"Error during prediction for : {BikeSharing}... Error: {e}")
        status_code = "500"
        logger.info(f"except Exception:")        
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")
    finally:
        end_time = time.perf_counter()
        # Durée de la requête
        duration = end_time - start_time
        #stock le quoi qu'il arrive
        logger.info(f"finally:") 
        PROM_api_request_duration_seconds.labels(endpoint="/predict", method="POST", status_code=status_code).observe(duration)
        PROM_api_requests_total.labels(endpoint="/predict", method="POST", status_code=status_code).inc()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/toogle")
def toogle(button: apiButton):
    app_state.ToogleTrafic = not app_state.ToogleTrafic
    button.switch =  app_state.ToogleTrafic
    return(button)

@app.post("/evaluate/period")
def evaluate_period(periods: Periods):
    logger.info(periods)
    for name,period in periods.dict.items():
        logger.info(f"Période {name} : {period.start} → {period.end}")
    return {"received": len(periods.dict)}

#internal definition of calendars in various trial shapes
calendar1=Periods(dict={
        "jan11"  : Period(start="2011-01-01 00:00:00", end="2011-01-28 23:59:59"),
        "feb11"  : Period(start="2011-02-01 00:00:00", end="2011-02-28 23:59:59"),
        "week1_february" : Period(start="2011-01-29 00:00:00", end="2011-02-07 23:59:59"),
        "week2_february" : Period(start="2011-02-07 00:00:00", end="2011-02-14 23:59:59"),
        "week3_february" : Period(start="2011-02-15 00:00:00", end="2011-02-21 23:59:59"),
    })

calendar = { 
      "jan11"  : ['2011-01-01 00:00:00' , '2011-01-31 23:59:59'] 
    , "feb11"  : ['2011-02-01 00:00:00' , '2011-02-28 23:59:59']          
    , "week1_february" : ['2011-02-01 00:00:00' , '2011-02-08 23:59:59']
    , "week2_february" : ['2011-02-09 00:00:00' , '2011-02-15 23:59:59']
    , "week3_february" : ['2011-02-16 00:00:00' , '2011-02-22 23:59:59']
}



# ✅ en sortie (response)
@app.get("/getPeriods", response_model=Periods)
def get_periods():
    return calendar1


@app.post("/setPeriods")
def setWeeks(selectedPeriods: Periods):
    global lstPeriods
    lstPeriods=selectedPeriods.dict
    return (lstPeriods)


def get_by_name(results, name):
    for k, v in results.items():
        if v.display_name == name:
            return v
    return None


@app.post("/evaluate", response_model=EvaluationReportOutput)
#def evaluate(WeekEvaluation: WeekEvaluationInput):
#def evaluate(aPeriodName: PeriodName):
def evaluate(aPeriodName: EvaluationData):
    start_time = time.perf_counter() # Début du timer pour la durée de la requête
    status_code = "200"

    raw_data = _process_data(_fetch_data())
    # 1. Convertir le payload en DataFrame
    logger.info(aPeriodName.evaluation_period_name)

    current_df = get_subset(raw_data,calendar[aPeriodName.evaluation_period_name])
    #logger.info(current_df.shape)
    nbitems=current_df.shape[0]
    logger.info(nbitems)

    # 2. Prédictions avec le modèle gelé
    current_df[prediction] = app_state.RFRegressor.predict(current_df[numerical_features + categorical_features])

    # 3. Dataset Evidently wrappé
    data_definition = DataDefinition(
        regression=[Regression(target=target, prediction=prediction)],
        numerical_columns=numerical_features + [target, prediction]
    )

    #benchmark
    raw_data = _process_data(_fetch_data())
    reference_jan11 = get_subset(raw_data,calendar["jan11"])
    reference_jan11[prediction] = app_state.RFRegressor.predict(reference_jan11[numerical_features + categorical_features])

    logger.info(current_df.shape)
    logger.info(reference_jan11.shape)
    
    logger.info(current_df.head())

    logger.info(reference_jan11.head())

    logger.info(reference_jan11[[target, prediction]].isnull().sum())
    logger.info(current_df[[target, prediction]].isnull().sum())
    logger.info(reference_jan11[[target, prediction]].describe())
    logger.info(current_df[[target, prediction]].describe())

    reference_dataset = Dataset.from_pandas(reference_jan11, data_definition=data_definition)
    current_dataset   = Dataset.from_pandas(current_df     , data_definition=data_definition)

    # 4. Rapport Evidently
    #initialisation des commandes de métriques, certaine métriques peuvent être en overlap entre des présets ou des demandes séparé
    rmse, mape, mae, r2score,datadrift = RMSE(), MAPE(), MAE(), R2Score(), DataDriftPreset()
    logger.info([rmse, mape, mae, r2score,datadrift])
    #report = Report(metrics=[RegressionPreset(), rmse, mae, r2score,datadrift ])

    #Construction du report evidently et Collection des métriques
    report = Report(metrics=[rmse, mape, mae, r2score,datadrift ])
    snapshot = report.run(reference_data=reference_dataset, current_data=current_dataset)

    # 5. Extraire les métriques de Evidently pour les envoyer vers prométhéus
    # logger.info(snapshot)
    # alors là, y a tout qui plance si on y prend garde - 
    # la méthode va consister à afficher les métriques keys qu'on a fabriqué juste avant et
    # à progressivement les push vers prometheus, en les définissant progressivement

    results = snapshot.metric_results

    # AVOIR un affichage systématique des clefs de métrics obtenues et des labels associés
    # objectif, savoir si un accés par clés ou par labels est approprié pour différencier les métriques du report
    # récupération du tableau des cléfs de métric
    resuKeys=list(results.keys())
    # affichage du display name
    for z in [(resuKeys[k],results[resuKeys[k]].display_name) for k in range(len(resuKeys))]:
        logger.info(z)
    #attention, je donne cette méthode, mais dans le debugger de vscode les quatre premières métrics ne sont jamais "display"

    # logger.info(results) - RECUPERATION DES METRIQUES
    rmse  = get_by_name(results, "RMSE")
    mape  = get_by_name(results, "Mean Absolute Percentage Error")
    mae   = get_by_name(results, "Mean Absolute Error")
    r2    = get_by_name(results, "R2 Score")
    drift = get_by_name(results, "Count of Drifted Columns")

    # RECUPERATION DES VALEURS DES METRIQUES
    # certaines métriques renvoient deux valeurs (mean, std) ou (count, share) par exemple
    v_rmse    = rmse.value
    logger.info(f"mape properties : {type(mape)}")
    logger.info([a for a in dir(mape) if not a.startswith("_")])
    # Ce qu'on apprend, evidently.core.metric_types.MeanStdValue dit que la métrique contient mean et std comme propriété
    # see the litle specification "Metric result" in MAPE() ref https://docs.evidentlyai.com/metrics/all_metrics

    v_mape    = mape.mean.value
    v_mae     = mae.mean.value
    v_mae_std = mae.std.value
    v_r2      = r2.value
    v_drift_count = drift.count.value
    v_drift_share = drift.share.value

    
    # 6. Mettre à jour Prometheus en déclarant au dessus les métriques manquantes
    labels = {"endpoint": "/evaluate", "method": "POST", "status_code": "200"}
    PROM_model_rmse.labels(**labels).set(v_rmse)
    PROM_model_mape.labels(**labels).set(v_mape)
    PROM_model_mae.labels(**labels).set(v_mae)
    PROM_model_mae_std.labels(**labels).set(v_mae_std)
    PROM_model_r2.labels(**labels).set(v_r2)
    PROM_model_drift_count.labels(**labels).set(v_drift_count)
    PROM_model_drift_share.labels(**labels).set(v_drift_share)
    
    #une fois définit, revoyer la définition des GAUGES en haut du script, attention, 
    # il n'est pas vraiment possible de les définir ici can le registry prometheus de collect est global 
    # CE PROCESS EST FASTIDIEUX SURTOUT SI ON DECOUVRE LES CHANGES EVIDENTLY

    end_time = time.perf_counter()
    # Durée de la requête
    duration = end_time - start_time
    return EvaluationReportOutput(message=f"evaluation for {aPeriodName.evaluation_period_name} terminated in {duration:.3f}"
            , rmse=v_rmse, mape=v_mape, mae=v_mae, r2=v_r2
            , drift_detected=v_drift_count 
            , drift_shr_detected=v_drift_share
            , evaluated_items= nbitems)


@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(PROM_registry), media_type="text/plain")
