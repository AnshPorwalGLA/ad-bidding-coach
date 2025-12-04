# app.py
import os
from flask import Flask, render_template, request, jsonify
from threading import Thread
import pandas as pd
from prepare_avazu import prepare_avazu
from train_dqn import train_model, MODEL_FILE
from model_evaluation import evaluate_model

app = Flask(__name__, template_folder="templates", static_folder="static")

TRAIN_STATUS = {"running": False, "message": ""}
EVAL_STATUS = {"running": False, "message": "", "roas": None, "spend": None, "revenue": None}

DATA_FILE = "ad_data.csv"

def background_train(total_timesteps, sample_rows):
    try:
        TRAIN_STATUS.update({"running": True, "message": "Training started."})
        model_path = train_model(total_timesteps=total_timesteps, sample_rows=sample_rows)
        TRAIN_STATUS.update({"running": False, "message": f"Training finished. Model saved: {model_path}"})
    except Exception as e:
        TRAIN_STATUS.update({"running": False, "message": f"Training failed: {e}"})

def background_eval():
    try:
        EVAL_STATUS.update({"running": True, "message": "Evaluating..."})
        roas, spend, revenue = evaluate_model()
        EVAL_STATUS.update({"running": False, "message": "Evaluation finished.", "roas": roas, "spend": spend, "revenue": revenue})
    except Exception as e:
        EVAL_STATUS.update({"running": False, "message": f"Evaluation failed: {e}"})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    # if datafile exists, show summary stats
    stats = {}
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        stats = {
            "rows": len(df),
            "avg_ctr": float(df["ctr"].mean()),
            "avg_cvr": float(df["cvr"].mean()),
            "total_spend": float(df["cost"].sum()),
            "total_revenue": float(df["revenue"].sum()),
        }
    return render_template("dashboard.html", stats=stats)

@app.route("/api/generate-data", methods=["POST"])
def generate_data():
    try:
        nrows = int(request.form.get("nrows", 200000))
        prepare_avazu(nrows=nrows)
        return jsonify({"status": "success", "message": "Dataset prepared."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/train", methods=["POST"])
def api_train():
    if TRAIN_STATUS["running"]:
        return jsonify({"status": "running", "message": TRAIN_STATUS["message"]})
    try:
        timesteps = int(request.json.get("timesteps", 20000))
        sample_rows = int(request.json.get("sample_rows", 50000))
        thread = Thread(target=background_train, args=(timesteps, sample_rows))
        thread.start()
        return jsonify({"status": "started", "message": "Training started in background."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/train-status", methods=["GET"])
def train_status():
    return jsonify(TRAIN_STATUS)

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    if EVAL_STATUS["running"]:
        return jsonify({"status": "running", "message": EVAL_STATUS["message"]})
    thread = Thread(target=background_eval)
    thread.start()
    return jsonify({"status": "started", "message": "Evaluation started."})

@app.route("/api/eval-status", methods=["GET"])
def eval_status():
    return jsonify(EVAL_STATUS)

@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    # Accept csv or gz
    filename = file.filename
    save_path = "uploaded_dataset.csv"
    file.save(save_path)
    # convert to ad_data if needed
    try:
        df = pd.read_csv(save_path)
        # Expect click column or ctr/cvr/cost/revenue
        if "click" in df.columns and ("ctr" not in df.columns):
            # prepare similar to Avazu
            df["impressions"] = 1
            df["ctr"] = df["click"]
            df["cvr"] = df["click"].apply(lambda x: 0.2 if x==1 else 0.02)
            import numpy as np
            rng = np.random.default_rng(43)
            df["cost"] = rng.uniform(0.5, 2.0, size=len(df))
            df["revenue"] = df["click"] * rng.uniform(10,40,size=len(df))
            out = df[["ctr","cvr","impressions","cost","revenue"]]
            out.to_csv(DATA_FILE, index=False)
        elif all(c in df.columns for c in ["ctr","cvr","impressions","cost","revenue"]):
            df[["ctr","cvr","impressions","cost","revenue"]].to_csv(DATA_FILE, index=False)
        else:
            return jsonify({"status":"error","message":"Uploaded CSV missing required columns"}),400
        return jsonify({"status":"success","message":"Uploaded and converted dataset saved."})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),400

if __name__ == "__main__":
    app.run(debug=True)
