"""
Flask web application for Voice Call Sentiment Analysis.
"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    send_file,
    abort,
    session,
)
from werkzeug.utils import secure_filename
from auth import login_required, admin_required, login_user, logout_user, load_current_user
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from config import (
    BASE_DIR,
    UPLOADS_DIR,
    PLOTS_DIR,
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
    MODEL_PATH,
    SENTIMENT_MAP,
    CONFIDENCE_THRESHOLD,
)
from database import (
    init_db,
    save_prediction,
    get_prediction_by_id,
    get_statistics,
    get_predictions_filtered,
    export_predictions_csv,
    verify_user,
    create_user,
    get_all_users,
    user_can_access_prediction,
)
from predict import (
    predict_emotion,
    analyze_audio,
    generate_waveform_plot,
    generate_spectrogram_plot,
    warmup_model,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "voice-sentiment-dev-key-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
app.config["PERMANENT_SESSION_LIFETIME"] = 86400 * 7


@app.before_request
def before_request():
    load_current_user()


def current_user_id():
    return session.get("user_id")


def is_admin():
    return session.get("role") == "admin"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def model_ready():
    return os.path.exists(MODEL_PATH)


def save_uploaded_file(file_storage):
    """Save upload to disk; return (unique_name, secure_name, full_path)."""
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filename = secure_filename(file_storage.filename)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    save_path = os.path.join(UPLOADS_DIR, unique_name)
    file_storage.save(save_path)
    return unique_name, filename, save_path

def run_prediction_pipeline(filepath, original_name):
    """Predict, plot, save to DB; return context for templates."""
    result, y, sr = analyze_audio(filepath)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_id = uuid.uuid4().hex[:12]
    waveform_path = os.path.join(PLOTS_DIR, f"wave_{plot_id}.png")
    spec_path = os.path.join(PLOTS_DIR, f"spec_{plot_id}.png")

    # Matplotlib is not thread-safe in this environment, so generate plots sequentially.
    generate_waveform_plot(filepath, waveform_path, y, sr)
    generate_spectrogram_plot(filepath, spec_path, y, sr)

    pred_id = save_prediction(
        filename=original_name,
        filepath=filepath,
        emotion=result["emotion"],
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        user_id=current_user_id(),
    )

    unique_name = os.path.basename(filepath)
    return {
        "result": result,
        "filename": original_name,
        "pred_id": pred_id,
        "audio_url": url_for("static", filename=f"uploads/{unique_name}"),
        "waveform_url": url_for("static", filename=f"plots/wave_{plot_id}.png"),
        "spectrogram_url": url_for("static", filename=f"plots/spec_{plot_id}.png"),
    }


@app.context_processor
def inject_globals():
    return {
        "model_ready": model_ready(),
        "sentiment_map": SENTIMENT_MAP,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


# --- Auth routes ---

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        if session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("dashboard"))

    next_url = request.args.get("next") or request.form.get("next") or url_for("dashboard")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = verify_user(username, password)
        if not user:
            flash("Invalid username or password.", "danger")
        elif user["role"] == "admin":
            flash("Please use the Admin Login page.", "warning")
            return redirect(url_for("admin_login"))
        else:
            login_user(user)
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(next_url)

    return render_template("login.html", next_url=next_url)


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("user_id") and session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    next_url = request.args.get("next") or request.form.get("next") or url_for("admin_dashboard")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = verify_user(username, password)
        if not user:
            flash("Invalid admin credentials.", "danger")
        elif user["role"] != "admin":
            flash("This account is not an administrator.", "danger")
        else:
            login_user(user)
            flash(f"Admin logged in: {user['username']}", "success")
            return redirect(next_url)

    return render_template("admin_login.html", next_url=next_url)


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip() or None
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif create_user(username, password, email=email) is None:
            flash("Username already taken.", "danger")
        else:
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# --- Routes ---

@app.route("/")
def home():
    """Home page."""
    stats = None
    if model_ready() and session.get("user_id"):
        uid = None if is_admin() else current_user_id()
        stats = get_statistics(user_id=uid)
    return render_template("index.html", stats=stats)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    """Upload audio page with drag-and-drop."""
    if request.method == "GET":
        return render_template("upload.html", model_ready=model_ready())

    if not model_ready():
        flash("Model not trained yet. Run: python train_model.py", "warning")
        return redirect(url_for("upload"))

    if "audio" not in request.files:
        flash("No audio file provided.", "danger")
        return redirect(url_for("upload"))

    file = request.files["audio"]
    if file.filename == "":
        flash("No file selected.", "danger")
        return redirect(url_for("upload"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Allowed: WAV, MP3, OGG, FLAC.", "danger")
        return redirect(url_for("upload"))

    unique_name, filename, _ = save_uploaded_file(file)
    return redirect(url_for("predict_audio", filename=unique_name, original=filename))


@app.route("/predict/<filename>")
@login_required
def predict_audio(filename):
    """Run prediction on uploaded file and show results."""
    if not model_ready():
        flash("Model not trained yet.", "warning")
        return redirect(url_for("upload"))

    original = request.args.get("original", filename)
    filepath = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(filepath):
        flash("File not found.", "danger")
        return redirect(url_for("upload"))

    try:
        ctx = run_prediction_pipeline(filepath, original)
    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "danger")
        return redirect(url_for("upload"))

    return render_template("result.html", **ctx)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """Real-time prediction API for WAV/audio uploads."""
    if not model_ready():
        return jsonify({"error": "Model not trained. Run train_model.py first."}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request (field: audio)"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    _, original, save_path = save_uploaded_file(file)

    try:
        result = predict_emotion(save_path)
        pred_id = save_prediction(
            filename=original,
            filepath=save_path,
            emotion=result["emotion"],
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            user_id=current_user_id(),
        )
        result["prediction_id"] = pred_id
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/training")
def training():
    """Training results page with accuracy/loss plots."""
    history_img = os.path.join(PLOTS_DIR, "training_history.png")
    cm_img = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    report_path = os.path.join(BASE_DIR, "saved_model", "classification_report.txt")
    report_text = None
    if os.path.exists(report_path):
        with open(report_path) as f:
            report_text = f.read()

    return render_template(
        "training.html",
        has_history=os.path.exists(history_img),
        has_cm=os.path.exists(cm_img),
        report_text=report_text,
        model_ready=model_ready(),
    )


@app.route("/results")
@login_required
def results():
    """Prediction history with filter and search."""
    sentiment = request.args.get("sentiment", "").strip()
    search = request.args.get("q", "").strip()
    if sentiment and sentiment not in ("Positive", "Neutral", "Negative"):
        sentiment = ""
    uid = None if is_admin() else current_user_id()
    predictions = get_predictions_filtered(
        sentiment=sentiment or None,
        search=search or None,
        user_id=uid,
        limit=100,
    )
    return render_template(
        "results.html",
        predictions=predictions,
        current_sentiment=sentiment,
        search_query=search,
    )


@app.route("/export/csv")
@login_required
def export_csv():
    """Download prediction history as CSV."""
    sentiment = request.args.get("sentiment", "").strip() or None
    search = request.args.get("q", "").strip() or None
    if sentiment and sentiment not in ("Positive", "Neutral", "Negative"):
        sentiment = None
    uid = None if is_admin() else current_user_id()
    csv_data = export_predictions_csv(sentiment=sentiment, search=search, user_id=uid)
    buffer = BytesIO(csv_data.encode("utf-8-sig"))
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"predictions_{datetime.utcnow().strftime('%Y%m%d')}.csv",
    )


@app.route("/dashboard")
@login_required
def dashboard():
    """User dashboard — own predictions only."""
    if is_admin():
        return redirect(url_for("admin_dashboard"))
    stats = get_statistics(user_id=current_user_id())
    predictions = get_predictions_filtered(user_id=current_user_id(), limit=20)
    return render_template(
        "dashboard.html",
        stats=stats,
        predictions=predictions,
        uploads=[],
    )


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Admin dashboard — all users and predictions."""
    stats = get_statistics()
    predictions = get_predictions_filtered(limit=50)
    users = get_all_users()
    uploads = []
    if os.path.isdir(UPLOADS_DIR):
        uploads = sorted(
            [f for f in os.listdir(UPLOADS_DIR) if os.path.isfile(os.path.join(UPLOADS_DIR, f))],
            key=lambda x: os.path.getmtime(os.path.join(UPLOADS_DIR, x)),
            reverse=True,
        )[:30]
    return render_template(
        "admin_dashboard.html",
        stats=stats,
        predictions=predictions,
        users=users,
        uploads=uploads,
    )


@app.route("/about")
def about():
    """About project page."""
    return render_template("about.html")


@app.route("/report/<int:pred_id>")
@login_required
def download_report(pred_id):
    """Download prediction report as PDF."""
    pred = get_prediction_by_id(pred_id)
    if not pred:
        abort(404)
    if not user_can_access_prediction(pred, current_user_id(), session.get("role")):
        flash("You cannot access this report.", "danger")
        return redirect(url_for("results"))

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=20,
        textColor=colors.HexColor("#0d6efd"),
    )
    story = [
        Paragraph("Voice Sentiment Analysis Report", title_style),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]),
        Spacer(1, 0.3 * inch),
    ]

    data = [
        ["Field", "Value"],
        ["File Name", pred["filename"]],
        ["Detected Emotion", pred["emotion"].title()],
        ["Sentiment", pred["sentiment"]],
        ["Confidence", f"{pred['confidence']:.2f}%"],
        ["Recorded At", pred["created_at"]],
    ]
    table = Table(data, colWidths=[2.2 * inch, 4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.4 * inch))
    story.append(
        Paragraph(
            "This report was generated by the AI Voice Call Sentiment Analysis system "
            "using deep learning on speech audio features.",
            styles["Italic"],
        )
    )
    doc.build(story)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"sentiment_report_{pred_id}.pdf",
    )


@app.route("/api/stats")
@login_required
def api_stats():
    """JSON stats for dashboard charts."""
    uid = None if is_admin() else current_user_id()
    return jsonify(get_statistics(user_id=uid))


@app.errorhandler(413)
def too_large(e):
    flash(f"File too large. Max size: {MAX_UPLOAD_SIZE_MB} MB.", "danger")
    return redirect(url_for("upload"))


@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404


if __name__ == "__main__":
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "saved_model"), exist_ok=True)
    init_db()
    from predict import _use_numpy

    print("Voice Sentiment Analysis App")
    print(f"Model ready: {model_ready()}")
    print(
        "Inference: NumPy (Windows-safe, no TensorFlow)"
        if _use_numpy()
        else "Inference: TensorFlow/Keras"
    )
    if model_ready():
        print("Warming up model...", end=" ", flush=True)
        if warmup_model():
            print("done")
        else:
            print("skipped")
    print("Admin login: username=admin  password=admin123")
    
    # Run with environment-aware settings
    debug = os.getenv("FLASK_ENV") == "development"
    port = int(os.getenv("PORT", 5000))
    app.run(debug=debug, host="0.0.0.0", port=port)
