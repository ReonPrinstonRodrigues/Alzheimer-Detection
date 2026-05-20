"""
app.py — Flask Web Application for Alzheimer's Disease Detection
Provides user authentication, MRI scan upload, and prediction using trained models.
"""

import os

# Suppress TensorFlow verbose logging (must be before TF import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only (saves memory)

import re
import io
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session, jsonify, send_file, make_response)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from database import init_db, add_user, get_user_by_email, add_prediction

# PDF generation (optional — installed via: pip install fpdf2)
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# Limit TensorFlow to use minimal memory
tf.get_logger().setLevel('ERROR')

# ─── App Configuration ────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'alzheimer-detection-secret-key-2026'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model and class configurations
CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

AVAILABLE_MODELS = {
    'cnn': 'CNN (Custom)',
    'mlp': 'MLP (Multi-Layer Perceptron)',
    'vgg16': 'VGG16 (Transfer Learning)',
    'resnet50': 'ResNet50 (Transfer Learning)',
    'efficientnet': 'EfficientNetB0 (Transfer Learning)',
    'mobilenet': 'MobileNetV2 (Transfer Learning)'
}

# Clinical information per class
CLASS_INFO = {
    'Non Demented': {
        'color': '#2ecc71',
        'badge_class': 'success',
        'severity': 'Normal',
        'cause': '''<ul class="text-start mb-0 ps-3 text-muted small">
            <li class="mb-2"><strong>Healthy Brain Structure:</strong> No significant brain atrophy, cortical shrinkage, or ventricular enlargement detected in the MRI scan.</li>
            <li class="mb-2"><strong>Normal Aging:</strong> Brain structure and cognitive function are generally consistent with normal aging.</li>
            <li><strong>Absence of Biomarkers:</strong> There is no noticeable indication of widespread beta-amyloid plaques or neurofibrillary tau tangles that characterize dementia.</li>
        </ul>''',
        'treatment': '''<ul class="text-start mb-0 ps-3 text-muted small">
            <li class="mb-2"><strong>Diet & Nutrition:</strong> Adopt a brain-healthy diet such as the Mediterranean or MIND diet, rich in antioxidants, leafy greens, and omega-3 fatty acids.</li>
            <li class="mb-2"><strong>Physical Exercise:</strong> Engage in at least 150 minutes of moderate aerobic exercise weekly (like brisk walking or swimming) to promote cardiovascular and brain health.</li>
            <li class="mb-2"><strong>Cognitive Engagement:</strong> Challenge the brain continuously through reading, puzzles, learning new skills, or social engagement.</li>
            <li><strong>Systemic Health:</strong> Regularly monitor and manage blood pressure, cholesterol levels, and blood sugar, as these are critical for long-term brain health.</li>
        </ul>'''
    },
    'Very Mild Demented': {
        'color': '#f39c12',
        'badge_class': 'warning',
        'severity': 'Mild Cognitive Impairment (Early Stage)',
        'cause': '''<ul class="text-start mb-0 ps-3 text-muted small">
            <li class="mb-2"><strong>Early Protein Accumulation:</strong> Often linked to the initial accumulation of beta-amyloid plaques or tau tangles in localized brain regions.</li>
            <li class="mb-2"><strong>Subtle Atrophy:</strong> The MRI may appear mostly normal or show only very minimal hippocampal volume loss, making it difficult to detect visually without AI assistance.</li>
            <li><strong>Symptomatic Manifestation:</strong> Manifests as subtle memory lapses (e.g., forgetting familiar, everyday words or consistently losing items) that are typically noticed only by the individual or close family members.</li>
        </ul>''',
        'treatment': '''<ul class="text-start mb-0 ps-3 text-muted small">
            <li class="mb-2"><strong>Early Medical Intervention:</strong> Promptly consult a neurologist for a comprehensive baseline cognitive clinical evaluation.</li>
            <li class="mb-2"><strong>Emerging Therapies:</strong> Discuss candidacy for FDA-approved early-stage immunotherapies (e.g., Lecanemab/Leqembi) aimed at clearing amyloid plaques.</li>
            <li class="mb-2"><strong>Cognitive Training:</strong> Undergo rigorous, structured cognitive training programs to strengthen neural pathways and build cognitive reserve.</li>
            <li><strong>Strict Lifestyle Control:</strong> Intensify cardiovascular health management, strict adherence to a brain-healthy diet, and regular aerobic exercise to slow progression.</li>
        </ul>'''
    },
    'Mild Demented': {
        'color': '#e67e22',
        'badge_class': 'orange',
        'severity': 'Early-Stage Alzheimer\'s',
        'cause': '''<ul class="text-start mb-0 ps-3 text-muted small">
            <li class="mb-2"><strong>Progressive Brain Damage:</strong> The gradual spread of plaques and tangles causes noticeable damage to neural networks, particularly attacking the hippocampus which is crucial for memory formation.</li>
            <li class="mb-2"><strong>Measurable MRI Atrophy:</strong> MRI typically begins to show clear, measurable brain atrophy (shrinkage) in the temporal and parietal lobes.</li>
            <li><strong>Functional Impact:</strong> Symptoms escalate to obvious short-term memory deficits, increasing difficulty with complex planning, organizing, and keeping track of recent events.</li>
        </ul>''',
        'treatment': '''<ul class="text-start mb-0 ps-3 text-muted small">
            <li class="mb-2"><strong>Pharmacological Management:</strong> Medical management frequently includes Cholinesterase inhibitors (e.g., Donepezil, Rivastigmine, Galantamine) to help manage cognitive symptoms by boosting acetylcholine levels in the brain.</li>
            <li class="mb-2"><strong>Occupational Therapy:</strong> Engage occupational therapists to establish structured, predictable daily routines and maintain independence.</li>
            <li class="mb-2"><strong>Home Safety:</strong> Implement vital safety modifications at home, such as removing tripping hazards, installing grab bars, and securing medications.</li>
            <li><strong>Support Systems:</strong> It is highly recommended that family members join caregiver support groups and seek psychological counseling to prepare for the progressive nature of the disease.</li>
        </ul>'''
    },
    'Moderate Demented': {
        'color': '#e74c3c',
        'badge_class': 'danger',
        'severity': 'Middle-Stage Alzheimer\'s',
        'cause': '''<ul class="text-start mb-0 ps-3 text-muted small">
            <li class="mb-2"><strong>Widespread Cortical Shrinkage:</strong> Extensive brain damage leading to significant overall cortical shrinkage and markedly enlarged ventricles (fluid-filled spaces in the brain).</li>
            <li class="mb-2"><strong>Severe Network Disruption:</strong> Plaques and tangles have spread extensively throughout multiple brain lobes, severely disrupting communication between neurons.</li>
            <li><strong>Pronounced Clinical Symptoms:</strong> This structural damage results in pronounced cognitive decline affecting language, logic, and emotional regulation. Patients may experience personality shifts, confusion regarding time and place, and require assistance with daily self-care tasks.</li>
        </ul>''',
        'treatment': '''<ul class="text-start mb-0 ps-3 text-muted small">
            <li class="mb-2"><strong>Advanced Medication:</strong> Treatment usually involves NMDA receptor antagonists like Memantine (Namenda), often utilized in combination with Cholinesterase inhibitors to manage moderate-to-severe symptoms.</li>
            <li class="mb-2"><strong>Behavioral Management:</strong> Focus shifts heavily toward behavioral and psychological management, attempting to reduce anxiety, agitation, and sleep disturbances without over-relying on sedatives.</li>
            <li class="mb-2"><strong>Full-Time Supervision:</strong> Patients will likely require comprehensive, full-time supervision to assist with Activities of Daily Living (ADLs) such as dressing, bathing, and eating.</li>
            <li><strong>Specialized Memory Care:</strong> Families should evaluate long-term care options, including adult day care centers, in-home specialized nursing services, or residential memory care communities designed for dementia patients.</li>
        </ul>'''
    }
}


def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    """Decorator to require authentication for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ─── Model Cache (avoids reloading on every prediction) ───────

_model_cache = {}

def get_cached_model(model_path):
    """Load a model from disk only once, then serve from cache."""
    if model_path not in _model_cache:
        print(f"  [Cache] Loading model: {os.path.basename(model_path)} ...")
        _model_cache[model_path] = tf.keras.models.load_model(model_path)
        # Warmup: run a dummy prediction to pre-compile the graph
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _model_cache[model_path](dummy, training=False)
        print(f"  [Cache] Model loaded, cached, and warmed up.")
    return _model_cache[model_path]


# ─── Routes ───────────────────────────────────────────────────

@app.route('/')
@app.route('/home')
def home():
    """Home page with hero section and project statistics."""
    stats = {
        'classes': 4,
        'dataset_size': '86,437',
        'models_trained': 1,
        'best_accuracy': '95%+'
    }
    return render_template('home.html', stats=stats)


@app.route('/about')
def about():
    """About page with disease info, dataset, and technology details."""
    return render_template('about.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration with form validation."""
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        errors = []
        if not full_name or len(full_name) < 2:
            errors.append('Full name must be at least 2 characters.')
        if not email or '@' not in email:
            errors.append('Please enter a valid email address.')

        # Password strength validation
        import re
        if len(password) < 8:
            errors.append('Password must be at least 8 characters.')
        if not re.search(r'[A-Za-z]', password):
            errors.append('Password must contain at least one letter.')
        if not re.search(r'[0-9]', password):
            errors.append('Password must contain at least one number.')
        if not re.search(r'[^A-Za-z0-9]', password):
            errors.append('Password must contain at least one special character (e.g. @, #, $, !).')

        if password != confirm_password:
            errors.append('Passwords do not match.')

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('register.html')

        # Hash password and create user
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        if add_user(full_name, email, password_hash):
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Email already registered. Please use a different email or log in.', 'danger')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login with session management."""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        user = get_user_by_email(email)
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['full_name']
            session['user_email'] = user['email']
            flash(f'Welcome back, {user["full_name"]}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Clear session and log out user."""
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))


@app.route('/methodology')
def methodology():
    """ML pipeline methodology with visual explanations."""
    # Check which plot files exist
    plot_files = {}
    if os.path.exists(PLOTS_DIR):
        for f in os.listdir(PLOTS_DIR):
            if f.endswith('.png'):
                plot_files[f] = url_for('static', filename=f'../plots/{f}')

    return render_template('methodology.html', plot_files=plot_files)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """MRI scan upload and prediction page."""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'mri_image' not in request.files:
            flash('No file uploaded. Please select an MRI image.', 'danger')
            return redirect(url_for('predict'))

        file = request.files['mri_image']
        if file.filename == '':
            flash('No file selected. Please choose an MRI image.', 'danger')
            return redirect(url_for('predict'))

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a JPG or PNG image.', 'danger')
            return redirect(url_for('predict'))

        # Get selected model
        model_key = request.form.get('model_select', 'mobilenet')
        model_filename = f'{model_key}_model.h5'
        model_path = os.path.join(MODELS_DIR, model_filename)

        # Check if model file exists
        if not os.path.exists(model_path):
            flash(f'Model file "{model_filename}" not found. Please ensure models are trained and placed in the models/ directory.', 'danger')
            return redirect(url_for('predict'))

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Load model (cached) and predict — direct call is ~10x faster than .predict()
            model = get_cached_model(model_path)
            predictions = model(img_array, training=False).numpy()
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx]) * 100
            predicted_class = CLASS_NAMES[predicted_class_idx]

            # Get clinical info
            info = CLASS_INFO.get(predicted_class, CLASS_INFO['Non Demented'])

            # Log prediction
            add_prediction(
                user_id=session['user_id'],
                image_filename=filename,
                model_used=AVAILABLE_MODELS.get(model_key, model_key),
                predicted_class=predicted_class,
                confidence=confidence
            )

            result = {
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'severity': info['severity'],
                'cause': info['cause'],
                'treatment': info['treatment'],
                'color': info['color'],
                'badge_class': info['badge_class'],
                'model_used': AVAILABLE_MODELS.get(model_key, model_key),
                'image_url': url_for('static', filename=f'uploads/{filename}'),
                'image_path': filepath,
                'all_probabilities': {
                    CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
                    for i in range(len(CLASS_NAMES))
                }
            }

            # Store result in session for PDF download
            session['last_result'] = {
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'severity': result['severity'],
                'cause': result['cause'],
                'treatment': result['treatment'],
                'model_used': result['model_used'],
                'image_path': filepath,
                'all_probabilities': result['all_probabilities'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'patient_name': session.get('user_name', 'N/A'),
            }

            return render_template('result.html', result=result)

        except Exception as e:
            flash(f'Error during prediction: {str(e)}', 'danger')
            return redirect(url_for('predict'))

    return render_template('predict.html', models=AVAILABLE_MODELS)


# ─── PDF Report Download ──────────────────────────────────────

def strip_html(html_text):
    """Strip HTML tags and convert to plain text bullet points."""
    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', '', html_text)
    # Replace HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Replace unicode chars that Helvetica doesn't support
    text = text.replace('\u2013', '-').replace('\u2014', '-')   # en/em dash
    text = text.replace('\u2018', "'").replace('\u2019', "'")   # curly quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2022', '-')                          # bullet
    text = text.replace('\u2026', '...')                        # ellipsis
    # Remove any remaining non-latin1 characters
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    # Clean whitespace
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines


@app.route('/download_report')
@login_required
def download_report():
    """Generate and download a PDF report for the last prediction."""
    result = session.get('last_result')
    if not result:
        flash('No prediction result found. Please analyze an MRI scan first.', 'warning')
        return redirect(url_for('predict'))

    try:
        if not FPDF_AVAILABLE:
            flash('PDF generation library (fpdf2) is not installed. Run: pip install fpdf2', 'danger')
            return redirect(url_for('predict'))

        class ReportPDF(FPDF):
            """Custom PDF with premium medical report styling."""

            # Color palette
            PRIMARY = (88, 60, 210)       # Deep purple
            PRIMARY_LIGHT = (167, 139, 250)  # Light purple
            ACCENT = (45, 198, 140)       # Teal green
            DARK = (30, 30, 45)           # Near-black
            TEXT = (55, 55, 70)           # Body text
            TEXT_LIGHT = (120, 120, 135)  # Muted text
            BG_LIGHT = (245, 245, 252)   # Light background
            WHITE = (255, 255, 255)
            BORDER = (220, 220, 230)

            SEVERITY_COLORS = {
                'Non Demented': (46, 204, 113),
                'Very Mild Demented': (243, 156, 18),
                'Mild Demented': (230, 126, 34),
                'Moderate Demented': (231, 76, 60),
            }

            def header(self):
                # Purple header banner
                self.set_fill_color(*self.PRIMARY)
                self.rect(0, 0, 210, 18, 'F')
                # Accent stripe
                self.set_fill_color(*self.ACCENT)
                self.rect(0, 18, 210, 1.5, 'F')

                # Header text on banner
                self.set_y(4)
                self.set_font('Helvetica', 'B', 11)
                self.set_text_color(*self.WHITE)
                self.cell(0, 5, 'NeuroScan AI', 0, 0, 'L')
                self.set_font('Helvetica', '', 9)
                self.cell(0, 5, 'Alzheimer\'s Disease Detection Report', 0, 1, 'R')

                self.set_y(10)
                self.set_font('Helvetica', '', 7)
                self.set_text_color(200, 200, 220)
                self.cell(0, 5, 'AI-Powered Medical Imaging Analysis', 0, 1, 'L')

                self.set_y(24)

            def footer(self):
                self.set_y(-18)
                # Footer line
                self.set_draw_color(*self.BORDER)
                self.set_line_width(0.3)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(2)
                self.set_font('Helvetica', '', 7)
                self.set_text_color(*self.TEXT_LIGHT)
                self.cell(95, 5, f'Report generated: {result["timestamp"]}', 0, 0, 'L')
                self.cell(95, 5, f'Page {self.page_no()}/{{nb}}', 0, 1, 'R')
                self.set_font('Helvetica', 'I', 6)
                self.cell(0, 4, 'This report is for educational purposes only and does not constitute medical advice.', 0, 0, 'C')

            def colored_section(self, title, icon_char='>', r=None, g=None, b=None):
                """Section header with colored left accent bar."""
                color = (r, g, b) if r is not None else self.PRIMARY
                # Left accent bar
                self.set_fill_color(*color)
                y_start = self.get_y()
                self.rect(10, y_start, 2.5, 9, 'F')
                # Title text
                self.set_x(16)
                self.set_font('Helvetica', 'B', 12)
                self.set_text_color(*self.DARK)
                self.cell(0, 9, f'  {title}', 0, 1, 'L')
                self.ln(2)

            def info_row(self, label, value, bold_value=False):
                """Key-value row with label and value."""
                self.set_x(18)
                self.set_font('Helvetica', '', 9)
                self.set_text_color(*self.TEXT_LIGHT)
                self.cell(45, 6, label, 0, 0)
                self.set_text_color(*self.DARK)
                self.set_font('Helvetica', 'B' if bold_value else '', 9)
                self.cell(0, 6, str(value), 0, 1)

            def detail_block(self, text):
                """Bullet point with bold heading and description."""
                self.set_x(18)
                self.set_font('Helvetica', '', 8.5)
                self.set_text_color(*self.TEXT)
                parts = text.split(':', 1)
                if len(parts) == 2 and len(parts[0]) < 60:
                    # Bullet marker
                    self.set_fill_color(*self.PRIMARY_LIGHT)
                    self.rect(self.get_x(), self.get_y() + 1.5, 2, 2, 'F')
                    self.set_x(23)
                    self.set_font('Helvetica', 'B', 8.5)
                    self.set_text_color(*self.DARK)
                    self.cell(0, 5, parts[0].strip(), 0, 1)
                    self.set_x(23)
                    self.set_font('Helvetica', '', 8.5)
                    self.set_text_color(*self.TEXT)
                    self.multi_cell(170, 4.5, parts[1].strip())
                else:
                    self.set_fill_color(*self.PRIMARY_LIGHT)
                    self.rect(self.get_x(), self.get_y() + 1.5, 2, 2, 'F')
                    self.set_x(23)
                    self.multi_cell(170, 4.5, text)
                self.ln(2)

        pdf = ReportPDF()
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=22)
        pdf.add_page()

        # ── Report Title ──
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(*ReportPDF.DARK)
        pdf.cell(0, 10, 'Diagnostic Analysis Report', 0, 1, 'C')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(*ReportPDF.TEXT_LIGHT)
        pdf.cell(0, 6, 'MRI-Based Alzheimer\'s Disease Classification', 0, 1, 'C')
        pdf.ln(6)

        # ── Diagnosis Result Card (prominent box at top) ──
        sev_color = ReportPDF.SEVERITY_COLORS.get(result['predicted_class'], (100, 100, 100))
        card_y = pdf.get_y()
        # Card background
        pdf.set_fill_color(*ReportPDF.BG_LIGHT)
        pdf.set_draw_color(*ReportPDF.BORDER)
        pdf.set_line_width(0.3)
        pdf.rect(10, card_y, 190, 32, 'DF')
        # Severity color stripe on left
        pdf.set_fill_color(*sev_color)
        pdf.rect(10, card_y, 3, 32, 'F')

        # Diagnosis text inside card
        pdf.set_xy(18, card_y + 3)
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(*ReportPDF.TEXT_LIGHT)
        pdf.cell(80, 5, 'DIAGNOSIS', 0, 0, 'L')
        pdf.cell(50, 5, 'SEVERITY', 0, 0, 'L')
        pdf.cell(0, 5, 'CONFIDENCE', 0, 1, 'L')

        pdf.set_x(18)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(*sev_color)
        pdf.cell(80, 8, result['predicted_class'], 0, 0, 'L')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*ReportPDF.DARK)
        pdf.cell(50, 8, result['severity'], 0, 0, 'L')
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(*sev_color)
        pdf.cell(0, 8, f"{result['confidence']}%", 0, 1, 'L')

        # Confidence bar
        pdf.set_x(18)
        bar_y = pdf.get_y() + 2
        bar_width = 175
        # Background bar
        pdf.set_fill_color(220, 220, 230)
        pdf.rect(18, bar_y, bar_width, 4, 'F')
        # Filled bar
        pdf.set_fill_color(*sev_color)
        fill_width = bar_width * (result['confidence'] / 100)
        pdf.rect(18, bar_y, fill_width, 4, 'F')

        pdf.set_y(card_y + 36)

        # ── Patient Information ──
        pdf.colored_section('Patient Information')
        pdf.info_row('Patient Name', result.get('patient_name', 'N/A'))
        pdf.info_row('Date & Time', result['timestamp'])
        pdf.info_row('AI Model Used', result['model_used'])
        pdf.ln(4)

        # ── MRI Scan Image ──
        image_path = result.get('image_path', '')
        if image_path and os.path.exists(image_path):
            pdf.colored_section('Uploaded MRI Scan')
            # Image with border
            img_x = 60
            img_w = 85
            img_y = pdf.get_y()
            pdf.set_draw_color(*ReportPDF.BORDER)
            pdf.set_line_width(0.5)
            pdf.image(image_path, x=img_x + 1, w=img_w - 2)
            img_end_y = pdf.get_y()
            pdf.rect(img_x, img_y, img_w, img_end_y - img_y + 1)
            pdf.ln(6)

        # ── Class Probabilities ──
        pdf.colored_section('Classification Probabilities')
        probs = result['all_probabilities']
        max_prob = max(probs.values()) if probs else 1

        for cls, prob in probs.items():
            cls_color = ReportPDF.SEVERITY_COLORS.get(cls, (100, 100, 100))
            is_predicted = (cls == result['predicted_class'])

            pdf.set_x(18)
            # Class name
            pdf.set_font('Helvetica', 'B' if is_predicted else '', 9)
            pdf.set_text_color(*ReportPDF.DARK)
            pdf.cell(50, 6, cls, 0, 0, 'L')

            # Probability bar
            bar_x = pdf.get_x()
            bar_y = pdf.get_y() + 1
            bar_max_w = 100
            # Background
            pdf.set_fill_color(235, 235, 245)
            pdf.rect(bar_x, bar_y, bar_max_w, 4, 'F')
            # Fill
            pdf.set_fill_color(*cls_color)
            fill_w = bar_max_w * (prob / 100) if prob > 0 else 0.5
            pdf.rect(bar_x, bar_y, fill_w, 4, 'F')

            # Percentage text
            pdf.set_x(bar_x + bar_max_w + 3)
            pdf.set_font('Helvetica', 'B' if is_predicted else '', 9)
            pdf.set_text_color(*cls_color)
            pdf.cell(20, 6, f'{prob}%', 0, 1, 'R')
            pdf.ln(1)
        pdf.ln(3)

        # ── Probable Cause ──
        pdf.colored_section('Probable Cause', r=230, g=126, b=34)
        cause_lines = strip_html(result.get('cause', ''))
        for line in cause_lines:
            if line:
                pdf.detail_block(line)
        pdf.ln(2)

        # ── Treatment Suggestions ──
        pdf.colored_section('Treatment Recommendations', r=46, g=204, b=113)
        treatment_lines = strip_html(result.get('treatment', ''))
        for line in treatment_lines:
            if line:
                pdf.detail_block(line)
        pdf.ln(4)

        # ── Disclaimer Box ──
        disc_y = pdf.get_y()
        # Check if enough space, else add page
        if disc_y > 255:
            pdf.add_page()
            disc_y = pdf.get_y()

        pdf.set_fill_color(255, 248, 235)
        pdf.set_draw_color(243, 186, 80)
        pdf.set_line_width(0.4)
        pdf.rect(10, disc_y, 190, 28, 'DF')
        # Warning stripe
        pdf.set_fill_color(243, 186, 80)
        pdf.rect(10, disc_y, 3, 28, 'F')

        pdf.set_xy(16, disc_y + 2)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(180, 120, 20)
        pdf.cell(0, 5, 'IMPORTANT DISCLAIMER', 0, 1)
        pdf.set_x(16)
        pdf.set_font('Helvetica', '', 7.5)
        pdf.set_text_color(130, 100, 50)
        pdf.multi_cell(180, 4,
            'This report is generated by NeuroScan AI, an artificial intelligence system, for educational and '
            'informational purposes only. It does NOT constitute a medical diagnosis and should NOT be used as '
            'a substitute for professional medical advice. Please consult a qualified neurologist or healthcare '
            'professional for proper diagnosis and treatment. The developers assume no liability for decisions '
            'made based on this report.'
        )

        # ── Generate PDF bytes ──
        pdf_bytes = pdf.output()
        buffer = io.BytesIO(pdf_bytes)
        buffer.seek(0)

        safe_class = result['predicted_class'].replace(' ', '_')
        filename = f"NeuroScan_Report_{safe_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )

    except ImportError:
        flash('PDF generation library (fpdf2) is not installed. Run: pip install fpdf2', 'danger')
        return redirect(url_for('predict'))
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f'Error generating PDF: {str(e)}', 'danger')
        return redirect(url_for('predict'))


# ─── Error Handlers ───────────────────────────────────────────

@app.errorhandler(404)
def page_not_found(e):
    return render_template('base.html', error='Page not found'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('base.html', error='Internal server error'), 500


# ─── App Initialization ───────────────────────────────────────

def initialize_app():
    """Initialize directories and database (called on startup)."""
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    init_db()


# Always initialize on import (works for both gunicorn and direct run)
initialize_app()


# ─── Main ─────────────────────────────────────────────────────

if __name__ == '__main__':
    # Run the application — model loads lazily on first prediction via get_cached_model()
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print("=" * 60)
    print("  Alzheimer's Disease Detection System")
    print(f"  Running on http://localhost:{port}")
    print("=" * 60)
    app.run(debug=debug, port=port, host='0.0.0.0')
