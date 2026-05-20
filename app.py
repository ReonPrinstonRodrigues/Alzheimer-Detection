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
            """Custom PDF with header and footer."""
            def header(self):
                self.set_font('Helvetica', 'B', 10)
                self.set_text_color(100, 100, 100)
                self.cell(0, 8, 'NeuroScan AI  |  Alzheimer\'s Disease Detection Report', 0, 1, 'C')
                self.set_draw_color(167, 139, 250)
                self.set_line_width(0.5)
                self.line(10, 14, 200, 14)
                self.ln(6)

            def footer(self):
                self.set_y(-15)
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(150, 150, 150)
                self.cell(0, 10,
                    f'Generated on {result["timestamp"]}  |  Page {self.page_no()}/{{nb}}  |  For Educational Purposes Only',
                    0, 0, 'C')

            def section_title(self, title, r=60, g=60, b=60):
                self.set_font('Helvetica', 'B', 13)
                self.set_text_color(r, g, b)
                self.cell(0, 10, title, 0, 1, 'L')
                self.set_draw_color(200, 200, 200)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(3)

            def key_value(self, key, value):
                self.set_font('Helvetica', 'B', 10)
                self.set_text_color(80, 80, 80)
                self.cell(55, 7, key + ':', 0, 0)
                self.set_font('Helvetica', '', 10)
                self.set_text_color(40, 40, 40)
                self.cell(0, 7, str(value), 0, 1)

            def bullet_point(self, text):
                self.set_font('Helvetica', '', 9)
                self.set_text_color(60, 60, 60)
                # Split bold heading from description
                parts = text.split(':', 1)
                x = self.get_x()
                self.cell(5, 5, '-', 0, 0)  # bullet point
                if len(parts) == 2 and len(parts[0]) < 60:
                    self.set_font('Helvetica', 'B', 9)
                    self.cell(0, 5, parts[0].strip() + ':', 0, 1)
                    self.set_x(x + 5)
                    self.set_font('Helvetica', '', 9)
                    self.multi_cell(175, 5, parts[1].strip())
                else:
                    self.multi_cell(180, 5, text)
                self.ln(1)

        pdf = ReportPDF()
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # ── Title ──
        pdf.set_font('Helvetica', 'B', 22)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 12, 'Detection Report', 0, 1, 'C')
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 7, 'Alzheimer\'s Disease MRI Classification Result', 0, 1, 'C')
        pdf.ln(8)

        # ── Patient / Session Info ──
        pdf.section_title('Patient Information')
        pdf.key_value('Patient Name', result.get('patient_name', 'N/A'))
        pdf.key_value('Date & Time', result['timestamp'])
        pdf.key_value('Model Used', result['model_used'])
        pdf.ln(5)

        # ── MRI Image ──
        image_path = result.get('image_path', '')
        if image_path and os.path.exists(image_path):
            pdf.section_title('Uploaded MRI Scan')
            pdf.image(image_path, x=65, w=80)
            pdf.ln(5)

        # ── Diagnosis Result ──
        pdf.section_title('Diagnosis Result')
        pdf.key_value('Predicted Class', result['predicted_class'])
        pdf.key_value('Clinical Severity', result['severity'])
        pdf.key_value('Confidence Score', f"{result['confidence']}%")
        pdf.ln(3)

        # ── Class Probabilities ──
        pdf.section_title('Class Probabilities')
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(240, 240, 245)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(95, 7, '  Class', 1, 0, 'L', True)
        pdf.cell(95, 7, 'Probability (%)', 1, 1, 'C', True)
        pdf.set_font('Helvetica', '', 9)
        for cls, prob in result['all_probabilities'].items():
            pdf.cell(95, 7, '  ' + cls, 1, 0, 'L')
            pdf.cell(95, 7, f'{prob}%', 1, 1, 'C')
        pdf.ln(5)

        # ── Probable Cause ──
        pdf.section_title('Probable Cause')
        cause_lines = strip_html(result.get('cause', ''))
        for line in cause_lines:
            if line:
                pdf.bullet_point(line)
        pdf.ln(3)

        # ── Treatment Suggestions ──
        pdf.section_title('Treatment Suggestions')
        treatment_lines = strip_html(result.get('treatment', ''))
        for line in treatment_lines:
            if line:
                pdf.bullet_point(line)
        pdf.ln(5)

        # ── Disclaimer ──
        pdf.set_draw_color(243, 156, 18)
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(180, 120, 20)
        pdf.cell(0, 6, 'DISCLAIMER', 0, 1)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(0, 5,
            'This report is generated by an AI model (NeuroScan AI) for educational and informational purposes only. '
            'It is NOT a substitute for professional medical diagnosis. Please consult a qualified neurologist or '
            'healthcare professional for proper medical advice, diagnosis, and treatment. The developers of this '
            'system assume no liability for decisions made based on this report.'
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
    # Preload the default model at startup so first prediction is fast
    default_model = os.path.join(MODELS_DIR, 'mobilenet_model.h5')
    if os.path.exists(default_model):
        print("  Preloading MobileNetV2 model...")
        get_cached_model(default_model)
        print("  Model ready for instant predictions!")

    # Run the application
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print("=" * 60)
    print("  Alzheimer's Disease Detection System")
    print(f"  Running on http://localhost:{port}")
    print("=" * 60)
    app.run(debug=debug, port=port, host='0.0.0.0')
