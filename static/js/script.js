/**
 * NeuroScan AI — JavaScript
 * Handles animations, form validation, file upload preview, and counters.
 */

document.addEventListener('DOMContentLoaded', () => {
    initScrollAnimations();
    initNavbarScroll();
    initStatCounters();
    initPasswordToggle();
    initPasswordStrength();
    initRegisterValidation();
    initFileUpload();
    initModelSelection();
    initPredictForm();
});

/* ─── Scroll Animations ─────────────────────────────────────── */
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const delay = entry.target.getAttribute('data-delay') || 0;
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, parseInt(delay));
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    document.querySelectorAll('[data-animate]').forEach(el => observer.observe(el));
}

/* ─── Navbar Scroll Effect ──────────────────────────────────── */
function initNavbarScroll() {
    const navbar = document.getElementById('mainNavbar');
    if (!navbar) return;

    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
}

/* ─── Stat Counter Animation ────────────────────────────────── */
function initStatCounters() {
    const counters = document.querySelectorAll('.stat-number[data-count]');
    if (!counters.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(counter => observer.observe(counter));
}

function animateCounter(el) {
    const target = parseInt(el.getAttribute('data-count'));
    const suffix = el.getAttribute('data-suffix') || '';
    const duration = 2000;
    const start = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = Math.floor(eased * target);

        if (target > 1000) {
            el.textContent = current.toLocaleString() + suffix;
        } else {
            el.textContent = current + suffix;
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

/* ─── Password Toggle ───────────────────────────────────────── */
function initPasswordToggle() {
    document.querySelectorAll('.toggle-password').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-target');
            const input = document.getElementById(targetId);
            if (!input) return;

            const icon = btn.querySelector('i');
            if (input.type === 'password') {
                input.type = 'text';
                icon.classList.replace('bi-eye', 'bi-eye-slash');
            } else {
                input.type = 'password';
                icon.classList.replace('bi-eye-slash', 'bi-eye');
            }
        });
    });
}

/* ─── Password Strength ─────────────────────────────────────── */
function initPasswordStrength() {
    const passwordInput = document.getElementById('password');
    const strengthContainer = document.getElementById('passwordStrength');
    const reqContainer = document.getElementById('passwordRequirements');
    if (!passwordInput || !strengthContainer) return;

    passwordInput.addEventListener('input', () => {
        const pw = passwordInput.value;

        // Rule checks
        const hasLength  = pw.length >= 8;
        const hasLetter  = /[A-Za-z]/.test(pw);
        const hasNumber  = /[0-9]/.test(pw);
        const hasSpecial = /[^A-Za-z0-9]/.test(pw);

        // Update requirement checklist UI
        if (reqContainer) {
            reqContainer.style.display = pw.length > 0 ? 'block' : 'none';
            updateReqItem('reqLength',  hasLength);
            updateReqItem('reqLetter',  hasLetter);
            updateReqItem('reqNumber',  hasNumber);
            updateReqItem('reqSpecial', hasSpecial);
        }

        // Strength score (0-4)
        let strength = [hasLength, hasLetter, hasNumber, hasSpecial]
            .filter(Boolean).length;

        const colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71'];
        const widths = ['25%', '50%', '75%', '100%'];
        const labels = ['Weak', 'Fair', 'Good', 'Strong'];

        if (pw.length === 0) {
            strengthContainer.innerHTML = '';
        } else {
            const idx = Math.max(strength - 1, 0);
            strengthContainer.innerHTML =
                `<div class="strength-bar" style="width: ${widths[idx]}; background: ${colors[idx]};"></div>` +
                `<small class="d-block mt-1" style="color: ${colors[idx]}; font-size: 0.72rem; font-weight: 600;">${labels[idx]}</small>`;
        }
    });
}

/** Update a single requirement checklist item (green check or red cross). */
function updateReqItem(id, passed) {
    const el = document.getElementById(id);
    if (!el) return;
    const icon = el.querySelector('i');
    if (passed) {
        icon.className = 'bi bi-check-circle-fill text-success me-1';
    } else {
        icon.className = 'bi bi-x-circle-fill text-danger me-1';
    }
}

/* ─── Register Form Validation ──────────────────────────────── */
function initRegisterValidation() {
    const form = document.getElementById('registerForm');
    if (!form) return;

    form.addEventListener('submit', (e) => {
        const password = document.getElementById('password');
        const confirmPassword = document.getElementById('confirm_password');
        const errorDiv = document.getElementById('passwordMatchError');

        // Password requirements check
        if (password) {
            const pw = password.value;
            const valid = pw.length >= 8
                && /[A-Za-z]/.test(pw)
                && /[0-9]/.test(pw)
                && /[^A-Za-z0-9]/.test(pw);

            if (!valid) {
                e.preventDefault();
                password.classList.add('is-invalid');
                // Trigger the input event so the checklist shows what's missing
                password.dispatchEvent(new Event('input'));
                return;
            } else {
                password.classList.remove('is-invalid');
            }
        }

        // Password match check
        if (password && confirmPassword && password.value !== confirmPassword.value) {
            e.preventDefault();
            confirmPassword.classList.add('is-invalid');
            if (errorDiv) errorDiv.style.display = 'block';
        }
    });

    const confirmPassword = document.getElementById('confirm_password');
    if (confirmPassword) {
        confirmPassword.addEventListener('input', () => {
            const password = document.getElementById('password');
            const errorDiv = document.getElementById('passwordMatchError');
            if (password && confirmPassword.value === password.value) {
                confirmPassword.classList.remove('is-invalid');
                if (errorDiv) errorDiv.style.display = 'none';
            }
        });
    }
}

/* ─── File Upload Preview ───────────────────────────────────── */
function initFileUpload() {
    const fileInput = document.getElementById('mri_image');
    const uploadZone = document.getElementById('uploadZone');
    const uploadContent = document.getElementById('uploadContent');
    const uploadPreview = document.getElementById('uploadPreview');
    const previewImage = document.getElementById('previewImage');
    const removeBtn = document.getElementById('removePreview');

    if (!fileInput || !uploadZone) return;

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) showPreview(file);
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            showPreview(file);
        }
    });

    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.value = '';
            uploadContent.style.display = 'block';
            uploadPreview.style.display = 'none';
        });
    }

    function showPreview(file) {
        if (!file.type.startsWith('image/')) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            uploadPreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

/* ─── Model Selection ───────────────────────────────────────── */
function initModelSelection() {
    const modelOptions = document.querySelectorAll('.model-option');
    if (!modelOptions.length) return;

    modelOptions.forEach(option => {
        option.addEventListener('click', () => {
            modelOptions.forEach(o => o.classList.remove('active'));
            option.classList.add('active');
            const radio = option.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
        });
    });
}

/* ─── Predict Form Submission ───────────────────────────────── */
function initPredictForm() {
    const form = document.getElementById('predictForm');
    const btn = document.getElementById('predictBtn');
    const spinner = document.getElementById('predictSpinner');

    if (!form || !btn) return;

    form.addEventListener('submit', () => {
        btn.disabled = true;
        btn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i>Analyzing...';
        if (spinner) spinner.classList.remove('d-none');
    });
}
