/**
 * Voice Sentiment Analysis - Frontend JavaScript
 */

/** Initialize drag-and-drop upload page */
function initUploadPage() {
    const dropZone = document.getElementById('dropZone');
    const audioInput = document.getElementById('audioInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewSection = document.getElementById('previewSection');
    const audioPreview = document.getElementById('audioPreview');
    const fileName = document.getElementById('fileName');
    const submitBtn = document.getElementById('submitBtn');

    if (!dropZone || !audioInput) return;

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        audioInput.click();
    });

    dropZone.addEventListener('click', () => audioInput.click());

    ['dragenter', 'dragover'].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            audioInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    audioInput.addEventListener('change', () => {
        if (audioInput.files.length > 0) {
            handleFileSelect(audioInput.files[0]);
        }
    });

    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', () => {
            submitBtn.disabled = true;
            const text = document.getElementById('submitText');
            const spinner = document.getElementById('submitSpinner');
            if (text) text.classList.add('d-none');
            if (spinner) spinner.classList.remove('d-none');
        });
    }

    function handleFileSelect(file) {
        const allowed = ['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/mp4', 'audio/x-wav', ''];
        const ext = file.name.split('.').pop().toLowerCase();
        const validExt = ['wav', 'mp3', 'ogg', 'flac', 'm4a'];

        if (!validExt.includes(ext)) {
            alert('Invalid file type. Please upload WAV, MP3, OGG, or FLAC.');
            return;
        }

        fileName.textContent = file.name + ' (' + formatFileSize(file.size) + ')';
        audioPreview.src = URL.createObjectURL(file);
        previewSection.classList.remove('d-none');
        submitBtn.disabled = false;
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/** Render horizontal bar chart for emotion probabilities */
function renderProbabilityChart(canvasId, probabilities) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !probabilities) return;

    const labels = Object.keys(probabilities).map((e) => e.charAt(0).toUpperCase() + e.slice(1));
    const values = Object.values(probabilities);
    const colors = values.map((v) => {
        if (v > 30) return 'rgba(13, 110, 253, 0.8)';
        if (v > 15) return 'rgba(102, 16, 242, 0.6)';
        return 'rgba(108, 117, 125, 0.5)';
    });

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: values,
                backgroundColor: colors,
                borderRadius: 6,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: '%' },
                },
            },
        },
    });
}

/** Render dashboard pie/doughnut charts */
function renderDashboardCharts(sentimentData, emotionData) {
    const sentimentCanvas = document.getElementById('sentimentChart');
    const emotionCanvas = document.getElementById('emotionChart');

    if (sentimentCanvas && sentimentData) {
        const sLabels = Object.keys(sentimentData);
        const sValues = Object.values(sentimentData);
        const sColors = sLabels.map((l) => {
            if (l === 'Positive') return '#28a745';
            if (l === 'Neutral') return '#ffc107';
            return '#dc3545';
        });

        new Chart(sentimentCanvas, {
            type: 'doughnut',
            data: {
                labels: sLabels,
                datasets: [{
                    data: sValues.length ? sValues : [1],
                    backgroundColor: sColors.length ? sColors : ['#dee2e6'],
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'bottom' } },
            },
        });
    }

    if (emotionCanvas && emotionData) {
        const eLabels = Object.keys(emotionData).map((e) => e.charAt(0).toUpperCase() + e.slice(1));
        const eValues = Object.values(emotionData);

        new Chart(emotionCanvas, {
            type: 'bar',
            data: {
                labels: eLabels.length ? eLabels : ['No data'],
                datasets: [{
                    label: 'Count',
                    data: eValues.length ? eValues : [0],
                    backgroundColor: 'rgba(13, 110, 253, 0.7)',
                    borderRadius: 6,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
            },
        });
    }
}
