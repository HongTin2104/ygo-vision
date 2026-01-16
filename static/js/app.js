// Yu-Gi-Oh! Card Recognition - Main JavaScript
// PERSISTENT MODE: Keep detected card until replaced by better one
// SIMPLIFIED VERSION - Always update display

class CardRecognitionApp {
    constructor() {
        this.videoFeed = document.getElementById('videoFeed');
        this.captureBtn = document.getElementById('captureBtn');
        this.searchInput = document.getElementById('searchInput');
        this.searchBtn = document.getElementById('searchBtn');
        this.cardInfoContainer = document.getElementById('cardInfo');
        this.detectedCardInfo = document.getElementById('detectedCardInfo');
        this.detectionCard = document.getElementById('detectionCard');
        this.detectionStatus = document.getElementById('detectionStatus');
        this.alertContainer = document.getElementById('alertContainer');

        this.cardsScanned = 0;
        this.searchCount = 0;

        // PERSISTENT MODE - Keep card until replaced by better one
        this.currentCard = null; // Currently displayed card name
        this.currentConfidence = 0; // Current card's confidence

        this.init();
    }

    init() {
        // Set video feed source
        this.videoFeed.src = '/video_feed';

        // Event listeners
        this.captureBtn.addEventListener('click', () => this.captureCard());
        this.searchBtn.addEventListener('click', () => this.searchCard());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.searchCard();
        });

        // Load stats
        this.loadStats();

        // Create animated background
        this.createParticles();

        // Start real-time card detection
        this.startRealtimeDetection();

        console.log('✅ App initialized - Persistent mode active');
    }

    startRealtimeDetection() {
        // FASTER POLLING: 200ms for quick detection
        setInterval(() => this.checkForCard(), 200);
    }

    async checkForCard() {
        try {
            const response = await fetch('/current_card');
            const data = await response.json();

            if (data.detected && data.card_info && data.card_info.name) {
                const cardName = data.card_info.name;
                const confidence = data.card_info.confidence || 0.95;

                console.log(`📥 Detected: ${cardName} (${(confidence * 100).toFixed(1)}%)`);

                // PERSISTENT MODE: Only update if:
                // 1. No card currently displayed, OR
                // 2. Different card with HIGHER confidence, OR
                // 3. Same card with HIGHER confidence
                const shouldUpdate =
                    !this.currentCard || // No card yet
                    (cardName !== this.currentCard && confidence > this.currentConfidence) || // Different card, better confidence
                    (cardName === this.currentCard && confidence > this.currentConfidence); // Same card, better confidence

                if (shouldUpdate) {
                    console.log(`✅ UPDATE: ${cardName} (${(confidence * 100).toFixed(1)}%)`);
                    this.currentCard = cardName;
                    this.currentConfidence = confidence;

                    this.updateDetectionStatus('detected', cardName, confidence);
                    this.displayDetectedCard(data.card_info);

                    // Update counter if it's a truly new card
                    const isNewCard = this.currentCard !== cardName;
                    if (isNewCard) {
                        this.cardsScanned++;
                        this.updateSessionStats();
                    }
                } else {
                    console.log(`⏸️  KEEP: ${this.currentCard} (${(this.currentConfidence * 100).toFixed(1)}%) - New: ${cardName} (${(confidence * 100).toFixed(1)}%)`);
                    this.updateDetectionStatus('holding', this.currentCard, this.currentConfidence);
                }
            } else {
                // No detection - but KEEP showing current card
                if (this.currentCard) {
                    this.updateDetectionStatus('holding', this.currentCard, this.currentConfidence);
                } else {
                    this.updateDetectionStatus('waiting');
                }
            }
        } catch (error) {
            console.error('❌ Error:', error);
            // Keep showing current card even on error
            if (this.currentCard) {
                this.updateDetectionStatus('holding', this.currentCard, this.currentConfidence);
            } else {
                this.updateDetectionStatus('error');
            }
        }
    }

    updateDetectionStatus(status, cardName = '', confidence = 0) {
        const indicator = this.detectionStatus.querySelector('.status-indicator');
        const text = this.detectionStatus.querySelector('.status-text');

        indicator.className = 'status-indicator';

        if (status === 'detected') {
            indicator.classList.add('detected');
            const displayName = cardName.substring(0, 25);
            const confidencePercent = (confidence * 100).toFixed(1);
            text.textContent = `${displayName}${cardName.length > 25 ? '...' : ''} (${confidencePercent}%)`;
        } else if (status === 'holding') {
            indicator.classList.add('detected');
            const displayName = cardName.substring(0, 25);
            const confidencePercent = (confidence * 100).toFixed(1);
            text.textContent = `${displayName}${cardName.length > 25 ? '...' : ''} (${confidencePercent}%) 📌`;
        } else if (status === 'waiting') {
            text.textContent = 'Waiting for card...';
        } else if (status === 'error') {
            text.textContent = 'Connection error';
        }
    }

    displayDetectedCard(cardInfo) {
        console.log('🎨 Displaying card:', cardInfo.name);

        this.detectionCard.classList.add('active');

        const confidence = cardInfo.confidence || 0.95;
        const confidencePercent = (confidence * 100).toFixed(1);

        this.detectedCardInfo.innerHTML = `
            <div class="card-display">
                ${cardInfo.image_url ? `
                    <div style="margin-bottom: 1rem; text-align: center;">
                        <img src="${cardInfo.image_url}" 
                             alt="${cardInfo.name}" 
                             style="max-width: 100%; height: auto; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"
                             onerror="this.style.display='none'">
                    </div>
                ` : ''}
                
                <div class="card-name">${cardInfo.name || 'Unknown Card'}</div>
                ${cardInfo.type ? `<div class="card-type">${cardInfo.type}</div>` : ''}
                
                <div class="confidence-bar">
                    <div class="confidence-label">
                        <span>Confidence</span>
                        <span>${confidencePercent}%</span>
                    </div>
                    <div class="confidence-progress">
                        <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                    </div>
                </div>
                
                ${cardInfo.desc ? `
                    <div class="card-description">
                        ${cardInfo.desc.substring(0, 200)}${cardInfo.desc.length > 200 ? '...' : ''}
                    </div>
                ` : ''}
                
                <div style="margin-top: 1rem; display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.875rem;">
                    ${cardInfo.atk !== null && cardInfo.atk !== undefined ? `
                        <div style="padding: 0.5rem; background: rgba(15, 23, 42, 0.4); border-radius: 0.5rem; text-align: center;">
                            <div style="color: var(--text-secondary); font-size: 0.75rem;">ATK</div>
                            <div style="color: var(--primary-color); font-weight: 700;">${cardInfo.atk}</div>
                        </div>
                    ` : ''}
                    ${cardInfo.def !== null && cardInfo.def !== undefined ? `
                        <div style="padding: 0.5rem; background: rgba(15, 23, 42, 0.4); border-radius: 0.5rem; text-align: center;">
                            <div style="color: var(--text-secondary); font-size: 0.75rem;">DEF</div>
                            <div style="color: var(--secondary-color); font-weight: 700;">${cardInfo.def}</div>
                        </div>
                    ` : ''}
                    ${cardInfo.level ? `
                        <div style="padding: 0.5rem; background: rgba(15, 23, 42, 0.4); border-radius: 0.5rem; text-align: center;">
                            <div style="color: var(--text-secondary); font-size: 0.75rem;">Level</div>
                            <div style="color: var(--accent-color); font-weight: 700;">${cardInfo.level}</div>
                        </div>
                    ` : ''}
                    ${cardInfo.attribute ? `
                        <div style="padding: 0.5rem; background: rgba(15, 23, 42, 0.4); border-radius: 0.5rem; text-align: center;">
                            <div style="color: var(--text-secondary); font-size: 0.75rem;">Attribute</div>
                            <div style="color: var(--success); font-weight: 700;">${cardInfo.attribute}</div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;

        console.log('✅ Card displayed successfully');
    }

    clearDetectedCard() {
        this.detectionCard.classList.remove('active');
        this.detectedCardInfo.innerHTML = `
            <div class="no-detection">
                <div class="no-detection-icon">🎴</div>
                <p>No card detected</p>
                <small>Place a Yu-Gi-Oh! card in the frame</small>
            </div>
        `;
    }

    updateSessionStats() {
        document.getElementById('cardsScanned').textContent = this.cardsScanned;
        document.getElementById('searchCount').textContent = this.searchCount;
    }

    createParticles() {
        const particleCount = 20;
        const body = document.body;

        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 20 + 's';
            particle.style.animationDuration = (15 + Math.random() * 10) + 's';
            body.appendChild(particle);
        }
    }

    async loadStats() {
        try {
            const response = await fetch('/stats');
            const data = await response.json();

            document.getElementById('totalCards').textContent =
                data.total_cards.toLocaleString();
            document.getElementById('dbStatus').textContent =
                data.database_loaded ? 'Loaded' : 'Not Loaded';
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    async captureCard() {
        this.showAlert('Saving current card...', 'info');
        this.captureBtn.disabled = true;

        try {
            const response = await fetch('/capture', {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok) {
                this.showAlert(data.message, 'success');
            } else {
                this.showAlert(data.error || 'Failed to save card', 'error');
            }
        } catch (error) {
            this.showAlert('Error: ' + error.message, 'error');
        } finally {
            this.captureBtn.disabled = false;
        }
    }

    async searchCard() {
        const query = this.searchInput.value.trim();

        if (!query) {
            this.showAlert('Please enter a card name', 'error');
            return;
        }

        this.showAlert('Searching for card...', 'info');
        this.searchCount++;
        this.updateSessionStats();

        try {
            const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();

            if (response.ok) {
                this.showAlert('Card found!', 'success');
                this.displayCardInfo(data);
            } else {
                this.showAlert(data.error || 'Card not found', 'error');
                this.cardInfoContainer.innerHTML = `
                    <div class="no-info">
                        <div class="no-info-icon">📋</div>
                        <p>Card not found</p>
                    </div>
                `;
            }
        } catch (error) {
            this.showAlert('Error: ' + error.message, 'error');
        }
    }

    displayCardInfo(cardData) {
        const fields = [
            { key: 'type', label: 'Type' },
            { key: 'desc', label: 'Description' },
            { key: 'atk', label: 'ATK' },
            { key: 'def', label: 'DEF' },
            { key: 'level', label: 'Level' },
            { key: 'race', label: 'Race' },
            { key: 'attribute', label: 'Attribute' },
            { key: 'archetype', label: 'Archetype' }
        ];

        let detailsHtml = '<div class="card-details">';

        fields.forEach(field => {
            const value = cardData[field.key];
            if (value !== null && value !== undefined && value !== '') {
                detailsHtml += `
                    <div class="detail-row">
                        <span class="detail-label">${field.label}:</span>
                        <span class="detail-value">${this.formatValue(value)}</span>
                    </div>
                `;
            }
        });

        detailsHtml += '</div>';

        this.cardInfoContainer.innerHTML = `
            <div class="card-info">
                <h3 style="margin-bottom: 1rem; font-size: 1.3rem; color: var(--primary-color);">
                    ${cardData.name || 'Unknown Card'}
                </h3>
                ${detailsHtml}
            </div>
        `;
    }

    formatValue(value) {
        if (typeof value === 'string' && value.length > 150) {
            return value.substring(0, 150) + '...';
        }
        return value;
    }

    showAlert(message, type = 'info') {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;

        const icon = type === 'success' ? '✓' :
            type === 'error' ? '✗' : 'ℹ';

        alert.innerHTML = `
            <span style="font-size: 1.2rem;">${icon}</span>
            <span>${message}</span>
        `;

        this.alertContainer.innerHTML = '';
        this.alertContainer.appendChild(alert);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Starting Yu-Gi-Oh! Card Recognition App...');
    new CardRecognitionApp();
});
