// Yu-Gi-Oh! Card Recognition - Main JavaScript
// PERSISTENT MODE: Keep detected card until replaced by better one
// SIMPLIFIED VERSION - Always update display

class CardRecognitionApp {
    constructor() {
        this.videoFeed = document.getElementById('videoFeed');
        this.searchInput = document.getElementById('searchInput');
        this.searchBtn = document.getElementById('searchBtn');
        this.cardInfoContainer = document.getElementById('cardInfo');
        this.detectedCardInfo = document.getElementById('detectedCardInfo');
        this.detectionCard = document.getElementById('detectionCard');
        this.detectionStatus = document.getElementById('detectionStatus');
        this.alertContainer = document.getElementById('alertContainer');
        this.cornerCardImage = document.getElementById('cornerCardImage');
        this.priceContainer = document.getElementById('priceContainer');

        this.cardsScanned = 0;
        this.searchCount = 0;

        // PERSISTENT MODE - Keep card until replaced by better one
        this.currentCard = null; // Currently displayed card name
        this.currentConfidence = 0; // Current card's confidence
        this.currentCardImageUrl = null; // Current card's image URL
        this.currentCardDetails = null;

        this.init();
    }

    init() {
        // Set video feed source
        this.videoFeed.src = 'video_feed';

        // Event listeners
        this.searchBtn.addEventListener('click', () => this.searchCard());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.searchCard();
        });

        // Tab switching
        this.initTabSwitching();

        // Create animated background
        this.createParticles();

        // Load falling card images
        this.loadFallingCardImages();

        // Start real-time card detection
        this.initModal();
        this.startRealtimeDetection();

        console.log('App initialized - Persistent mode active');
    }

    initTabSwitching() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-tab');

                // Remove active class from all buttons and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked button and corresponding content
                button.classList.add('active');
                document.getElementById(`${targetTab}-tab`).classList.add('active');

                console.log(`Switched to ${targetTab} tab`);
            });
        });
    }

    initModal() {
        this.cardModal = document.getElementById('cardModal');
        this.modalImage = document.getElementById('modalCardImage');
        this.modalPriceContent = document.getElementById('modalPriceContent');
        this.modalStatsContent = document.getElementById('modalStatsContent');
        this.modalDescContent = document.getElementById('modalDescContent');
        this.closeModalBtn = document.querySelector('.close-modal');

        if (!this.cardModal || !this.modalImage || !this.closeModalBtn || !this.modalPriceContent || !this.modalStatsContent || !this.modalDescContent) return;

        // Close events
        this.closeModalBtn.addEventListener('click', () => this.closeModal());
        this.cardModal.addEventListener('click', (e) => {
            if (e.target === this.cardModal || e.target.classList.contains('modal-content-grid')) {
                this.closeModal();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.cardModal.style.display === 'flex') {
                this.closeModal();
            }
        });

        this.detectedCardInfo.addEventListener('click', (e) => {
            if (e.target.classList.contains('clickable-card')) {
                // Use stored card details
                this.openModal(e.target.src, this.currentCardDetails);
            }
        });

        // 3D Tilt Effect
        this.modalImage.addEventListener('mousemove', (e) => this.handleCardTilt(e));
        this.modalImage.addEventListener('mouseleave', () => this.resetCardTilt());
    }

    openModal(imageUrl, cardData) {
        if (!imageUrl) return;

        this.modalImage.src = imageUrl;
        this.cardModal.style.display = 'flex';

        // Populate Stats Info
        const hasStats = cardData && (
            (cardData.atk !== null && cardData.atk !== undefined) ||
            (cardData.def !== null && cardData.def !== undefined) ||
            cardData.level ||
            cardData.attribute
        );

        if (hasStats) {
            this.modalStatsContent.innerHTML = `
                <h2>Card Stats</h2>
                ${this.generateStatsHtml(cardData)}
             `;
            this.modalStatsContent.style.display = 'block';
        } else {
            this.modalStatsContent.style.display = 'none';
        }

        // Populate Price Info
        if (cardData && cardData.prices) {
            this.modalPriceContent.innerHTML = `
                <h2>Market Prices</h2>
                ${this.formatPriceDetails(cardData.prices, !hasStats)}
            `;
        } else {
            this.modalPriceContent.innerHTML = `
                <h2>Card Info</h2>
                <p style="color: var(--text-secondary);">No price information available for this card.</p>
            `;
        }

        // Populate Description Info
        if (cardData) {
            this.modalDescContent.innerHTML = `
                <h2>Card Lore</h2>
                <div class="modal-card-name">${cardData.name || 'Unknown Name'}</div>
                ${cardData.type ? `<div class="modal-card-type">[ ${cardData.type} ]</div>` : ''}
                <div class="modal-card-desc-text">
                    ${cardData.desc ? cardData.desc.replace(/\n/g, '<br>') : 'No description available.'}
                </div>
            `;
            this.modalDescContent.parentNode.style.display = 'block';
        } else {
            this.modalDescContent.parentNode.style.display = 'none';
        }

        // Add Active class for info animation after short delay
        setTimeout(() => {
            this.cardModal.classList.add('active');
        }, 100);

        // Reset and trigger image animation
        this.modalImage.classList.remove('fly-in');
        void this.modalImage.offsetWidth; // Force reflow
        this.modalImage.classList.add('fly-in');

        // Lock body scroll
        document.body.classList.add('noscroll');
    }

    handleCardTilt(e) {
        const el = this.modalImage;
        // Stop animation if user interacts
        if (el.classList.contains('fly-in')) {
            el.classList.remove('fly-in');
        }

        const rect = el.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        // Calculate rotation (max 15 degrees)
        // "Face the mouse" effect
        const rotateX = ((y - centerY) / centerY) * -15;
        const rotateY = ((x - centerX) / centerX) * 15;

        // Apply transform with perspective
        // scale3d(1.05, ...) gives a slight pop when tilting
        el.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.05, 1.05, 1.05)`;
    }

    resetCardTilt() {
        this.modalImage.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale3d(1, 1, 1)';
    }

    closeModal() {
        this.cardModal.classList.remove('active');
        // Delay hiding to allow transitions if we had them, or just hide immediately
        this.cardModal.style.display = 'none';
        // Unlock body scroll
        document.body.classList.remove('noscroll');
        this.resetCardTilt();
        setTimeout(() => {
            this.modalImage.src = '';
            this.modalImage.classList.remove('fly-in');
            this.modalPriceContent.innerHTML = ''; // Clear content
            this.modalStatsContent.innerHTML = '';
            this.modalDescContent.innerHTML = '';
        }, 100);
    }

    generateStatsHtml(cardInfo) {
        return `
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
                        <div style="color: var(--text-secondary); font-size: 0.75rem; margin-bottom: 0.25rem;">Attribute</div>
                        <img src="static/images/attribute/${cardInfo.attribute}.svg" 
                             alt="${cardInfo.attribute}" 
                             style="width: 24px; height: 24px; margin: 0 auto; display: block;"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                        <div style="color: var(--success); font-weight: 700; display: none;">${cardInfo.attribute}</div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    startRealtimeDetection() {
        // FASTER POLLING: 200ms for quick detection
        setInterval(() => this.checkForCard(), 200);
    }

    async checkForCard() {
        try {
            const response = await fetch('current_card');
            const data = await response.json();

            if (data.detected && data.card_info && data.card_info.name) {
                const cardName = data.card_info.name;
                const confidence = data.card_info.confidence || 0.95;

                // STRICT MODE: Require very high confidence (effectively 100%)
                const THRESHOLD = 0.99;

                // Only proceed if confidence is high enough
                if (confidence < THRESHOLD) {
                    // Treat as waiting/holding if not sure enough
                    if (this.currentCard) {
                        this.updateDetectionStatus('holding', this.currentCard, this.currentConfidence);
                    } else {
                        this.updateDetectionStatus('waiting');
                    }
                    return;
                }

                console.log(`Detected: ${cardName} (${(confidence * 100).toFixed(1)}%)`);

                // PERSISTENT MODE UPDATE LOGIC:
                const shouldUpdate =
                    !this.currentCard || // No card yet
                    (cardName !== this.currentCard) || // New card detected (must meet >99% threshold)
                    (cardName === this.currentCard && confidence > this.currentConfidence); // Same card, better confidence

                if (shouldUpdate) {
                    console.log(`UPDATE: ${cardName} (${(confidence * 100).toFixed(1)}%)`);
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
                    console.log(`KEEP: ${this.currentCard} (${(this.currentConfidence * 100).toFixed(1)}%) - New: ${cardName} (${(confidence * 100).toFixed(1)}%)`);
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
            console.error('Error:', error);
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
            text.textContent = `${displayName}${cardName.length > 25 ? '...' : ''} (${confidencePercent}%)`;
        } else if (status === 'waiting') {
            text.textContent = 'Waiting for card...';
        } else if (status === 'error') {
            text.textContent = 'Connection error';
        }
    }

    displayDetectedCard(cardInfo) {
        console.log('Displaying card:', cardInfo.name);
        this.currentCardDetails = cardInfo; // Store for modal usage

        this.detectionCard.classList.add('active');

        const confidence = cardInfo.confidence || 0.95;
        const confidencePercent = (confidence * 100).toFixed(1);

        // Update corner card image
        if (cardInfo.image_url) {
            this.currentCardImageUrl = cardInfo.image_url;
            this.updateCornerCardImage(cardInfo.image_url);
        }

        this.detectedCardInfo.innerHTML = `
            <div class="card-display">
                ${cardInfo.image_url ? `
                    <div style="margin-bottom: 1rem; text-align: center;">
                        <img src="${cardInfo.image_url}" 
                             alt="${cardInfo.name}" 
                             class="clickable-card"
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
                            <div style="color: var(--text-secondary); font-size: 0.75rem; margin-bottom: 0.25rem;">Attribute</div>
                            <img src="static/images/attribute/${cardInfo.attribute}.svg" 
                                 alt="${cardInfo.attribute}" 
                                 style="width: 24px; height: 24px; margin: 0 auto; display: block;"
                                 onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                            <div style="color: var(--success); font-weight: 700; display: none;">${cardInfo.attribute}</div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;

        // Render Price Info into Separate Container
        if (cardInfo.prices) {
            this.priceContainer.innerHTML = `
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" style="color: #4caf50;">$</span>
                        Market Prices
                    </h2>
                </div>
                <div style="padding: 1.5rem;">
                    ${this.formatPriceDetails(cardInfo.prices)}
                </div>
            `;
            this.priceContainer.style.display = 'block';
        } else {
            this.priceContainer.style.display = 'none';
        }

        console.log('Card displayed successfully');
    }

    formatPriceDetails(priceData, isExpanded = false) {
        if (!priceData) return '';

        // Handle backward compatibility or different structure
        const market = priceData.market_prices || priceData;
        const sets = priceData.sets || [];

        // Sort sets by price (High to Low)
        const sortedSets = [...sets].sort((a, b) => parseFloat(b.set_price) - parseFloat(a.set_price));

        const listMaxHeight = isExpanded ? '450px' : '200px';

        let setsHtml = '';
        if (sortedSets.length > 0) {
            setsHtml = `
                <div style="margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 0.5rem;">
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                        <span>Versions & Printings</span>
                        <span style="font-size: 0.7rem; opacity: 0.7;">${sortedSets.length} found</span>
                    </div>
                    <div style="max-height: ${listMaxHeight}; overflow-y: auto; padding-right: 5px;" class="custom-scrollbar">
                        ${sortedSets.map(set => {
                const price = parseFloat(set.set_price);
                const priceDisplay = (price > 0) ? `$${set.set_price}` : '<span style="opacity: 0.5; font-size: 0.7rem;">N/A</span>';

                return `
                            <div style="display: flex; justify-content: space-between; align-items: start; padding: 0.5rem 0; font-size: 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.05);">
                                <div style="flex: 1; margin-right: 0.5rem;">
                                    <div style="color: #fff; font-weight: 500; font-family: monospace;">${set.set_code}</div>
                                    <div style="color: var(--text-secondary); font-size: 0.7rem; line-height: 1.2;">
                                        ${set.set_rarity} <span style="opacity: 0.5;">(${set.set_rarity_code})</span>
                                        <br>
                                        <span style="opacity: 0.7;">${set.set_name}</span>
                                    </div>
                                </div>
                                <div style="color: #4caf50; font-weight: bold; white-space: nowrap;">${priceDisplay}</div>
                            </div>
                        `;
            }).join('')}
                    </div>
                </div>
            `;
        }

        return `
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Average Market Price</div>
                <div style="display: flex; justify-content: space-between; gap: 0.5rem;">
                    <div style="flex: 1; padding: 0.5rem; background: rgba(0, 200, 83, 0.1); border: 1px solid rgba(0, 200, 83, 0.3); border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 0.7rem; color: var(--text-secondary);">TCGPlayer</div>
                        <div style="font-weight: bold; color: #4caf50;">$${market.tcgplayer_price || 'N/A'}</div>
                    </div>
                    <div style="flex: 1; padding: 0.5rem; background: rgba(33, 150, 243, 0.1); border: 1px solid rgba(33, 150, 243, 0.3); border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 0.7rem; color: var(--text-secondary);">CardMarket</div>
                        <div style="font-weight: bold; color: #2196f3;">€${market.cardmarket_price || 'N/A'}</div>
                    </div>
                </div>
                ${setsHtml}
            </div>
        `;
    }

    updateCornerCardImage(imageUrl) {
        if (this.cornerCardImage && imageUrl) {
            this.cornerCardImage.src = imageUrl;
            console.log('Updated corner card image:', imageUrl);
        }
    }

    resetCornerCardImage() {
        if (this.cornerCardImage) {
            this.cornerCardImage.src = 'static/images/Back-EN.webp';
            console.log('Reset corner card to back image');
        }
    }

    clearDetectedCard() {
        this.detectionCard.classList.remove('active');
        this.detectedCardInfo.innerHTML = `
            <div class="no-detection">
                <img src="static/images/Back-EN.webp" 
                     alt="Card Back" 
                     style="width: 120px; height: auto; margin-bottom: 1rem; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <p>No card detected</p>
                <small>Place a Yu-Gi-Oh! card in the frame</small>
            </div>
        `;
        // Reset corner card image to back
        this.resetCornerCardImage();
        this.currentCardImageUrl = null;
        this.priceContainer.style.display = 'none';
    }

    updateSessionStats() {
        document.getElementById('cardsScanned').textContent = this.cardsScanned || 0;
        document.getElementById('searchCount').textContent = this.searchCount || 0;
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

    async loadFallingCardImages() {
        try {
            // Fetch random card images from API
            const response = await fetch('random_card_images?count=20');
            const data = await response.json();

            if (data.images && data.images.length > 0) {
                // Get all falling card elements
                const fallingCards = document.querySelectorAll('.falling-card');

                // Apply random card images to each falling card
                fallingCards.forEach((card, index) => {
                    if (index < data.images.length) {
                        const imageId = data.images[index];
                        card.style.backgroundImage = `url('card_image/${imageId}')`;
                    }
                });

                console.log('Loaded', data.images.length, 'falling card images');
            }
        } catch (error) {
            console.error('Failed to load falling card images:', error);
            // Fallback to default background if API fails
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
            const response = await fetch(`search?q=${encodeURIComponent(query)}`);
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
                ${this.formatPriceDetails(cardData.prices)}
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

        alert.innerHTML = `
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
    console.log('Starting Yu-Gi-Oh! Card Recognition App...');
    new CardRecognitionApp();
});
