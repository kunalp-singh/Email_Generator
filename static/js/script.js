// Complete fixed JavaScript for the Post Office Email Campaign Generator

let currentMailData = null;

const campaignForm = document.getElementById('campaignForm');
const mailboxOverlay = document.getElementById('mailboxOverlay');
const mailContent = document.getElementById('mailContent');
const closeMailboxBtn = document.getElementById('closeMailbox');
const postalActions = document.getElementById('postalActions');

let btnExportCsv, btnGenerateAudio, btnCopy, btnPrint;
let lastFocusedElement = null;

/* Bind action buttons once DOM is ready */
document.addEventListener('DOMContentLoaded', () => {
    btnExportCsv = document.getElementById('btnExportCsv');
    btnGenerateAudio = document.getElementById('btnGenerateAudio');
    btnCopy = document.getElementById('btnCopy');
    btnPrint = document.getElementById('btnPrint');

    if (btnExportCsv) btnExportCsv.addEventListener('click', exportPostalCSV);
    if (btnGenerateAudio) btnGenerateAudio.addEventListener('click', generateMailAudio);
    if (btnCopy) btnCopy.addEventListener('click', copyMailToClipboard);
    if (btnPrint) btnPrint.addEventListener('click', printMailCampaign);
});

/* Submit handler: single source of truth for generating campaigns */
campaignForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    showMailbox();
    showMailDelivery();
    // Hide buttons during loading
    if (postalActions) {
        postalActions.style.display = 'none';
    }

    const formData = {
        accounts: [{
            account_name: document.getElementById('accountName').value,
            industry: document.getElementById('industry').value,
            pain_points: document.getElementById('painPoints').value.split(',').map(p => p.trim()).filter(Boolean),
            contacts: [{
                name: document.getElementById('contactName').value,
                email: document.getElementById('contactEmail').value,
                job_title: document.getElementById('jobTitle').value
            }],
            campaign_objective: document.getElementById('objective').value,
            tone: document.getElementById('tone').value,
            language: document.getElementById('language').value,
            interest: document.getElementById('interest').value
        }],
        number_of_emails: parseInt(document.getElementById('numberOfEmails').value, 10)
    };

    try {
        // For demonstration purposes, we'll simulate the API response
        // Replace this with actual API call when backend is available
        const data = await simulateAPIResponse(formData);
        
        currentMailData = { formData, data };
        displayMailResults(data);
        
        // Force show buttons with multiple approaches
        console.log('Attempting to show buttons...', postalActions);
        if (postalActions) {
            postalActions.style.display = 'flex';
            postalActions.style.visibility = 'visible';
            postalActions.style.opacity = '1';
            postalActions.removeAttribute('hidden');
            
            // Additional debugging
            console.log('Button display after setting:', getComputedStyle(postalActions).display);
        }
    } catch (error) {
        showMailError('📪 Delivery Error: ' + error.message);
        // Keep buttons hidden on error
        if (postalActions) {
            postalActions.style.display = 'none';
        }
    }
});

/* Simulate API response for testing - replace with actual API call */
async function simulateAPIResponse(formData) {
    // Simulate loading time
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const numberOfEmails = formData.number_of_emails;
    const emails = [];
    
    for (let i = 0; i < numberOfEmails; i++) {
        emails.push({
            variants: [{
                subject: `Unlock Growth Opportunities for ${formData.accounts[0].account_name} - Email ${i + 1}`,
                body: `Dear ${formData.accounts[0].contacts[0].name},\n\nI hope this email finds you well. As someone working in the ${formData.accounts[0].industry} industry, I understand the unique challenges you face, particularly around ${formData.accounts[0].pain_points[0] || 'operational efficiency'}.\n\nOur solutions are specifically designed to address these pain points and help companies like ${formData.accounts[0].account_name} achieve their goals in ${formData.accounts[0].interest}.\n\nWould you be interested in a brief conversation to explore how we can support your initiatives?\n\nBest regards,\nYour Campaign Team`,
                call_to_action: "Schedule a 15-minute discovery call",
                suggested_send_time: i === 0 ? "Tuesday 10:00 AM" : `${['Wednesday', 'Friday', 'Monday', 'Thursday'][i % 4]} ${['9:00', '2:00', '11:00', '3:00'][i % 4]} ${'AM PM'.split(' ')[i % 2]}`,
                sub_variants: [
                    `Transform ${formData.accounts[0].account_name}'s ${formData.accounts[0].interest} Strategy`,
                    `${formData.accounts[0].contacts[0].name}, Let's Discuss Your ${formData.accounts[0].industry} Goals`,
                    `Quick Question About ${formData.accounts[0].account_name}'s Growth Plans`
                ]
            }]
        });
    }
    
    return {
        campaigns: [{
            emails: emails
        }]
    };
}

/* Modal controls */
closeMailboxBtn.addEventListener('click', hideMailbox);
mailboxOverlay.addEventListener('click', (e) => {
    if (e.target === mailboxOverlay) hideMailbox();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && mailboxOverlay.classList.contains('show')) {
        hideMailbox();
    }
});

function showMailbox() {
    mailboxOverlay.classList.add('show');
    mailboxOverlay.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    lastFocusedElement = document.activeElement;
    closeMailboxBtn.focus();
}

function hideMailbox() {
    mailboxOverlay.classList.remove('show');
    mailboxOverlay.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    if (lastFocusedElement && typeof lastFocusedElement.focus === 'function') {
        lastFocusedElement.focus();
    }
}

/* UI states */
function showMailDelivery() {
    mailContent.innerHTML = `
        <div class="loading-mail">
            <div class="mail-spinner"></div>
            <div class="loading-text">📮 Processing Your Mail Campaign...</div>
            <div class="loading-subtext">Our postal workers are crafting personalized letters</div>
        </div>
    `;
}

function showMailError(message) {
    mailContent.innerHTML = `
        <div class="error-delivery">
            <span class="error-icon">❌</span>
            ${escapeHtml(message)}
        </div>
    `;
}

/* Render results - FIXED: Corrected optional chaining syntax */
function displayMailResults(data) {
    let html = '';
    
    if (data.campaigns && Array.isArray(data.campaigns)) {
        data.campaigns.forEach(campaign => {
            if (campaign.emails && Array.isArray(campaign.emails)) {
                campaign.emails.forEach((email, index) => {
                    // Fixed: Correct optional chaining for accessing first variant
                    const variant = email.variants && email.variants[0] ? email.variants[0] : {};
                    
                    const subVariants = variant.sub_variants && Array.isArray(variant.sub_variants) && variant.sub_variants.length > 0
                        ? `
                          <div class="mail-field">
                            <span class="mail-label">Alternative Subject Lines:</span>
                            <div class="mail-content-text">${variant.sub_variants.map(escapeHtml).join(' • ')}</div>
                          </div>
                        `
                        : '';

                    html += `
                        <div class="envelope">
                            <div class="letter">
                                <div class="mail-header">✉️ Letter ${index + 1}</div>

                                <div class="mail-field">
                                    <span class="mail-label">Subject Line:</span>
                                    <div class="mail-content-text">${escapeHtml(variant.subject || '')}</div>
                                </div>

                                <div class="mail-field">
                                    <span class="mail-label">Message Body:</span>
                                    <div class="mail-content-text">${escapeHtml(variant.body || '').replace(/\n/g, '<br>')}</div>
                                </div>

                                <div class="mail-field">
                                    <span class="mail-label">Call to Action:</span>
                                    <div class="mail-content-text">${escapeHtml(variant.call_to_action || '')}</div>
                                </div>

                                <div class="mail-field">
                                    <span class="mail-label">Recommended Delivery Time:</span>
                                    <div class="mail-content-text">⏰ ${escapeHtml(variant.suggested_send_time || '')}</div>
                                </div>

                                ${subVariants}
                            </div>
                        </div>
                    `;
                });
            }
        });
    }

    mailContent.innerHTML = html || '<div class="error-delivery">No mail content generated</div>';
}

/* Actions */
async function exportPostalCSV() {
    if (!currentMailData) {
        showMailNotification('📪 No campaign data available for export');
        return;
    }

    try {
        // Create CSV content from the current mail data
        const csvContent = generateCSVContent(currentMailData);
        
        // Create and download the CSV file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `postal_campaign_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        setTimeout(() => {
            URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }, 100);
        
        showMailNotification('📦 Postal records exported successfully!');
    } catch (error) {
        showMailNotification('📪 Export Error: ' + error.message);
    }
}

function generateCSVContent(mailData) {
    const headers = ['Email Number', 'Subject', 'Body', 'Call to Action', 'Send Time', 'Alternative Subjects'];
    const rows = [headers.join(',')];
    
    if (mailData.data.campaigns && mailData.data.campaigns[0] && mailData.data.campaigns[0].emails) {
        mailData.data.campaigns[0].emails.forEach((email, index) => {
            const variant = email.variants && email.variants[0] ? email.variants[0] : {};
            const row = [
                index + 1,
                `"${(variant.subject || '').replace(/"/g, '""')}"`,
                `"${(variant.body || '').replace(/"/g, '""').replace(/\n/g, ' ')}"`,
                `"${(variant.call_to_action || '').replace(/"/g, '""')}"`,
                `"${(variant.suggested_send_time || '').replace(/"/g, '""')}"`,
                `"${variant.sub_variants ? variant.sub_variants.join('; ').replace(/"/g, '""') : ''}"`
            ];
            rows.push(row.join(','));
        });
    }
    
    return rows.join('\n');
}

async function generateMailAudio() {
    if (!currentMailData) {
        showMailNotification('📪 No campaign data available for audio generation');
        return;
    }

    try {
        // Get the first email content for audio generation
        const firstEmail = currentMailData.data?.campaigns?.[0]?.emails?.[0];
        const emailBody = firstEmail?.variants?.[0]?.body || '';
        
        if (!emailBody) {
            showMailNotification('📪 No email content available for audio generation');
            return;
        }

        // Use Web Speech API for text-to-speech
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(emailBody);
            utterance.rate = 0.8;
            utterance.pitch = 1;
            utterance.volume = 0.8;
            
            // Try to use a more natural voice if available
            const voices = speechSynthesis.getVoices();
            const englishVoice = voices.find(voice => 
                voice.lang.startsWith('en') && voice.name.includes('Natural')
            ) || voices.find(voice => voice.lang.startsWith('en'));
            
            if (englishVoice) {
                utterance.voice = englishVoice;
            }

            speechSynthesis.speak(utterance);
            showMailNotification('🎧 Voice Mail is now playing');
        } else {
            showMailNotification('🎵 Audio generation not supported in this browser');
        }
    } catch (error) {
        showMailNotification('🎵 Audio Error: ' + error.message);
    }
}

function copyMailToClipboard() {
    if (!mailContent.textContent) {
        showMailNotification('📪 No content available to copy');
        return;
    }

    // Extract text content from the mail display
    let textContent = '';
    const letters = mailContent.querySelectorAll('.letter');
    
    letters.forEach((letter, index) => {
        textContent += `\n=== EMAIL ${index + 1} ===\n\n`;
        
        const fields = letter.querySelectorAll('.mail-field');
        fields.forEach(field => {
            const label = field.querySelector('.mail-label')?.textContent || '';
            const content = field.querySelector('.mail-content-text')?.textContent || '';
            if (label && content) {
                textContent += `${label}\n${content}\n\n`;
            }
        });
    });

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(textContent.trim()).then(
            () => showMailNotification('📋 Mail content copied to clipboard'),
            (err) => showMailNotification('📪 Copy failed: ' + err.message)
        );
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = textContent.trim();
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            showMailNotification('📋 Mail content copied to clipboard');
        } catch (err) {
            showMailNotification('📪 Copy failed: ' + err.message);
        } finally {
            document.body.removeChild(textArea);
        }
    }
}

function printMailCampaign() {
    if (!mailContent.innerHTML) {
        showMailNotification('📪 No content available to print');
        return;
    }

    const printWindow = window.open('', '_blank');
    if (!printWindow) {
        showMailNotification('📪 Print blocked by browser. Please allow popups.');
        return;
    }

    const printHTML = `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Email Campaign - Print Version</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Crimson+Text:wght@400;600&display=swap');
                
                body {
                    font-family: 'Crimson Text', serif;
                    color: #2F4F4F;
                    margin: 20px;
                    background: white;
                }
                
                .mailbox-title {
                    font-family: 'Playfair Display', serif;
                    font-size: 2rem;
                    font-weight: 700;
                    text-align: center;
                    margin-bottom: 30px;
                    color: #8B0000;
                }
                
                .envelope {
                    border: 2px solid #8B4513;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    page-break-inside: avoid;
                }
                
                .letter {
                    padding: 20px;
                }
                
                .mail-header {
                    font-family: 'Playfair Display', serif;
                    color: #8B0000;
                    font-size: 1.4rem;
                    font-weight: 700;
                    margin-bottom: 20px;
                    border-bottom: 2px solid #D2691E;
                    padding-bottom: 10px;
                }
                
                .mail-field {
                    margin-bottom: 15px;
                }
                
                .mail-label {
                    color: #8B4513;
                    font-weight: 600;
                    font-size: 0.9rem;
                    display: block;
                    margin-bottom: 5px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                
                .mail-content-text {
                    color: #2F4F4F;
                    line-height: 1.6;
                    font-size: 1rem;
                    background: #FFFEF7;
                    padding: 10px;
                    border-radius: 5px;
                    border-left: 4px solid #D2691E;
                }
                
                @media print {
                    body { margin: 0; }
                    .envelope { 
                        break-inside: avoid;
                        margin-bottom: 40px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="mailbox-title">📬 EMAIL CAMPAIGN</div>
            ${mailContent.innerHTML}
            <script>
                window.onload = function() {
                    setTimeout(function() {
                        window.focus();
                        window.print();
                    }, 500);
                };
            </script>
        </body>
        </html>
    `;

    printWindow.document.write(printHTML);
    printWindow.document.close();
    showMailNotification('🖨️ Print dialog opened');
}

function showMailNotification(message) {
    // Remove existing notifications
    const existingToasts = document.querySelectorAll('.notification-toast');
    existingToasts.forEach(toast => toast.remove());

    const toast = document.createElement('div');
    toast.className = 'notification-toast';
    toast.setAttribute('role', 'status');
    toast.setAttribute('aria-live', 'polite');
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 3000);
}

/* Utility: HTML escape for safe insertion */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}