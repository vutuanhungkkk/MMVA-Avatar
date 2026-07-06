const API_BASE = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/chat";

let ws;
let isGenerating = false;
let currentAssistantBubble = null;
let currentImageBase64 = null;
let currentDocument = null;

// --- DOM Elements ---
const chatHistory = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');
const btnSend = document.getElementById('btn-send');
const wsStatus = document.getElementById('ws-status');

const providerSelect = document.getElementById('provider-select');
const btnApply = document.getElementById('btn-apply');
const setupStatus = document.getElementById('setup-status');
const btnClear = document.getElementById('btn-clear');
const attachmentPreview = document.getElementById('attachment-preview');

// --- Sidebar Logic ---
btnApply.addEventListener('click', async () => {
    const provider = providerSelect.value;

    btnApply.disabled = true;
    setupStatus.textContent = "⏳ Initialising...";
    setupStatus.className = "status-msg";

    try {
        const res = await fetch(`${API_BASE}/api/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider })
        });
        
        if (res.ok) {
            setupStatus.textContent = "✅ Assistant is ready!";
            setupStatus.className = "status-msg success";
            
            // Auto-trigger a hardcoded greeting from the assistant (Bypass LLM)
            setTimeout(() => {
                const greeting = "Hi, I am your Health AI Assistant. How can I help you today?";
                createAssistantBubble();
                if (currentAssistantBubble) {
                    currentAssistantBubble.innerHTML = greeting;
                }
                
                if (window.speakText) {
                    window.speakText(greeting);
                }
            }, 500);
        } else {
            const err = await res.json();
            setupStatus.textContent = "❌ Error: " + err.error;
        }
    } catch(e) {
        setupStatus.textContent = "❌ Failed to connect to server.";
    }
    btnApply.disabled = false;
});

btnClear.addEventListener('click', async () => {
    try {
        await fetch(`${API_BASE}/api/clear`, { method: 'POST' });
        chatHistory.innerHTML = '';
        currentImageBase64 = null;
        currentDocument = null;
        renderAttachmentPreview();
    } catch(e) {
        console.error("Failed to clear context:", e);
    }
});

function renderAttachmentPreview() {
    if (!attachmentPreview) return;
    if (currentImageBase64) {
        attachmentPreview.innerHTML =
            `<span class="attachment-chip">
                🖼️ Image attached
                <button class="chip-remove-btn" title="Remove image" onclick="removeImage()">✕</button>
            </span>`;
        return;
    }
    if (currentDocument) {
        attachmentPreview.innerHTML =
            `<span class="attachment-chip">
                📎 ${currentDocument.name}
                <button class="chip-remove-btn" title="Remove document" onclick="removeDocument('${currentDocument.name}')">✕</button>
            </span>`;
        return;
    }
    attachmentPreview.innerHTML = '';
}

function removeImage() {
    currentImageBase64 = null;
    renderAttachmentPreview();
}

async function removeDocument(filename) {
    // Optimistically clear from UI immediately
    const prevDoc = currentDocument;
    currentDocument = null;
    renderAttachmentPreview();

    try {
        const res = await fetch(`${API_BASE}/api/remove_document`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });
        if (!res.ok) {
            const err = await res.json();
            console.warn('Remove document error:', err.error);
            // Restore chip on failure
            currentDocument = prevDoc;
            renderAttachmentPreview();
        }
    } catch (e) {
        console.error('Failed to remove document:', e);
        // Restore chip on network error
        currentDocument = prevDoc;
        renderAttachmentPreview();
    }
}


function unlockAudio() {
    if (window.head && window.head.audioCtx && window.head.audioCtx.state === 'suspended') {
        window.head.audioCtx.resume();
    }
}


function connectWebSocket() {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => { wsStatus.textContent = "Connected"; wsStatus.style.color = "#4CAF50"; };
    ws.onclose = () => { wsStatus.textContent = "Disconnected"; wsStatus.style.color = "#F44336"; setTimeout(connectWebSocket, 3000); };
    
    let sentenceBuffer = "";
    
    ws.onmessage = async (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === "text_stream") {
                const chunk = data.content;
                if (currentAssistantBubble) {
                    currentAssistantBubble.innerHTML += chunk.replace(/\n/g, '<br>');
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }
                
                sentenceBuffer += chunk;
                
                if (/[.!?\n]/.test(chunk)) {
                    if (window.speakText) {
                        window.speakText(sentenceBuffer);
                    }
                    sentenceBuffer = ""; 
                }
            } 
            else if (data.type === "done") {
                isGenerating = false;
                if (sentenceBuffer.trim() && window.speakText) {
                    window.speakText(sentenceBuffer);
                }
                sentenceBuffer = "";
            }
        } catch (e) {
            if (event.data === "[DONE]") {
                isGenerating = false;
            } else {
                console.error("WebSocket error:", e, event.data);
            }
        }
    };
}
connectWebSocket();


function appendUserMessage(text, imgSrc = null) {
    const div = document.createElement('div');
    div.className = 'message user-message';
    let attachmentHtml = '';
    if (imgSrc) {
        attachmentHtml = `<img src="${imgSrc}" style="max-width:100%; border-radius:8px; margin-bottom:8px;"><br>`;
    }
    div.innerHTML = `${attachmentHtml}${text}`;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function createAssistantBubble() {
    const div = document.createElement('div');
    div.className = 'message assistant-message';
    chatHistory.appendChild(div);
    currentAssistantBubble = div;
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function sendMessage() {
    if (isGenerating || (!chatInput.value.trim() && !currentImageBase64)) return;
    
    const prompt = chatInput.value.trim();
    appendUserMessage(prompt || (currentImageBase64 ? "[Image Attached]" : "[Document Attached]"), currentImageBase64);
    createAssistantBubble();
    isGenerating = true;

    ws.send(JSON.stringify({ prompt, image_b64: currentImageBase64, document: currentDocument }));

    chatInput.value = "";
    currentImageBase64 = null;
    renderAttachmentPreview();
}

btnSend.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// --- Upload Logic (Image & PDF) ---
document.getElementById('image-upload').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = event => {
        currentImageBase64 = event.target.result;
        currentDocument = null;
        renderAttachmentPreview();
        alert("📷 Medical image attached! Describe what you need analyzed.");
    };
    reader.readAsDataURL(file);
});

document.getElementById('pdf-upload').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("pdf", file);
    
    alert("📋 Uploading medical document...");
    try {
        const res = await fetch(`${API_BASE}/api/process_pdf`, { method: "POST", body: formData });
        if(res.ok) {
            currentDocument = { name: file.name, type: file.type || file.name.split('.').pop() };
            currentImageBase64 = null;
            renderAttachmentPreview();
            alert("✅ Medical document indexed! You can now ask questions about it.");
        } else {
            const err = await res.json();
            alert("❌ Error: " + err.error);
        }
    } catch(err) { console.error(err); }
});

// --- Voice Record Logic (click to start / click again to stop) ---
let mediaRecorder, audioChunks = [], isRecording = false;
const btnRecord = document.getElementById('btn-record');

btnRecord.addEventListener('click', async () => {
    if (isRecording) {
        // --- Stop recording ---
        mediaRecorder?.stop();
        return;
    }

    // --- Start recording ---
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

        mediaRecorder.onstart = () => {
            isRecording = true;
            btnRecord.textContent = "⏹️ Click to Stop";
            btnRecord.style.background = "#EF4444";
            btnRecord.style.color = "white";
        };

        mediaRecorder.onstop = async () => {
            isRecording = false;
            // Stop all mic tracks so the browser releases the mic indicator
            stream.getTracks().forEach(t => t.stop());

            btnRecord.textContent = "⏳ Transcribing...";
            btnRecord.style.background = "#E8F0FE";
            btnRecord.style.color = "#1A1A2E";

            try {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const formData = new FormData();
                formData.append("audio", audioBlob, "voice.webm");

                const res = await fetch(`${API_BASE}/api/transcribe`, { method: "POST", body: formData });
                const data = await res.json();
                if (data.transcript) {
                    chatInput.value = data.transcript;
                    sendMessage();
                }
            } catch (e) {
                console.error("Transcription error:", e);
            }

            btnRecord.textContent = "🎙️ Click to Speak";
            btnRecord.style.background = "";
            btnRecord.style.color = "";
        };

        mediaRecorder.start();
    } catch (e) {
        console.error("Microphone access error:", e);
        alert("❌ Could not access microphone. Please check your browser permissions.");
    }
});