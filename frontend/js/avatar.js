import { TalkingHead } from "talkinghead";
import { HeadTTS } from "headtts";

let head;
let headtts;
window.avatarReady = false;

document.addEventListener('DOMContentLoaded', async function() {
    const nodeAvatar = document.getElementById('avatar');
    const globalLoading = document.getElementById('global-loading');
    const globalLoadingText = document.getElementById('global-loading-text');
    
    head = new TalkingHead(nodeAvatar, {
        ttsEndpoint: "N/A", 
        lipsyncModules: [], 
        cameraView: "upper"
    });
    window.head = head;
    
    headtts = new HeadTTS({
        transformersModule: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0/dist/transformers.min.js",
        endpoints: [ "webgpu", "wasm" ], 
        languages: ["en-us"], 
        voices: ["af_bella"], 
        voiceURL: "./voices", 
        audioCtx: head.audioCtx, 
        trace: 0,
    });

    headtts.onmessage = (message) => {
        if (message.type === "audio") {
            try { 
                head.speakAudio(message.data); 
            } catch(e) { 
                console.error("Lỗi phát audio:", e); 
            }
        }
    };

    try {
        await Promise.all([
            head.showAvatar({
                url: './avatars/brunette.glb', 
                body: 'F',
                avatarMood: 'neutral',
            }, (ev) => {
                if (ev.lengthComputable && globalLoadingText) {
                    globalLoadingText.textContent = `Downloading 3D Avatar: ${Math.round(ev.loaded/ev.total * 100)}%`;
                }
            }),
            (async () => {
                if(globalLoadingText) globalLoadingText.textContent = "Downloading AI TTS Model (Kokoro)...";
                await headtts.connect();
                if(globalLoadingText) globalLoadingText.textContent = "Initializing TTS Engine (Compiling Shaders)...";
            })()
        ]);

        headtts.setup({ voice: "af_bella", language: "en-us", speed: 1, audioEncoding: "wav" });
        
        if (globalLoading) {
            globalLoading.style.opacity = '0';
            setTimeout(() => globalLoading.style.display = 'none', 500);
        }
        
        // Hide local overlay too if it exists
        const localOverlay = document.getElementById('loading-overlay');
        if (localOverlay) localOverlay.style.display = 'none';
        
        head.start();

        window.avatarReady = true;
        window.headtts = headtts;  
        console.log("✅ Avatar & TTS đã sẵn sàng!");
    } catch (error) {
        if (globalLoadingText) globalLoadingText.textContent = "Error initializing Avatar: " + error;
        console.error(error);
    }
});

window.speakText = async function(text) {
    if (!window.avatarReady) return;
    
    let cleanText = text.replace(/[\*\_\[\]]/g, "").trim();
    if (cleanText.length < 2) return; 

    if (window.head.audioCtx.state === 'suspended') {
        await window.head.audioCtx.resume();
    }
    
    console.log("🗣️ Đang đọc:", cleanText);
    headtts.synthesize({ input: cleanText });
}