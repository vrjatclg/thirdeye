/**
 * THIRDEYE // Defense AI Frontend Application Controller
 * Handles synchronized dual-feed video comparison, interactive split sliders,
 * live model execution streaming, custom video uploads, and tactical HUD telemetry.
 */

document.addEventListener('DOMContentLoaded', () => {
    initSystemStatus();
    initClaheSplitSlider();
    initSyncedDemoPlayer();
    initTabs();
    initLiveRunner();
});

/* -------------------------------------------------------------------------
   1. System Status & Initialization
   ------------------------------------------------------------------------- */
async function initSystemStatus() {
    try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        
        const badge = document.getElementById('systemStatusBadge');
        if (badge && data.hardware) {
            badge.querySelector('.status-label').textContent = 
                `${data.hardware.device.toUpperCase()} // ${data.hardware.gpu_acceleration ? 'CUDA ONLINE' : 'CPU MODE'}`;
        }

        const specGpu = document.getElementById('specGpu');
        if (specGpu && data.hardware) {
            specGpu.textContent = `${data.hardware.device} (${data.hardware.vram_gb} GB VRAM)`;
        }
    } catch (e) {
        console.warn('Could not fetch system status:', e);
    }
}

/* -------------------------------------------------------------------------
   2. Interactive CLAHE Before / After Slider
   ------------------------------------------------------------------------- */
function initClaheSplitSlider() {
    const sliderInput = document.getElementById('splitSliderInput');
    const splitAfter = document.getElementById('splitAfter');
    const dividerLine = document.getElementById('splitDividerLine');

    if (!sliderInput || !splitAfter || !dividerLine) return;

    sliderInput.addEventListener('input', (e) => {
        const val = e.target.value;
        splitAfter.style.clipPath = `polygon(${val}% 0, 100% 0, 100% 100%, ${val}% 100%)`;
        dividerLine.style.left = `${val}%`;
    });
}

/* -------------------------------------------------------------------------
   3. Synchronized Dual-Video Combat Demo (Preloaded Video)
   ------------------------------------------------------------------------- */
function initSyncedDemoPlayer() {
    const vRaw = document.getElementById('videoRawDemo');
    const vHud = document.getElementById('videoHudDemo');
    const hudWrapper = document.getElementById('hudWrapper');
    const splitBar = document.getElementById('videoSplitBar');
    const container = document.getElementById('syncedContainer');
    
    const playBtn = document.getElementById('demoPlayBtn');
    const playIcon = document.getElementById('playIcon');
    const timeline = document.getElementById('demoTimeline');
    const progressFill = document.getElementById('timelineProgressFill');
    const timeDisplay = document.getElementById('demoTimeDisplay');
    const restartBtn = document.getElementById('btnRestartDemo');
    const fsBtn = document.getElementById('btnFullscreenDemo');

    const modeSplit = document.getElementById('modeSplitBtn');
    const modeHud = document.getElementById('modeHudBtn');
    const modeRaw = document.getElementById('modeRawBtn');

    if (!vRaw || !vHud || !container) return;

    let isPlaying = false;
    let splitPos = 50; // percentage
    let currentMode = 'split'; // 'split', 'hud', 'raw'

    // Synchronize play / pause
    function togglePlay() {
        if (isPlaying) {
            vRaw.pause();
            vHud.pause();
            isPlaying = false;
            if (playIcon) playIcon.textContent = '▶';
        } else {
            vRaw.play().catch(e => console.log('Raw play error:', e));
            vHud.play().catch(e => console.log('HUD play error:', e));
            isPlaying = true;
            if (playIcon) playIcon.textContent = '❚❚';
        }
    }

    if (playBtn) playBtn.addEventListener('click', togglePlay);
    container.addEventListener('click', (e) => {
        // Prevent toggle if clicking split bar
        if (e.target.closest('#videoSplitBar')) return;
        togglePlay();
    });

    if (restartBtn) {
        restartBtn.addEventListener('click', () => {
            vRaw.currentTime = 0;
            vHud.currentTime = 0;
            if (!isPlaying) togglePlay();
        });
    }

    if (fsBtn) {
        fsBtn.addEventListener('click', () => {
            if (!document.fullscreenElement) {
                container.requestFullscreen().catch(err => alert(`Fullscreen error: ${err.message}`));
            } else {
                document.exitFullscreen();
            }
        });
    }

    // Time update & sync lock
    vHud.addEventListener('timeupdate', () => {
        // Keep within 0.1s sync
        if (Math.abs(vRaw.currentTime - vHud.currentTime) > 0.15) {
            vRaw.currentTime = vHud.currentTime;
        }

        const cur = vHud.currentTime;
        const dur = vHud.duration || 20.0;
        const pct = (cur / dur) * 100;

        if (timeline) timeline.value = pct;
        if (progressFill) progressFill.style.width = `${pct}%`;
        if (timeDisplay) timeDisplay.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;

        updateCensusTelemetry(cur);
    });

    // Timeline scrubbing
    if (timeline) {
        timeline.addEventListener('input', (e) => {
            const pct = e.target.value;
            const dur = vHud.duration || 20.0;
            const targetTime = (pct / 100) * dur;
            vRaw.currentTime = targetTime;
            vHud.currentTime = targetTime;
        });
    }

    // Split Bar dragging logic
    let isDraggingSplit = false;

    function updateSplitPosition(clientX) {
        const rect = container.getBoundingClientRect();
        let x = clientX - rect.left;
        let pct = (x / rect.width) * 100;
        pct = Math.max(5, Math.min(95, pct));
        splitPos = pct;

        if (currentMode === 'split') {
            hudWrapper.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
            splitBar.style.left = `${pct}%`;
        }
    }

    splitBar.addEventListener('mousedown', (e) => {
        isDraggingSplit = true;
        e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDraggingSplit) return;
        updateSplitPosition(e.clientX);
    });

    window.addEventListener('mouseup', () => {
        isDraggingSplit = false;
    });

    // Touch support for split bar
    splitBar.addEventListener('touchstart', () => { isDraggingSplit = true; }, { passive: true });
    window.addEventListener('touchmove', (e) => {
        if (!isDraggingSplit || !e.touches[0]) return;
        updateSplitPosition(e.touches[0].clientX);
    }, { passive: true });
    window.addEventListener('touchend', () => { isDraggingSplit = false; });

    // Mode Switcher (Split, Full HUD, Raw)
    function setMode(mode) {
        currentMode = mode;
        [modeSplit, modeHud, modeRaw].forEach(btn => btn?.classList.remove('active'));

        if (mode === 'split') {
            modeSplit?.classList.add('active');
            hudWrapper.style.clipPath = `polygon(${splitPos}% 0, 100% 0, 100% 100%, ${splitPos}% 100%)`;
            splitBar.style.display = 'block';
        } else if (mode === 'hud') {
            modeHud?.classList.add('active');
            hudWrapper.style.clipPath = 'polygon(0 0, 100% 0, 100% 100%, 0 100%)';
            splitBar.style.display = 'none';
        } else if (mode === 'raw') {
            modeRaw?.classList.add('active');
            hudWrapper.style.clipPath = 'polygon(100% 0, 100% 0, 100% 100%, 100% 100%)';
            splitBar.style.display = 'none';
        }
    }

    if (modeSplit) modeSplit.addEventListener('click', () => setMode('split'));
    if (modeHud) modeHud.addEventListener('click', () => setMode('hud'));
    if (modeRaw) modeRaw.addEventListener('click', () => setMode('raw'));

    // Dynamic telemetry updates during demo playback
    function updateCensusTelemetry(t) {
        const cTank = document.getElementById('countTank');
        const cApc = document.getElementById('countAPC');
        const cAir = document.getElementById('countAirDefense');
        const logBox = document.getElementById('tacticalEventLog');

        if (t > 2.0 && cApc && cApc.textContent === '0') {
            cApc.textContent = '2';
            appendLog(logBox, `[T+${t.toFixed(1)}s] Target #1 Lead APC acquired in transit.`);
        }
        if (t > 6.0 && cTank && cTank.textContent === '0') {
            cTank.textContent = '1';
            appendLog(logBox, `[T+${t.toFixed(1)}s] Target #2 Russian Tank identified | Heading 085° at 19 px/s.`);
        }
        if (t > 11.0 && cApc && cApc.textContent === '2') {
            cApc.textContent = '4';
            appendLog(logBox, `[T+${t.toFixed(1)}s] Target #3 & #4 Trailing APCs acquired in convoy formation.`);
        }
        if (t > 15.0 && cAir && cAir.textContent === '0') {
            cAir.textContent = '1';
            appendLog(logBox, `[T+${t.toFixed(1)}s] Target #5 Mobile Air Defense platform tracked.`);
        }
    }

    function appendLog(box, text) {
        if (!box) return;
        const entry = document.createElement('div');
        entry.className = 'log-entry log-detect';
        entry.textContent = text;
        box.appendChild(entry);
        box.scrollTop = box.scrollHeight;
    }
}

function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

/* -------------------------------------------------------------------------
   4. Demo Section Tabs
   ------------------------------------------------------------------------- */
function initTabs() {
    const tabPreloadedBtn = document.getElementById('tabPreloadedBtn');
    const tabLiveRunnerBtn = document.getElementById('tabLiveRunnerBtn');
    const tabPreloadedContent = document.getElementById('tabPreloadedContent');
    const tabLiveRunnerContent = document.getElementById('tabLiveRunnerContent');

    if (!tabPreloadedBtn || !tabLiveRunnerBtn) return;

    tabPreloadedBtn.addEventListener('click', () => {
        tabPreloadedBtn.classList.add('active');
        tabLiveRunnerBtn.classList.remove('active');
        tabPreloadedContent.classList.add('active');
        tabLiveRunnerContent.classList.remove('active');
    });

    tabLiveRunnerBtn.addEventListener('click', () => {
        tabLiveRunnerBtn.classList.add('active');
        tabPreloadedBtn.classList.remove('active');
        tabLiveRunnerContent.classList.add('active');
        tabPreloadedContent.classList.remove('active');
    });
}

/* -------------------------------------------------------------------------
   5. Live Model Runner & Custom Video Upload Controller
   ------------------------------------------------------------------------- */
function initLiveRunner() {
    const radioDefault = document.getElementById('radioDefault');
    const radioUpload = document.getElementById('radioUpload');
    const radioDefaultLabel = document.getElementById('radioDefaultLabel');
    const radioUploadLabel = document.getElementById('radioUploadLabel');
    const dropzoneWrapper = document.getElementById('uploadDropzoneWrapper');
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('videoFileInput');
    const browseBtn = document.getElementById('browseFileBtn');
    const uploadStatus = document.getElementById('uploadStatusText');

    const confSlider = document.getElementById('confSlider');
    const confBadge = document.getElementById('confValueBadge');

    const runForm = document.getElementById('runModelForm');
    const btnLaunch = document.getElementById('btnLaunchRun');

    // Runner Monitor Elements
    const monitorPip = document.getElementById('liveRunnerPip');
    const monitorTitle = document.getElementById('liveRunnerStatusTitle');
    const fpsBadge = document.getElementById('liveFpsBadge');
    const placeholder = document.getElementById('viewportPlaceholder');
    const streamImg = document.getElementById('liveStreamImg');
    const resultVideo = document.getElementById('liveResultVideo');
    const progressFill = document.getElementById('runnerProgressBarFill');
    const progressLabel = document.getElementById('progressLabelText');
    const progressFrames = document.getElementById('progressFrameText');
    const logFeed = document.getElementById('liveLogFeed');
    const downloadBox = document.getElementById('downloadActionBox');
    const downloadBtn = document.getElementById('btnDownloadOutput');

    let uploadedFileId = null;
    let pollInterval = null;

    // Radio Source Switching
    function switchSource(source) {
        if (source === 'default') {
            radioDefault.checked = true;
            radioDefaultLabel.classList.add('active');
            radioUploadLabel.classList.remove('active');
            dropzoneWrapper.style.display = 'none';
        } else {
            radioUpload.checked = true;
            radioUploadLabel.classList.add('active');
            radioDefaultLabel.classList.remove('active');
            dropzoneWrapper.style.display = 'block';
        }
    }

    if (radioDefault) radioDefault.addEventListener('change', () => switchSource('default'));
    if (radioUpload) radioUpload.addEventListener('change', () => switchSource('upload'));
    if (radioDefaultLabel) radioDefaultLabel.addEventListener('click', () => switchSource('default'));
    if (radioUploadLabel) radioUploadLabel.addEventListener('click', () => switchSource('upload'));

    // File Upload Handling
    if (browseBtn && fileInput) {
        browseBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileSelect);
    }

    if (dropzone) {
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                uploadSelectedFile(e.dataTransfer.files[0]);
            }
        });
    }

    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            uploadSelectedFile(e.target.files[0]);
        }
    }

    async function uploadSelectedFile(file) {
        uploadStatus.textContent = `Uploading ${file.name} (${(file.size / (1024*1024)).toFixed(1)} MB)...`;
        uploadStatus.style.color = 'var(--accent-secondary)';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Upload failed');
            }

            const data = await res.json();
            uploadedFileId = data.file_id;
            uploadStatus.textContent = `✓ Uploaded: ${file.name} (Ready for model inference)`;
            uploadStatus.style.color = 'var(--accent-primary)';
        } catch (err) {
            uploadStatus.textContent = `Upload error: ${err.message}`;
            uploadStatus.style.color = 'var(--accent-danger)';
        }
    }

    // Confidence Slider
    if (confSlider && confBadge) {
        confSlider.addEventListener('input', (e) => {
            confBadge.textContent = `${e.target.value}%`;
        });
    }

    // Launch Model Execution
    if (runForm) {
        runForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const isDefault = radioDefault.checked;
            if (!isDefault && !uploadedFileId) {
                alert('Please upload a video file first or select Preloaded Combat Footage.');
                return;
            }

            const modelChoice = document.getElementById('modelSelect')?.value || 'auto';
            const conf = parseFloat(confSlider.value) / 100.0;
            const durationSec = parseInt(document.getElementById('durationSelect').value, 10);
            const startSec = parseFloat(document.getElementById('startOffsetInput').value) || 0.0;
            const useClahe = document.getElementById('checkClahe').checked;
            const useTracking = document.getElementById('checkTracking').checked;

            // Reset UI for running
            btnLaunch.disabled = true;
            btnLaunch.innerHTML = '<span class="status-pip pulse"></span> RUNNING THIRDEYE PIPELINE...';
            monitorPip.className = 'status-pip pulse';
            monitorTitle.textContent = 'MONITOR STATUS: INFERENCE ACTIVE';
            placeholder.style.display = 'none';
            resultVideo.style.display = 'none';
            streamImg.style.display = 'block';
            downloadBox.style.display = 'none';
            logFeed.innerHTML = '<div class="text-accent">[SYS] Launching YOLO11 & ByteTrack pipeline...</div>';

            const useSahi = document.getElementById('checkSahi')?.checked || false;
            const useTta = document.getElementById('checkTta')?.checked || false;
            const applyNvg = document.getElementById('checkNvg')?.checked || false;

            const payload = new FormData();
            payload.append('video_source', isDefault ? 'default' : 'uploaded');
            if (!isDefault) payload.append('uploaded_file_id', uploadedFileId);
            payload.append('model_choice', modelChoice);
            payload.append('conf', conf.toString());
            payload.append('duration_sec', durationSec.toString());
            payload.append('start_sec', startSec.toString());
            payload.append('use_clahe', useClahe.toString());
            payload.append('use_tracking', useTracking.toString());
            payload.append('use_sahi', useSahi.toString());
            payload.append('use_tta', useTta.toString());
            payload.append('apply_nvg', applyNvg.toString());

            try {
                const res = await fetch('/api/run', {
                    method: 'POST',
                    body: payload
                });

                if (!res.ok) throw new Error('Failed to initiate model execution.');
                const jobData = await res.json();
                const jobId = jobData.job_id;

                // Start polling progress
                startProgressPolling(jobId);
            } catch (err) {
                alert(`Error launching pipeline: ${err.message}`);
                resetRunButton();
            }
        });
    }

    function startProgressPolling(jobId) {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/progress/${jobId}`);
                if (!res.ok) return;
                const data = await res.json();

                // Update FPS & frames
                if (fpsBadge) fpsBadge.textContent = `${data.fps.toFixed(1)} FPS`;
                const pct = Math.min(100, Math.round(data.progress * 100));
                if (progressFill) progressFill.style.width = `${pct}%`;
                if (progressLabel) progressLabel.textContent = `Progress: ${pct}%`;
                if (progressFrames) progressFrames.textContent = `Frame: ${data.frame} / ${data.total_frames || '?'}`;

                // Live Stream Frame Decode
                if (data.latest_frame_b64) {
                    streamImg.src = `data:image/jpeg;base64,${data.latest_frame_b64}`;
                }

                // Update Live Census counts
                if (data.unique_targets) {
                    document.getElementById('liveCntSoldier').textContent = data.unique_targets['Soldier'] || 0;
                    document.getElementById('liveCntTank').textContent = data.unique_targets['Tank'] || 0;
                    document.getElementById('liveCntAPC').textContent = data.unique_targets['APC'] || 0;
                    document.getElementById('liveCntTruck').textContent = data.unique_targets['Military_Truck'] || 0;
                    document.getElementById('liveCntDefense').textContent = data.unique_targets['Mobile_Air_Defense'] || 0;
                }

                // Append events to log feed
                if (data.events && data.events.length > 0) {
                    data.events.forEach(ev => {
                        const existing = document.getElementById(`ev_${ev.track_id}_${ev.class}`);
                        if (!existing) {
                            const row = document.createElement('div');
                            row.id = `ev_${ev.track_id}_${ev.class}`;
                            row.className = 'log-entry log-detect';
                            row.textContent = `[T+${ev.time_sec}s] ${ev.message}`;
                            logFeed.appendChild(row);
                            logFeed.scrollTop = logFeed.scrollHeight;
                        }
                    });
                }

                // Check completion
                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    finishExecution(data);
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    alert(`Pipeline error: ${data.error || 'Unknown failure'}`);
                    resetRunButton();
                }
            } catch (e) {
                console.warn('Progress poll error:', e);
            }
        }, 500);
    }

    function finishExecution(data) {
        resetRunButton();
        monitorPip.className = 'status-pip';
        monitorTitle.textContent = 'MONITOR STATUS: COMPLETED';

        // Swap to native video player
        streamImg.style.display = 'none';
        resultVideo.style.display = 'block';
        resultVideo.src = data.video_url;
        resultVideo.load();
        resultVideo.play().catch(e => console.log('Autoplay blocked:', e));

        // Show download button
        downloadBox.style.display = 'block';
        downloadBtn.href = data.video_url;

        // Final event in log
        const fin = document.createElement('div');
        fin.className = 'log-entry log-census';
        fin.textContent = `✓ Ingestion completed: ${data.result.frames_processed} frames analyzed (${data.result.duration_sec}s). Output ready.`;
        logFeed.appendChild(fin);
        logFeed.scrollTop = logFeed.scrollHeight;
    }

    function resetRunButton() {
        btnLaunch.disabled = false;
        btnLaunch.innerHTML = '<span class="btn-icon">⚡</span><span>EXECUTE THIRDEYE PIPELINE ON VIDEO</span>';
    }
}
