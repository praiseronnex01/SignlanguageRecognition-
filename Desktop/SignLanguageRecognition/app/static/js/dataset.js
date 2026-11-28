/* ==========================================================================
   Dataset collection station.

   Captures frames in the browser, sends them to the server for hand
   landmark extraction and stores the resulting labelled samples.
   ========================================================================== */
(function () {
    "use strict";

    const video = document.getElementById("ds-video");
    if (!video) return;

    const els = {
        overlay: document.getElementById("ds-overlay"),
        placeholder: document.getElementById("ds-placeholder"),
        gesture: document.getElementById("collect-gesture"),
        participant: document.getElementById("collect-participant"),
        consent: document.getElementById("collect-consent"),
        cameraStatus: document.getElementById("ds-camera-status"),
        message: document.getElementById("ds-message"),
        sessionLabel: document.getElementById("ds-session-label"),
        counter: document.getElementById("ds-counter"),
        progress: document.getElementById("ds-progress"),
        sampleRows: document.getElementById("ds-sample-rows"),
        sampleCount: document.getElementById("ds-sample-count"),
        pageLabel: document.getElementById("ds-page"),
        filterGesture: document.getElementById("ds-filter-gesture"),
        filterParticipant: document.getElementById("ds-filter-participant"),
        btnRegister: document.getElementById("btn-register-participant"),
        btnStartSession: document.getElementById("btn-start-session"),
        btnStartCamera: document.getElementById("ds-start-camera"),
        btnStopCamera: document.getElementById("ds-stop-camera"),
        btnStartCollection: document.getElementById("ds-start-collection"),
        btnPauseCollection: document.getElementById("ds-pause-collection"),
        btnStopCollection: document.getElementById("ds-stop-collection"),
        btnRefresh: document.getElementById("btn-refresh-samples"),
        btnExport: document.getElementById("btn-export"),
        btnPrev: document.getElementById("ds-prev"),
        btnNext: document.getElementById("ds-next")
    };

    const CONNECTIONS = window.MSL_CONNECTIONS || [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9],
        [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
    ];

    const state = {
        camera: null,
        sessionId: null,
        sessionCode: null,
        collecting: false,
        paused: false,
        inFlight: false,
        timer: null,
        captured: 0,
        page: 1,
        pages: 1,
        total: 0
    };

    const TARGET_SAMPLES = 200;

    function showMessage(text, tone) {
        els.message.textContent = text;
        els.message.classList.remove("hidden");
        els.message.className =
            "rounded-md border px-4 py-3 text-sm " +
            (tone === "bad"
                ? "border-bad/40 bg-bad/5 text-bad"
                : tone === "ok"
                ? "border-ok/40 bg-ok/5 text-ok"
                : "border-line bg-surface text-navy-muted");
    }

    function clearMessage() {
        els.message.classList.add("hidden");
        els.message.textContent = "";
    }

    function setCameraStatus(state, error) {
        const dot =
            '<span class="inline-flex h-2 w-2 rounded-full ' +
            (state === "live" ? "bg-ok" : state === "error" ? "bg-bad" : "bg-slate-300") +
            '"></span>';
        const label = { live: "Camera live", starting: "Starting…", error: "Camera error", idle: "Camera off" };
        els.cameraStatus.innerHTML = dot + "<span>" + (label[state] || state) + "</span>";
        if (error) showMessage(error, "bad");
    }

    /* ------------------------------------------------------------ Camera */
    async function startCamera() {
        if (!state.camera) {
            state.camera = new CameraController(video, {
                onStateChange: setCameraStatus
            });
        }
        const ok = await state.camera.start();
        if (ok) {
            els.placeholder.classList.add("hidden");
            els.btnStartCamera.disabled = true;
            els.btnStopCamera.disabled = false;
            els.btnStartCollection.disabled = state.sessionId ? false : true;
        }
        return ok;
    }

    function stopCamera() {
        if (state.camera) state.camera.stop();
        els.placeholder.classList.remove("hidden");
        els.btnStartCamera.disabled = false;
        els.btnStopCamera.disabled = true;
        els.btnStartCollection.disabled = true;
        els.btnPauseCollection.disabled = true;
        stopLoop();
    }

    /* -------------------------------------------------------- Collection */
    function startLoop() {
        if (state.timer) window.clearTimeout(state.timer);
        state.collecting = true;
        state.paused = false;
        els.btnStartCollection.disabled = true;
        els.btnPauseCollection.disabled = false;
        els.btnStopCollection.disabled = false;
        scheduleTick();
    }

    function scheduleTick() {
        if (!state.collecting) return;
        state.timer = window.setTimeout(tick, 1000 / 5);
    }

    async function tick() {
        if (!state.collecting) return;
        if (!state.paused) {
            await captureSample();
        }
        scheduleTick();
    }

    async function captureSample() {
        if (state.inFlight || !state.camera || state.camera.state !== "live") return;
        if (!state.sessionId || !els.gesture.value) return;

        const canvas = state.camera.captureFrame(640, 480);
        if (!canvas) return;

        const blob = await new Promise(function (resolve) {
            canvas.toBlob(resolve, "image/jpeg", 0.82);
        });
        if (!blob) return;

        const form = new FormData();
        form.append("file", blob, "frame.jpg");
        form.append("gesture_id", els.gesture.value);
        form.append("session_id", state.sessionId);
        form.append("participant_code", els.participant.value.trim().toUpperCase());

        state.inFlight = true;
        try {
            const response = await window.MSL.request("/dataset/api/dataset/collect", {
                method: "POST",
                body: form
            });

            if (response.landmarks) drawOverlay(response.landmarks);

            if (response.stored) {
                state.captured += 1;
                els.counter.textContent = state.captured + " samples captured";
                els.progress.style.width = Math.min(100, (state.captured / TARGET_SAMPLES) * 100) + "%";
            } else if (response.message) {
                showMessage(response.message, "info");
            }
        } catch (error) {
            showMessage(error.message, "bad");
        } finally {
            state.inFlight = false;
        }
    }

    function drawOverlay(landmarks) {
        const canvas = els.overlay;
        if (!canvas || !landmarks || !landmarks.hands) return;
        const rect = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(rect.width * ratio)) {
            canvas.width = Math.round(rect.width * ratio);
            canvas.height = Math.round(rect.height * ratio);
        }
        const context = canvas.getContext("2d");
        context.clearRect(0, 0, canvas.width, canvas.height);
        const width = canvas.width;
        const height = canvas.height;

        landmarks.hands.forEach(function (hand) {
            const points = (hand.landmarks || []).map(function (point) {
                return [point[0] * width, point[1] * height];
            });
            if (points.length !== 21) return;
            context.lineWidth = Math.max(2, 2 * ratio);
            context.strokeStyle = "rgba(37, 99, 235, 0.95)";
            CONNECTIONS.forEach(function (pair) {
                context.beginPath();
                context.moveTo(points[pair[0]][0], points[pair[0]][1]);
                context.lineTo(points[pair[1]][0], points[pair[1]][1]);
                context.stroke();
            });
            context.fillStyle = "rgba(21, 128, 61, 0.95)";
            points.forEach(function (point) {
                context.beginPath();
                context.arc(point[0], point[1], 3.5 * ratio, 0, Math.PI * 2);
                context.fill();
            });
        });
    }

    function stopLoop() {
        state.collecting = false;
        if (state.timer) {
            window.clearTimeout(state.timer);
            state.timer = null;
        }
        els.btnStartCollection.disabled = !state.sessionId;
        els.btnPauseCollection.disabled = true;
        els.btnStopCollection.disabled = true;
    }

    /* ------------------------------------------------------------ Session */
    els.btnRegister.addEventListener("click", async function () {
        const code = els.participant.value.trim().toUpperCase();
        if (!code) {
            showMessage("Enter a participant code first.", "bad");
            return;
        }
        try {
            const response = await window.MSL.api.post("/dataset/api/dataset/participants", {
                participant_code: code,
                consent_status: els.consent.value
            });
            showMessage(
                "Participant " + response.participant.participant_code + " registered with consent status '" +
                response.participant.consent_status + "'.",
                "ok"
            );
            refreshStats();
        } catch (error) {
            showMessage(error.message, "bad");
        }
    });

    els.btnStartSession.addEventListener("click", async function () {
        const code = els.participant.value.trim().toUpperCase();
        if (!code) {
            showMessage("Enter a participant code first.", "bad");
            return;
        }
        try {
            const response = await window.MSL.api.post("/dataset/api/dataset/sessions", {
                participant_code: code
            });
            state.sessionId = response.session.id;
            state.sessionCode = response.session.session_code;
            els.sessionLabel.textContent = "Session " + response.session.session_code;
            showMessage("Collection session " + response.session.session_code + " is active.", "ok");
            els.btnStartCollection.disabled = state.camera && state.camera.state === "live" ? false : true;
            refreshStats();
        } catch (error) {
            showMessage(error.message, "bad");
        }
    });

    /* ----------------------------------------------------------- Samples */
    async function loadSamples() {
        const params = new URLSearchParams();
        params.set("page", state.page);
        params.set("per_page", 20);
        if (els.filterGesture.value) params.set("gesture_id", els.filterGesture.value);
        if (els.filterParticipant.value) params.set("participant_code", els.filterParticipant.value);

        try {
            const response = await window.MSL.api.get("/dataset/api/dataset/samples?" + params.toString());
            state.total = response.total;
            state.pages = response.pages || 1;
            renderSamples(response.samples || []);
            els.sampleCount.textContent = response.total + " sample(s) stored";
            els.pageLabel.textContent = response.page + " / " + (response.pages || 1);
        } catch (error) {
            els.sampleRows.innerHTML =
                '<tr><td colspan="9" class="py-8 text-center text-sm text-bad">' +
                window.MSL.escapeHtml(error.message) + "</td></tr>";
        }
    }

    function renderSamples(samples) {
        if (!samples.length) {
            els.sampleRows.innerHTML =
                '<tr><td colspan="9" class="py-8 text-center text-sm text-slate-400">No samples match the current filters.</td></tr>';
            return;
        }
        els.sampleRows.innerHTML = "";
        samples.forEach(function (sample) {
            const row = document.createElement("tr");
            row.innerHTML =
                "<td>" + sample.id + "</td>" +
                '<td class="font-medium text-navy">' + window.MSL.escapeHtml(sample.gesture_label || "-") + "</td>" +
                "<td>" + window.MSL.escapeHtml(sample.participant_code || "-") + "</td>" +
                "<td>" + window.MSL.escapeHtml(sample.session_code || "-") + "</td>" +
                "<td>" + (sample.hand_count || 0) + "</td>" +
                "<td>" + window.MSL.escapeHtml(sample.handedness || "-") + "</td>" +
                "<td>" + (sample.quality_score === null || sample.quality_score === undefined
                    ? "-"
                    : Number(sample.quality_score).toFixed(2)) + "</td>" +
                "<td class='whitespace-nowrap text-slate-500'>" + window.MSL.formatDateTime(sample.created_at) + "</td>" +
                '<td class="text-right"><button type="button" class="btn btn-secondary btn-sm js-delete-sample" data-id="' +
                sample.id + '"><i class="fa-solid fa-trash"></i></button></td>';
            els.sampleRows.appendChild(row);
        });

        els.sampleRows.querySelectorAll(".js-delete-sample").forEach(function (button) {
            button.addEventListener("click", async function () {
                if (!window.confirm("Delete sample #" + button.dataset.id + "?")) return;
                try {
                    await window.MSL.api.del("/dataset/api/dataset/samples/" + button.dataset.id);
                    window.MSL.toast("The sample was deleted.", "ok");
                    loadSamples();
                    refreshStats();
                } catch (error) {
                    window.MSL.toast(error.message, "bad");
                }
            });
        });
    }

    /* -------------------------------------------------------------- Stats */
    async function refreshStats() {
        try {
            const response = await window.MSL.api.get("/dataset/api/dataset/stats");
            const stats = response.stats;
            const set = function (id, value) {
                const element = document.getElementById(id);
                if (element) element.textContent = value;
            };
            set("ds-total", stats.total_samples);
            set("ds-participants", stats.participant_count);
            set("ds-duplicates", stats.duplicate_samples);
            set("ds-balance",
                stats.class_balance_ratio === null || stats.class_balance_ratio === undefined
                    ? "—"
                    : Math.round(stats.class_balance_ratio * 100) + "%");

            const badge = document.getElementById("ds-total-badge");
            if (badge) badge.textContent = stats.total_samples;

            const list = document.getElementById("ds-class-list");
            if (list) {
                const max = stats.per_gesture.reduce(function (acc, entry) {
                    return Math.max(acc, entry.sample_count);
                }, 0) || 1;
                list.innerHTML = "";
                stats.per_gesture.forEach(function (entry) {
                    const item = document.createElement("li");
                    item.innerHTML =
                        '<div class="flex items-center justify-between text-xs">' +
                        '<span class="font-medium text-navy">' + window.MSL.escapeHtml(entry.label) + "</span>" +
                        '<span class="text-slate-500">' + entry.sample_count + "</span></div>" +
                        '<div class="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-surface">' +
                        '<div class="h-full rounded-full ' + (entry.is_validated ? "bg-ok" : "bg-accent") +
                        '" style="width: ' + (entry.sample_count / max) * 100 + '%"></div></div>';
                    list.appendChild(item);
                });
            }
        } catch (error) {
            /* Statistics are refreshed on the next interaction. */
        }
    }

    /* ------------------------------------------------------------ Events */
    els.btnStartCamera.addEventListener("click", startCamera);
    els.btnStopCamera.addEventListener("click", stopCamera);
    els.btnStartCollection.addEventListener("click", startLoop);
    els.btnPauseCollection.addEventListener("click", function () {
        state.paused = !state.paused;
        els.btnPauseCollection.innerHTML = state.paused
            ? '<i class="fa-solid fa-play"></i> Resume'
            : '<i class="fa-solid fa-pause"></i> Pause collection';
    });
    els.btnStopCollection.addEventListener("click", async function () {
        stopLoop();
        els.btnPauseCollection.innerHTML = '<i class="fa-solid fa-pause"></i> Pause collection';
        if (state.sessionId) {
            try {
                await window.MSL.api.post("/dataset/api/dataset/sessions/" + state.sessionId + "/close", {
                    status: "completed"
                });
                showMessage("Collection session closed.", "ok");
                state.sessionId = null;
                els.sessionLabel.textContent = "No active session";
                els.btnStartCollection.disabled = true;
                refreshStats();
            } catch (error) {
                showMessage(error.message, "bad");
            }
        }
    });

    els.btnRefresh.addEventListener("click", function () {
        loadSamples();
        refreshStats();
    });
    els.btnExport.addEventListener("click", async function () {
        try {
            const response = await window.MSL.api.get("/dataset/api/dataset/export");
            window.MSL.toast(
                "Dataset " + response.dataset.version + " exported with " +
                response.dataset.sample_count + " sample(s).",
                "ok"
            );
        } catch (error) {
            window.MSL.toast(error.message, "bad");
        }
    });

    els.filterGesture.addEventListener("change", function () {
        state.page = 1;
        loadSamples();
    });
    els.filterParticipant.addEventListener("change", function () {
        state.page = 1;
        loadSamples();
    });
    els.btnPrev.addEventListener("click", function () {
        if (state.page > 1) {
            state.page -= 1;
            loadSamples();
        }
    });
    els.btnNext.addEventListener("click", function () {
        if (state.page < state.pages) {
            state.page += 1;
            loadSamples();
        }
    });

    window.addEventListener("beforeunload", function () {
        if (state.camera) state.camera.stop();
    });

    loadSamples();
})();
