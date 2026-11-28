/* ==========================================================================
   Live gesture recognition controller.

   Responsibilities
   ----------------
   * Drive the webcam and capture frames at a controlled rate (5-8 fps).
   * Send compressed JPEG frames to the Flask prediction endpoint.
   * Never let prediction requests pile up (single in-flight request).
   * Stabilise the output: a gesture must be confirmed across several
     consecutive frames before it is reported, and a gesture is registered
     only once until a different gesture (or a neutral interval) occurs.
   * Draw the hand landmark overlay returned by the server.
   ========================================================================== */
(function () {
    "use strict";

    const CONNECTION_LANDMARKS = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20],
        [0, 17]
    ];

    class RecognitionController {
        constructor(options) {
            this.options = Object.assign(
                {
                    framesPerSecond: 6,
                    captureWidth: 640,
                    captureHeight: 480,
                    jpegQuality: 0.82,
                    confidenceThreshold: 0.6,
                    confirmationFrames: 3,
                    neutralFramesToReset: 8,
                    predictUrl: "/recognition/api/predict?include_landmarks=1",
                    saveUrl: "/recognition/api/save",
                    landmarkConnection: CONNECTION_LANDMARKS
                },
                options || {}
            );

            this.video = document.getElementById("camera-video");
            this.overlay = document.getElementById("overlay-canvas");
            this.camera = null;

            this.running = false;
            this.paused = false;
            this.inFlight = false;
            this.timer = null;

            // Stabilisation state
            this.candidateLabel = null;
            this.candidateCount = 0;
            this.registeredLabel = null;
            this.neutralCount = 0;
            this.lastResult = null;

            this.callbacks = {
                onResult: this.options.onResult || function () {},
                onState: this.options.onState || function () {},
                onError: this.options.onError || function () {},
                onStatus: this.options.onStatus || function () {}
            };
        }

        /* ------------------------------------------------------- Lifecycle */
        async startCamera(deviceId) {
            if (!this.camera) {
                this.camera = new CameraController(this.video, {
                    onStateChange: (state, error) => {
                        this.callbacks.onStatus({
                            cameraState: state,
                            cameraError: error
                        });
                    }
                });
            }
            try {
                await this.camera.start(deviceId);
                this.callbacks.onStatus({ cameraState: "live", cameraError: null });
                await this.enumerateCameras();
                return true;
            } catch (error) {
                this.callbacks.onStatus({ cameraState: "error", cameraError: error.message });
                this.callbacks.onError(error.message);
                return false;
            }
        }

        async enumerateCameras() {
            if (!this.camera) return [];
            const devices = await this.camera.listCameras();
            this.callbacks.onStatus({ cameras: devices });
            return devices;
        }

        stopCamera() {
            this.pause();
            if (this.camera) {
                this.camera.stop();
            }
            this.clearOverlay();
            this.callbacks.onStatus({ cameraState: "idle", cameraError: null });
        }

        start() {
            if (this.running) return;
            this.running = true;
            this.paused = false;
            this.callbacks.onState("running");
            this.scheduleNext();
        }

        pause() {
            if (!this.running) return;
            this.paused = true;
            this.callbacks.onState("paused");
        }

        resume() {
            if (!this.running) {
                this.start();
                return;
            }
            this.paused = false;
            this.callbacks.onState("running");
        }

        stop() {
            this.running = false;
            this.paused = false;
            if (this.timer) {
                window.clearTimeout(this.timer);
                this.timer = null;
            }
            this.callbacks.onState("stopped");
        }

        toggle() {
            if (!this.running || this.paused) {
                this.resume();
            } else {
                this.pause();
            }
        }

        clearResult() {
            this.lastResult = null;
            this.candidateLabel = null;
            this.candidateCount = 0;
            this.registeredLabel = null;
            this.neutralCount = 0;
            this.callbacks.onResult(null);
        }

        /* ---------------------------------------------------------- Capture */
        scheduleNext() {
            if (!this.running) return;
            const interval = Math.max(60, Math.round(1000 / this.options.framesPerSecond));
            this.timer = window.setTimeout(() => this.tick(), interval);
        }

        async tick() {
            if (!this.running) return;
            if (!this.paused) {
                await this.processFrame();
            }
            this.scheduleNext();
        }

        async processFrame() {
            if (this.inFlight) return;
            if (!this.camera || this.camera.state !== "live") return;

            const canvas = this.camera.captureFrame(
                this.options.captureWidth,
                this.options.captureHeight
            );
            if (!canvas) return;

            const blob = await new Promise((resolve) => {
                canvas.toBlob(resolve, "image/jpeg", this.options.jpegQuality);
            });
            if (!blob) return;

            const form = new FormData();
            form.append("file", blob, "frame.jpg");

            this.inFlight = true;
            try {
                const response = await window.MSL.request(this.options.predictUrl, {
                    method: "POST",
                    body: form
                });
                this.handleResponse(response);
            } catch (error) {
                this.callbacks.onError(error.message);
            } finally {
                this.inFlight = false;
            }
        }

        /* --------------------------------------------------- Stabilisation */
        handleResponse(response) {
            if (!response) return;

            if (response.landmarks) {
                this.drawOverlay(response.landmarks);
            } else if (!response.hand_detected) {
                this.clearOverlay();
            }

            const result = {
                recognized: !!response.recognized,
                gesture: response.gesture || null,
                translation: response.translation || null,
                confidence:
                    typeof response.confidence === "number" ? response.confidence : 0,
                handDetected: !!response.hand_detected,
                handCount: response.hand_count || 0,
                modelVersion: response.model_version || null,
                message: response.message || null,
                latencyMs: response.latency_ms || null,
                confirmed: false,
                registered: false
            };

            const threshold = this.options.confidenceThreshold;

            if (!result.recognized || !result.gesture) {
                // Neutral observation: reset the confirmation counter.
                this.neutralCount += 1;
                this.candidateLabel = null;
                this.candidateCount = 0;
                if (this.neutralCount >= this.options.neutralFramesToReset) {
                    this.registeredLabel = null;
                    this.neutralCount = 0;
                }
                result.confirmed = false;
                this.lastResult = result;
                this.callbacks.onResult(result);
                return;
            }

            if (result.confidence < threshold) {
                result.confirmed = false;
                this.lastResult = result;
                this.callbacks.onResult(result);
                return;
            }

            if (result.gesture === this.candidateLabel) {
                this.candidateCount += 1;
            } else {
                this.candidateLabel = result.gesture;
                this.candidateCount = 1;
            }

            if (this.candidateCount >= this.options.confirmationFrames) {
                result.confirmed = true;
                if (this.registeredLabel !== result.gesture) {
                    this.registeredLabel = result.gesture;
                    result.registered = true;
                }
            }

            this.lastResult = result;
            this.callbacks.onResult(result);
        }

        /* ---------------------------------------------------------- Overlay */
        drawOverlay(landmarks) {
            if (!this.overlay || !landmarks || !landmarks.hands) return;
            const canvas = this.overlay;
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

            landmarks.hands.forEach((hand) => {
                const points = (hand.landmarks || []).map((point) => [
                    point[0] * width,
                    point[1] * height
                ]);
                if (points.length !== 21) return;

                context.lineWidth = Math.max(2, 2 * ratio);
                context.strokeStyle = "rgba(37, 99, 235, 0.95)";
                this.options.landmarkConnection.forEach((pair) => {
                    context.beginPath();
                    context.moveTo(points[pair[0]][0], points[pair[0]][1]);
                    context.lineTo(points[pair[1]][0], points[pair[1]][1]);
                    context.stroke();
                });

                context.fillStyle = "rgba(21, 128, 61, 0.95)";
                points.forEach((point) => {
                    context.beginPath();
                    context.arc(point[0], point[1], 3.5 * ratio, 0, Math.PI * 2);
                    context.fill();
                });
            });
        }

        clearOverlay() {
            if (!this.overlay) return;
            const context = this.overlay.getContext("2d");
            context.clearRect(0, 0, this.overlay.width, this.overlay.height);
        }

        /* ------------------------------------------------------------ Save */
        async saveResult() {
            const result = this.lastResult;
            if (!result || !result.recognized || !result.gesture) {
                throw new Error("There is no recognised gesture to save.");
            }
            const response = await window.MSL.api.post(this.options.saveUrl, {
                gesture: result.gesture,
                confidence: result.confidence,
                model_version: result.modelVersion
            });
            return response;
        }
    }

    window.RecognitionController = RecognitionController;
    window.MSL_CONNECTIONS = CONNECTION_LANDMARKS;
})(window);
