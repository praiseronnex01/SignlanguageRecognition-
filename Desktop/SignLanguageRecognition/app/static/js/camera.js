/* ==========================================================================
   Reusable webcam controller built on the MediaDevices API.
   Handles device enumeration, stream lifecycle and frame capture.
   ========================================================================== */
(function (global) {
    "use strict";

    class CameraController {
        /**
         * @param {HTMLVideoElement} videoElement  Element that displays the stream.
         * @param {object} options
         */
        constructor(videoElement, options) {
            this.video = videoElement;
            this.options = Object.assign(
                {
                    width: { ideal: 960 },
                    height: { ideal: 540 },
                    facingMode: "user",
                    deviceId: null
                },
                options || {}
            );
            this.stream = null;
            this.devices = [];
            this.activeDeviceId = null;
            this.state = "idle"; // idle | starting | live | error
            this.lastError = null;
            this.onStateChange = this.options.onStateChange || function () {};
        }

        /** Enumerate the available video input devices. */
        async listCameras() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
                return [];
            }
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                this.devices = devices.filter((device) => device.kind === "videoinput");
            } catch (error) {
                this.devices = [];
            }
            return this.devices;
        }

        /** True when the browser exposes a camera API at all. */
        static isSupported() {
            return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
        }

        _buildConstraints() {
            const video = this.options.deviceId
                ? { deviceId: { exact: this.options.deviceId } }
                : { facingMode: this.options.facingMode };
            return {
                audio: false,
                video: Object.assign(
                    {
                        width: this.options.width,
                        height: this.options.height
                    },
                    video
                )
            };
        }

        /** Start the camera.  Resolves with the active MediaStream. */
        async start(deviceId) {
            if (!CameraController.isSupported()) {
                this.state = "error";
                this.lastError =
                    "This browser does not expose the MediaDevices API. Use a recent version of Chrome, Edge or Firefox over HTTPS or localhost.";
                this.onStateChange(this.state, this.lastError);
                throw new Error(this.lastError);
            }

            this.stop();
            this.state = "starting";
            this.lastError = null;
            this.onStateChange(this.state, null);

            try {
                const constraints = this._buildConstraints();
                if (deviceId) {
                    constraints.video.deviceId = { exact: deviceId };
                    delete constraints.video.facingMode;
                }
                this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (error) {
                this.state = "error";
                this.lastError = CameraController.describeError(error);
                this.onStateChange(this.state, this.lastError);
                throw new Error(this.lastError);
            }

            this.video.srcObject = this.stream;
            this.video.setAttribute("playsinline", "");
            this.video.muted = true;

            try {
                await this.video.play();
            } catch (error) {
                /* Autoplay can be rejected; the stream still works for capture. */
            }

            const track = this.stream.getVideoTracks()[0];
            if (track) {
                this.activeDeviceId = track.getSettings().deviceId || deviceId || null;
                track.addEventListener("ended", () => {
                    this.state = "idle";
                    this.onStateChange(this.state, "The camera stream was stopped.");
                });
            }

            this.state = "live";
            this.onStateChange(this.state, null);
            await this.listCameras();
            return this.stream;
        }

        /** Stop the camera and release every media track. */
        stop() {
            if (this.stream) {
                this.stream.getTracks().forEach((track) => track.stop());
                this.stream = null;
            }
            if (this.video) {
                this.video.srcObject = null;
            }
            if (this.state !== "error") {
                this.state = "idle";
                this.onStateChange(this.state, null);
            }
        }

        /** Switch to a different camera by device id. */
        async switchCamera(deviceId) {
            this.options.deviceId = deviceId;
            return this.start(deviceId);
        }

        /**
         * Draw the current frame into an offscreen canvas at a reduced size.
         * @returns {HTMLCanvasElement|null}
         */
        captureFrame(targetWidth, targetHeight) {
            if (!this.video || !this.video.videoWidth) return null;

            const sourceWidth = this.video.videoWidth;
            const sourceHeight = this.video.videoHeight;
            const maxWidth = targetWidth || 640;

            let width = maxWidth;
            let height = Math.round((sourceHeight / sourceWidth) * maxWidth);
            if (height > (targetHeight || 480)) {
                height = targetHeight || 480;
                width = Math.round((sourceWidth / sourceHeight) * height);
            }

            const canvas = document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;

            const context = canvas.getContext("2d");
            // The frame is captured WITHOUT mirroring so that MediaPipe's
            // handedness labels match the ones produced during dataset
            // collection.  Only the on-screen preview is mirrored.
            context.drawImage(this.video, 0, 0, width, height);
            return canvas;
        }

        /** Human readable description of a getUserMedia error. */
        static describeError(error) {
            if (!error) return "The camera could not be started.";
            switch (error.name) {
                case "NotAllowedError":
                case "PermissionDeniedError":
                    return "Camera permission was denied. Allow camera access in the browser and try again.";
                case "NotFoundError":
                case "DevicesNotFoundError":
                    return "No camera was found on this device.";
                case "NotReadableError":
                case "TrackStartError":
                    return "The camera is already in use by another application.";
                case "OverconstrainedError":
                    return "The requested camera settings are not supported. Try another camera.";
                case "SecurityError":
                    return "Camera access requires a secure context (HTTPS or localhost).";
                default:
                    return error.message || "The camera could not be started.";
            }
        }
    }

    global.CameraController = CameraController;
})(window);
