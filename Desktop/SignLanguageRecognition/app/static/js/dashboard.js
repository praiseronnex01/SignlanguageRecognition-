/* ==========================================================================
   Dashboard: refresh the headline statistics from the API so the page can be
   refreshed without a full page reload.
   ========================================================================== */
(function () {
    "use strict";

    const targets = {
        gestures: document.getElementById("stat-gestures"),
        samples: document.getElementById("stat-samples"),
        participants: document.getElementById("stat-participants"),
        recognitions: document.getElementById("stat-recognitions"),
        users: document.getElementById("stat-users"),
        sessions: document.getElementById("stat-sessions")
    };

    if (!Object.keys(targets).some(function (key) { return targets[key]; })) {
        return;
    }

    function setText(element, value) {
        if (element && value !== null && value !== undefined) {
            element.textContent = value;
        }
    }

    window.MSL.api
        .get("/api/dashboard/stats")
        .then(function (response) {
            const stats = (response && response.stats) || {};
            if (!stats.gestures) return;

            setText(targets.gestures, stats.gestures.total);
            setText(targets.samples, stats.dataset.total_samples);
            setText(targets.participants, stats.participants.total);
            setText(targets.recognitions, stats.recognition.total);
            setText(targets.users, stats.users.total);
            setText(targets.sessions, stats.dataset.session_count);

            const accuracy = document.getElementById("stat-accuracy");
            const model = document.getElementById("stat-model");
            if (accuracy && stats.model.latest && stats.model.latest.accuracy !== null) {
                accuracy.textContent = (stats.model.latest.accuracy * 100).toFixed(1) + "%";
            }
            if (model && stats.model.published) {
                model.textContent = stats.model.published.version;
            }
        })
        .catch(function () {
            /* Keep the server-rendered values when the refresh fails. */
        });
})();
