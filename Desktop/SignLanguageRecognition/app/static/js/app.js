/* ==========================================================================
   Shared front-end helpers for the MSL recognition system.
   ========================================================================== */
(function () {
    "use strict";

    const meta = document.querySelector('meta[name="csrf-token"]');
    const CSRF_TOKEN = meta ? meta.getAttribute("content") : "";

    /* ------------------------------------------------------------- HTTP */
    async function request(url, options) {
        const opts = Object.assign({ credentials: "same-origin" }, options || {});
        opts.headers = Object.assign(
            {
                Accept: "application/json",
                "X-CSRFToken": CSRF_TOKEN,
                "X-Requested-With": "XMLHttpRequest"
            },
            opts.headers || {}
        );

        if (opts.body && !(opts.body instanceof FormData) && !opts.headers["Content-Type"]) {
            opts.headers["Content-Type"] = "application/json";
        }

        let response;
        try {
            response = await fetch(url, opts);
        } catch (error) {
            throw new Error("The server could not be reached. Check your connection.");
        }

        let payload = null;
        const contentType = response.headers.get("Content-Type") || "";
        if (contentType.indexOf("application/json") !== -1) {
            try {
                payload = await response.json();
            } catch (error) {
                payload = null;
            }
        }

        if (!response.ok) {
            const message =
                (payload && (payload.error || payload.message)) ||
                `Request failed with status ${response.status}.`;
            const failure = new Error(message);
            failure.status = response.status;
            failure.payload = payload;
            throw failure;
        }
        return payload;
    }

    const api = {
        get(url) {
            return request(url, { method: "GET" });
        },
        post(url, body) {
            return request(url, {
                method: "POST",
                body: body instanceof FormData ? body : JSON.stringify(body || {})
            });
        },
        put(url, body) {
            return request(url, {
                method: "PUT",
                body: body instanceof FormData ? body : JSON.stringify(body || {})
            });
        },
        del(url, body) {
            return request(url, {
                method: "DELETE",
                body: body === undefined ? undefined : JSON.stringify(body)
            });
        }
    };

    /* ----------------------------------------------------------- Toasts */
    function toast(message, tone) {
        let stack = document.querySelector(".toast-stack");
        if (!stack) {
            stack = document.createElement("div");
            stack.className = "toast-stack";
            document.body.appendChild(stack);
        }
        const icon =
            tone === "ok"
                ? "fa-circle-check"
                : tone === "bad"
                ? "fa-circle-exclamation"
                : "fa-circle-info";
        const element = document.createElement("div");
        element.className = "toast toast-" + (tone || "info");
        element.innerHTML =
            '<i class="fa-solid ' + icon + ' mt-0.5"></i>' +
            '<span class="flex-1"></span>' +
            '<button type="button" class="text-slate-400 hover:text-navy"><i class="fa-solid fa-xmark"></i></button>';
        element.querySelector("span").textContent = message;
        element.querySelector("button").addEventListener("click", () => element.remove());
        stack.appendChild(element);
        window.setTimeout(() => element.remove(), 5200);
    }

    /* --------------------------------------------------------- Formatting */
    function escapeHtml(value) {
        if (value === null || value === undefined) return "";
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatDateTime(value) {
        if (!value) return "-";
        const date = new Date(value);
        if (isNaN(date.getTime())) return String(value);
        return date.toLocaleString(undefined, {
            year: "numeric",
            month: "short",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit"
        });
    }

    function formatDate(value) {
        if (!value) return "-";
        const date = new Date(value);
        if (isNaN(date.getTime())) return String(value);
        return date.toLocaleDateString(undefined, {
            year: "numeric",
            month: "short",
            day: "2-digit"
        });
    }

    function formatPercent(value, digits) {
        if (value === null || value === undefined) return "-";
        return (Number(value) * 100).toFixed(digits === undefined ? 1 : digits) + "%";
    }

    function percentBadge(value) {
        if (value === null || value === undefined) return '<span class="badge badge-muted">n/a</span>';
        const number = Number(value) * 100;
        let tone = "badge-muted";
        if (number >= 80) tone = "badge-ok";
        else if (number >= 55) tone = "badge-info";
        else if (number >= 35) tone = "badge-warn";
        else tone = "badge-bad";
        return '<span class="badge ' + tone + '">' + number.toFixed(1) + "%</span>";
    }

    /* ------------------------------------------------------------ Modal */
    function openModal(id) {
        const element = document.getElementById(id);
        if (element) element.classList.remove("hidden");
    }

    function closeModal(id) {
        const element = document.getElementById(id);
        if (element) element.classList.add("hidden");
    }

    function confirmAction(message, onConfirm) {
        if (window.confirm(message)) {
            onConfirm();
        }
    }

    /* -------------------------------------------------------- Navigation */
    function setupSidebar() {
        const toggle = document.getElementById("sidebar-toggle");
        const sidebar = document.getElementById("sidebar");
        const overlay = document.getElementById("sidebar-overlay");
        if (!toggle || !sidebar) return;

        const close = () => {
            sidebar.classList.add("-translate-x-full");
            if (overlay) overlay.classList.add("hidden");
        };
        const open = () => {
            sidebar.classList.remove("-translate-x-full");
            if (overlay) overlay.classList.remove("hidden");
        };

        toggle.addEventListener("click", () => {
            if (sidebar.classList.contains("-translate-x-full")) {
                open();
            } else {
                close();
            }
        });
        if (overlay) overlay.addEventListener("click", close);
    }

    function setupUserMenu() {
        const button = document.getElementById("user-menu-button");
        const panel = document.getElementById("user-menu-panel");
        if (!button || !panel) return;
        button.addEventListener("click", (event) => {
            event.stopPropagation();
            panel.classList.toggle("hidden");
        });
        document.addEventListener("click", (event) => {
            if (!panel.contains(event.target)) panel.classList.add("hidden");
        });
    }

    function setupFlashDismiss() {
        document.querySelectorAll(".flash-dismiss").forEach((button) => {
            button.addEventListener("click", () => {
                const message = button.closest(".flash-message");
                if (message) message.remove();
            });
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        setupSidebar();
        setupUserMenu();
        setupFlashDismiss();
    });

    window.MSL = {
        api,
        request,
        toast,
        escapeHtml,
        formatDate,
        formatDateTime,
        formatPercent,
        percentBadge,
        openModal,
        closeModal,
        confirmAction,
        csrfToken: CSRF_TOKEN
    };
})();
