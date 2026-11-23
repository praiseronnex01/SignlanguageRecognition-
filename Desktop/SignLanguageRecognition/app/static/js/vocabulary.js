/* ==========================================================================
   Vocabulary management: search, filter and CRUD for MSL gesture entries.
   ========================================================================== */
(function () {
    "use strict";

    const grid = document.getElementById("vocabulary-grid");
    if (!grid) return;

    const searchInput = document.getElementById("vocabulary-search");
    const categorySelect = document.getElementById("vocabulary-category");
    const handSelect = document.getElementById("vocabulary-hands");
    const emptyState = document.getElementById("vocabulary-empty");

    const modal = document.getElementById("gesture-modal");
    const modalTitle = document.getElementById("gesture-modal-title");
    const form = document.getElementById("gesture-form");
    const formError = document.getElementById("gesture-form-error");
    const idField = document.getElementById("gesture-id");
    const labelField = document.getElementById("gesture-label");
    const meaningField = document.getElementById("gesture-meaning");
    const categoryField = document.getElementById("gesture-category");
    const handField = document.getElementById("gesture-hands");
    const descriptionField = document.getElementById("gesture-description");
    const imageField = document.getElementById("gesture-image");
    const videoField = document.getElementById("gesture-video");
    const sourceField = document.getElementById("gesture-source");
    const notesField = document.getElementById("gesture-notes");

    /* ------------------------------------------------------------ Render */
    function renderGestures(gestures) {
        grid.innerHTML = "";
        if (!gestures.length) {
            emptyState.classList.remove("hidden");
            return;
        }
        emptyState.classList.add("hidden");

        gestures.forEach(function (gesture) {
            const card = document.createElement("article");
            card.className = "card flex flex-col overflow-hidden";

            const imageHtml = gesture.reference_image
                ? '<div class="h-40 w-full overflow-hidden bg-surface">' +
                  '<img src="' + window.MSL.escapeHtml(gesture.reference_image) + '" alt="" ' +
                  'class="h-full w-full object-cover" loading="lazy"></div>'
                : "";

            let badge;
            if (gesture.is_validated && gesture.is_active) {
                badge = '<span class="badge badge-ok shrink-0"><i class="fa-solid fa-circle-check"></i> Validated</span>';
            } else if (!gesture.is_active) {
                badge = '<span class="badge badge-muted shrink-0">Inactive</span>';
            } else {
                badge = '<span class="badge badge-warn shrink-0"><i class="fa-solid fa-hourglass-half"></i> Candidate</span>';
            }

            const description = gesture.description
                ? '<p class="line-clamp-3 text-sm leading-relaxed text-navy-muted">' +
                  window.MSL.escapeHtml(gesture.description) + "</p>"
                : '<p class="text-sm italic text-slate-400">No description has been recorded yet.</p>';

            const videoHtml = gesture.reference_video
                ? '<a href="' + window.MSL.escapeHtml(gesture.reference_video) + '" target="_blank" rel="noopener noreferrer" ' +
                  'class="inline-flex items-center gap-2 text-xs font-medium text-accent hover:underline">' +
                  '<i class="fa-solid fa-circle-play"></i> Reference video</a>'
                : "";

            const adminActions = isAdmin()
                ? '<div class="card-footer flex flex-wrap gap-2">' +
                    '<button type="button" class="btn btn-secondary btn-sm js-edit-gesture" data-id="' + gesture.id + '" ' +
                    'data-label="' + window.MSL.escapeHtml(gesture.label) + '" ' +
                    'data-meaning="' + window.MSL.escapeHtml(gesture.english_meaning) + '" ' +
                    'data-category="' + window.MSL.escapeHtml(gesture.category || "") + '" ' +
                    'data-description="' + window.MSL.escapeHtml(gesture.description || "") + '" ' +
                    'data-hand-count="' + (gesture.hand_count || 1) + '" ' +
                    'data-reference-image="' + window.MSL.escapeHtml(gesture.reference_image || "") + '" ' +
                    'data-reference-video="' + window.MSL.escapeHtml(gesture.reference_video || "") + '" ' +
                    'data-notes="' + window.MSL.escapeHtml(gesture.validation_notes || "") + '" ' +
                    'data-source="' + window.MSL.escapeHtml(gesture.source_reference || "") + '" ' +
                    'data-validated="' + (gesture.is_validated ? "1" : "0") + '" ' +
                    'data-active="' + (gesture.is_active ? "1" : "0") + '">' +
                    '<i class="fa-solid fa-pen"></i> Edit</button>' +
                    (gesture.is_validated
                        ? '<button type="button" class="btn btn-secondary btn-sm js-unvalidate-gesture" data-id="' + gesture.id + '">' +
                          '<i class="fa-solid fa-rotate-left"></i> Unvalidate</button>'
                        : '<button type="button" class="btn btn-secondary btn-sm js-validate-gesture" data-id="' + gesture.id + '">' +
                          '<i class="fa-solid fa-circle-check"></i> Mark validated</button>') +
                    '<button type="button" class="btn btn-secondary btn-sm js-toggle-gesture" data-id="' + gesture.id + '" ' +
                    'data-active="' + (gesture.is_active ? "1" : "0") + '">' +
                    '<i class="fa-solid fa-toggle-' + (gesture.is_active ? "on" : "off") + '"></i> ' +
                    (gesture.is_active ? "Deactivate" : "Activate") + "</button></div>"
                : "";

            card.innerHTML =
                imageHtml +
                '<div class="card-body flex-1 space-y-3">' +
                    '<div class="flex items-start justify-between gap-2">' +
                        '<div class="min-w-0"><h3 class="truncate text-base font-semibold text-navy">' +
                        window.MSL.escapeHtml(gesture.label) + "</h3>" +
                        '<p class="truncate text-sm text-slate-600">' +
                        window.MSL.escapeHtml(gesture.english_meaning) + "</p></div>" + badge +
                    "</div>" +
                    '<div class="flex flex-wrap gap-2 text-xs">' +
                        '<span class="badge badge-muted"><i class="fa-solid fa-tag"></i> ' +
                        window.MSL.escapeHtml(gesture.category || "General") + "</span>" +
                        '<span class="badge badge-muted"><i class="fa-solid fa-hand"></i> ' +
                        (gesture.hand_count || 1) + " hand" + ((gesture.hand_count || 1) > 1 ? "s" : "") + "</span>" +
                        '<span class="badge badge-muted"><i class="fa-solid fa-database"></i> ' +
                        (gesture.sample_count || 0) + " samples</span>" +
                    "</div>" +
                    description + videoHtml +
                    '<p class="text-[11px] text-slate-400">Last updated ' +
                    window.MSL.formatDate(gesture.updated_at) + " &middot; recognition " +
                    (gesture.recognition_available ? "available" : "unavailable") + "</p>" +
                "</div>" + adminActions;

            grid.appendChild(card);
        });

        bindActionButtons();
    }

    function isAdmin() {
        return document.body.getAttribute("data-role") === "admin";
    }

    /* ------------------------------------------------------------- Fetch */
    let fetchTimer = null;

    function loadGestures() {
        const params = new URLSearchParams();
        const search = (searchInput.value || "").trim();
        const category = categorySelect.value;
        const handCount = handSelect.value;
        if (search) params.set("search", search);
        if (category && category !== "all") params.set("category", category);
        if (handCount) params.set("hand_count", handCount);
        params.set("include_inactive", "1");

        window.MSL.api
            .get("/vocabulary/api/vocabulary?" + params.toString())
            .then(function (response) {
                renderGestures(response.gestures || []);
            })
            .catch(function (error) {
                window.MSL.toast(error.message, "bad");
            });
    }

    searchInput.addEventListener("input", function () {
        window.clearTimeout(fetchTimer);
        fetchTimer = window.setTimeout(loadGestures, 220);
    });
    categorySelect.addEventListener("change", loadGestures);
    handSelect.addEventListener("change", loadGestures);

    /* ------------------------------------------------------------ Modal */
    function openModalFor(gesture) {
        formError.classList.add("hidden");
        idField.value = gesture ? gesture.id : "";
        modalTitle.textContent = gesture ? "Edit gesture" : "Add gesture";
        labelField.value = gesture ? gesture.label : "";
        meaningField.value = gesture ? gesture.english_meaning : "";
        categoryField.value = gesture ? gesture.category || "General" : "General";
        handField.value = gesture ? gesture.hand_count || 1 : 1;
        descriptionField.value = gesture ? gesture.description || "" : "";
        videoField.value = gesture ? gesture.reference_video || "" : "";
        sourceField.value = gesture ? gesture.source_reference || "" : "";
        notesField.value = gesture ? gesture.validation_notes || "" : "";
        imageField.value = "";
        window.MSL.openModal("gesture-modal");
        labelField.focus();
    }

    const addButton = document.getElementById("btn-add-gesture");
    if (addButton) {
        addButton.addEventListener("click", function () {
            openModalFor(null);
        });
    }

    document.getElementById("gesture-modal-close").addEventListener("click", function () {
        window.MSL.closeModal("gesture-modal");
    });
    document.getElementById("gesture-cancel").addEventListener("click", function () {
        window.MSL.closeModal("gesture-modal");
    });
    modal.addEventListener("click", function (event) {
        if (event.target === modal) window.MSL.closeModal("gesture-modal");
    });

    form.addEventListener("submit", async function (event) {
        event.preventDefault();
        formError.classList.add("hidden");

        const id = idField.value;
        const formData = new FormData();
        formData.append("label", labelField.value.trim());
        formData.append("english_meaning", meaningField.value.trim());
        formData.append("category", categoryField.value.trim() || "General");
        formData.append("description", descriptionField.value.trim());
        formData.append("hand_count", handField.value);
        formData.append("reference_video", videoField.value.trim());
        formData.append("source_reference", sourceField.value.trim());
        formData.append("validation_notes", notesField.value.trim());
        if (imageField.files && imageField.files[0]) {
            formData.append("reference_image", imageField.files[0]);
        }

        const submitButton = document.getElementById("gesture-submit");
        submitButton.disabled = true;

        try {
            if (id) {
                await window.MSL.request("/vocabulary/api/vocabulary/" + id, {
                    method: "PUT",
                    body: formData
                });
                window.MSL.toast("The gesture was updated.", "ok");
            } else {
                await window.MSL.request("/vocabulary/api/vocabulary", {
                    method: "POST",
                    body: formData
                });
                window.MSL.toast("The gesture was added as a candidate.", "ok");
            }
            window.MSL.closeModal("gesture-modal");
            loadGestures();
        } catch (error) {
            formError.textContent = error.message;
            formError.classList.remove("hidden");
        } finally {
            submitButton.disabled = false;
        }
    });

    /* ---------------------------------------------------------- Actions */
    function bindActionButtons() {
        grid.querySelectorAll(".js-edit-gesture").forEach(function (button) {
            button.addEventListener("click", function () {
                openModalFor({
                    id: button.dataset.id,
                    label: button.dataset.label,
                    english_meaning: button.dataset.meaning,
                    category: button.dataset.category,
                    description: button.dataset.description,
                    hand_count: button.dataset.handCount,
                    reference_image: button.dataset.referenceImage,
                    reference_video: button.dataset.referenceVideo,
                    validation_notes: button.dataset.notes,
                    source_reference: button.dataset.source
                });
            });
        });

        grid.querySelectorAll(".js-validate-gesture").forEach(function (button) {
            button.addEventListener("click", async function () {
                if (!window.confirm(
                    "Mark this sign as a validated MSL sign?\n\n" +
                    "Only do this after the sign has been verified against the MSL dictionary or with a competent signer."
                )) return;
                try {
                    await window.MSL.api.put("/vocabulary/api/vocabulary/" + button.dataset.id, {
                        is_validated: true
                    });
                    window.MSL.toast("The sign was marked as validated.", "ok");
                    loadGestures();
                } catch (error) {
                    window.MSL.toast(error.message, "bad");
                }
            });
        });

        grid.querySelectorAll(".js-unvalidate-gesture").forEach(function (button) {
            button.addEventListener("click", async function () {
                try {
                    await window.MSL.api.put("/vocabulary/api/vocabulary/" + button.dataset.id, {
                        is_validated: false
                    });
                    window.MSL.toast("The sign was returned to the candidate list.", "info");
                    loadGestures();
                } catch (error) {
                    window.MSL.toast(error.message, "bad");
                }
            });
        });

        grid.querySelectorAll(".js-toggle-gesture").forEach(function (button) {
            button.addEventListener("click", async function () {
                const activate = button.dataset.active !== "1";
                try {
                    await window.MSL.api.put("/vocabulary/api/vocabulary/" + button.dataset.id, {
                        is_active: activate
                    });
                    window.MSL.toast(
                        activate ? "The gesture was activated." : "The gesture was deactivated.",
                        "ok"
                    );
                    loadGestures();
                } catch (error) {
                    window.MSL.toast(error.message, "bad");
                }
            });
        });
    }

    bindActionButtons();
    loadGestures();
})();
