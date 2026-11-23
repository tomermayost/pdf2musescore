(function () {
  "use strict";

  var THEME_KEY = "pdf2musescore-theme";
  var root = document.documentElement;
  var themeToggle = document.getElementById("themeToggle");
  var previewImg = document.getElementById("heroPreviewImg");
  var previewCaption = document.getElementById("heroPreviewCaption");
  var tabDark = document.getElementById("tabDark");
  var tabLight = document.getElementById("tabLight");

  var PREVIEW_META = {
    dark: {
      src: "assets/ui-dark.png",
      alt: "PDF to MuseScore Dark Studio interface with Advanced AI settings drawer",
      caption: "Dark Studio + Advanced AI Settings"
    },
    light: {
      src: "assets/ui-light.png",
      alt: "PDF to MuseScore Light Studio interface with live sheet-music preview and Open in MuseScore banner",
      caption: "Light Studio + Completion Actions"
    }
  };

  function setPreviewMode(mode) {
    var choice = mode === "light" ? "light" : "dark";
    var info = PREVIEW_META[choice];
    if (previewImg) {
      previewImg.src = info.src;
      previewImg.alt = info.alt;
    }
    if (previewCaption) {
      previewCaption.textContent = info.caption;
    }
    if (tabDark && tabLight) {
      tabDark.setAttribute("aria-selected", String(choice === "dark"));
      tabLight.setAttribute("aria-selected", String(choice === "light"));
    }
  }

  function applyTheme(theme) {
    var resolved = theme === "light" ? "light" : "dark";
    root.setAttribute("data-theme", resolved);
    if (themeToggle) {
      themeToggle.textContent = resolved === "dark" ? "☀ Light" : "🌙 Dark";
      themeToggle.setAttribute(
        "aria-label",
        resolved === "dark" ? "Switch to light theme" : "Switch to dark theme"
      );
    }
    setPreviewMode(resolved);
    try {
      window.localStorage.setItem(THEME_KEY, resolved);
    } catch (_err) {
      /* ignore storage restrictions */
    }
  }

  var initialTheme = "dark";
  try {
    var saved = window.localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") {
      initialTheme = saved;
    }
  } catch (_err) {
    /* default dark */
  }
  applyTheme(initialTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      var current = root.getAttribute("data-theme") === "light" ? "light" : "dark";
      applyTheme(current === "dark" ? "light" : "dark");
    });
  }

  if (tabDark) {
    tabDark.addEventListener("click", function () {
      setPreviewMode("dark");
    });
  }
  if (tabLight) {
    tabLight.addEventListener("click", function () {
      setPreviewMode("light");
    });
  }

  /* Before / After Zoom Lightbox */
  var lightbox = document.getElementById("lightbox");
  var openZoom = document.getElementById("openBeforeAfterZoom");
  var closeZoom = document.getElementById("closeLightbox");

  function closeBox() {
    if (lightbox) {
      lightbox.classList.remove("is-open");
      lightbox.setAttribute("aria-hidden", "true");
    }
  }

  if (openZoom && lightbox) {
    openZoom.addEventListener("click", function () {
      lightbox.classList.add("is-open");
      lightbox.setAttribute("aria-hidden", "false");
    });
    lightbox.addEventListener("click", function (ev) {
      if (ev.target === lightbox) {
        closeBox();
      }
    });
  }
  if (closeZoom) {
    closeZoom.addEventListener("click", closeBox);
  }
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape") {
      closeBox();
    }
  });

  /* Non-blocking latest GitHub release tag badge */
  var tagBadge = document.getElementById("releaseTagBadge");
  if (tagBadge && typeof window.fetch === "function") {
    window
      .fetch("https://api.github.com/repos/tomermayost/pdf2musescore/releases/latest", {
        headers: { Accept: "application/vnd.github+json" }
      })
      .then(function (resp) {
        return resp.ok ? resp.json() : null;
      })
      .then(function (data) {
        if (data && typeof data.tag_name === "string" && data.tag_name.trim()) {
          tagBadge.textContent = data.tag_name.trim();
        }
      })
      .catch(function () {
        /* keep static fallback quietly */
      });
  }
})();
