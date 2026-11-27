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
      alt: "PDF to MuseScore Dark Studio interface with live sheet-music preview and Open in MuseScore completion banner",
      caption: "Dark Studio · Live Preview & Completion Banner"
    },
    light: {
      src: "assets/ui-light.png",
      alt: "PDF to MuseScore Light Studio interface with expanded Instrument, Clef, and Transpose drawer",
      caption: "Light Studio · Instrument, Clef & Transpose"
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

  /* Before / After Tabs + Zoom Lightbox */
  var baTabFullScore = document.getElementById("baTabFullScore");
  var baTabBach = document.getElementById("baTabBach");
  var baPanelFullScore = document.getElementById("baPanelFullScore");
  var baPanelBach = document.getElementById("baPanelBach");
  var baCaption = document.getElementById("baCaption");

  function setBeforeAfterExample(choice) {
    var isFullScore = choice !== "bach";
    if (baTabFullScore && baTabBach) {
      baTabFullScore.setAttribute("aria-selected", String(isFullScore));
      baTabBach.setAttribute("aria-selected", String(!isFullScore));
    }
    if (baPanelFullScore) {
      baPanelFullScore.hidden = !isFullScore;
    }
    if (baPanelBach) {
      baPanelBach.hidden = isFullScore;
    }
    if (baCaption) {
      baCaption.textContent = isFullScore
        ? "13-Staff Orchestral Conductor Score · Public Domain (Mutopia Project)"
        : "Solo Cello Part · Public Domain Scan (Bach Cello Suite No. 2)";
    }
  }

  if (baTabFullScore) {
    baTabFullScore.addEventListener("click", function () {
      setBeforeAfterExample("fullscore");
    });
  }
  if (baTabBach) {
    baTabBach.addEventListener("click", function () {
      setBeforeAfterExample("bach");
    });
  }

  var lightbox = document.getElementById("lightbox");
  var lightboxImg = document.getElementById("lightboxImg");
  var zoomButtons = document.querySelectorAll(".ba-zoom-btn");
  var closeZoom = document.getElementById("closeLightbox");

  function closeBox() {
    if (lightbox) {
      lightbox.classList.remove("is-open");
      lightbox.setAttribute("aria-hidden", "true");
    }
  }

  if (zoomButtons.length > 0 && lightbox) {
    zoomButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var nextSrc = btn.getAttribute("data-zoom-src");
        var nextAlt = btn.getAttribute("data-zoom-alt");
        if (lightboxImg && nextSrc) {
          lightboxImg.src = nextSrc;
          if (nextAlt) {
            lightboxImg.alt = nextAlt;
          }
        }
        lightbox.classList.add("is-open");
        lightbox.setAttribute("aria-hidden", "false");
      });
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

  /* Non-blocking latest GitHub release tag badge & versioned .exe link */
  var tagBadge = document.getElementById("releaseTagBadge");
  var heroDownloadBtn = document.getElementById("heroDownloadBtn");
  var bottomDownloadBtn = document.getElementById("bottomDownloadBtn");
  if ((tagBadge || heroDownloadBtn || bottomDownloadBtn) && typeof window.fetch === "function") {
    window
      .fetch("https://api.github.com/repos/tomermayost/pdf2musescore/releases/latest", {
        headers: { Accept: "application/vnd.github+json" }
      })
      .then(function (resp) {
        return resp.ok ? resp.json() : null;
      })
      .then(function (data) {
        if (!data) {
          return;
        }
        if (tagBadge && typeof data.tag_name === "string" && data.tag_name.trim()) {
          tagBadge.textContent = data.tag_name.trim();
        }
        if (Array.isArray(data.assets)) {
          var exeAsset =
            data.assets.find(function (a) {
              return a && typeof a.name === "string" && /setup.*\.exe$/i.test(a.name);
            }) ||
            data.assets.find(function (a) {
              return a && typeof a.name === "string" && /\.exe$/i.test(a.name);
            });
          if (exeAsset && typeof exeAsset.browser_download_url === "string" && exeAsset.browser_download_url) {
            if (heroDownloadBtn) {
              heroDownloadBtn.href = exeAsset.browser_download_url;
            }
            if (bottomDownloadBtn) {
              bottomDownloadBtn.href = exeAsset.browser_download_url;
            }
          }
        }
      })
      .catch(function () {
        /* keep static fallback quietly */
      });
  }
})();
