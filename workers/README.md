# PDF to MuseScore — Bug Report Cloudflare Worker

Minimal serverless proxy that accepts privacy-sanitized JSON bug reports from `pdf2musescore`, deduplicates open GitHub Issues by error `fingerprint`, and returns the resulting GitHub Issue URL without embedding any GitHub token in the desktop app.

## 1. Create a Fine-Grained GitHub Token

1. Open **GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens**.
2. Create a token scoped **only** to the `tomermayost/pdf2musescore` repository.
3. Grant **Repository permissions**:
   - **Issues**: Read and write (the only write permission required).
   - **Metadata**: Read-only (automatically included).
4. Copy the generated token (you will store it as a Cloudflare Secret below).

## 2. Create the `auto-report` Label on GitHub

In your repository's **Issues → Labels**, create a label named `auto-report` (GitHub will auto-create it on first issue if missing, though creating it explicitly lets you pick a color).

## 3. Deploy to Cloudflare Workers

Create a temporary `wrangler.toml` alongside `bug_report_worker.js` (or deploy from the Cloudflare Dashboard):

```toml
name = "pdf2musescore-bug-report"
main = "bug_report_worker.js"
compatibility_date = "2026-01-01"

[vars]
GITHUB_REPO = "tomermayost/pdf2musescore"
```

Deploy and upload the secret token:

```powershell
npx wrangler secret put GITHUB_TOKEN
npx wrangler deploy
```

> **Security:** Never commit `GITHUB_TOKEN` to git or embed it in Python code. It lives exclusively in Cloudflare encrypted Worker secrets.

## 4. Recommended Cloudflare Edge Rate Limiting

`bug_report_worker.js` includes a lightweight per-isolate IP limit (5 reports per 10 minutes). For strict global rate limiting across all Cloudflare edge POPs, add a free **Security → WAF → Rate limiting rule** on your Worker route (e.g., max 10 `POST` requests per IP per hour).

## 5. Configure `pdf2musescore`

Once deployed (e.g. `https://pdf2musescore-bug-report.<subdomain>.workers.dev`), wire it in one of three ways:

- **Default for all users:** Set `DEFAULT_BUG_REPORT_URL` in `src/pdf2musescore/config.py`.
- **Per-user config (`%APPDATA%\pdf2musescore\config.toml`):**
  ```toml
  bug_report_url = "https://pdf2musescore-bug-report.<subdomain>.workers.dev"
  ```
- **Environment override:** `PDF2MSCZ_BUG_REPORT_URL=https://...`

When `bug_report_url` is blank/unset, the app's **Send Bug Report** button automatically falls back to opening a pre-filled GitHub Issue page in the user's default browser.
