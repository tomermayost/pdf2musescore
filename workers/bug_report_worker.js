/**
 * Cloudflare Worker: anonymous bug-report proxy -> GitHub Issues with deduplication.
 * Secrets: GITHUB_TOKEN (fine-grained PAT with Issues: Read & Write on target repo).
 * Vars:    GITHUB_REPO (optional, defaults to "tomermayost/pdf2musescore").
 */

const RATE_WINDOW_MS = 10 * 60 * 1000;
const RATE_MAX_REQ = 5;
const MAX_BODY_BYTES = 65_536;
const ipHits = new Map();

const json = (data, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });

function checkRateLimit(ip, now = Date.now()) {
  for (const [k, v] of ipHits) if (now >= v.resetAt) ipHits.delete(k);
  const entry = ipHits.get(ip);
  if (!entry || now >= entry.resetAt) {
    ipHits.set(ip, { count: 1, resetAt: now + RATE_WINDOW_MS });
    return true;
  }
  if (entry.count >= RATE_MAX_REQ) return false;
  entry.count += 1;
  return true;
}

async function ghFetch(url, token, init = {}) {
  return fetch(url, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "User-Agent": "pdf2musescore-bug-worker/1.0",
      "X-GitHub-Api-Version": "2022-11-28",
      ...(init.body ? { "Content-Type": "application/json" } : {}),
    },
  });
}

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return json({ error: "Method not allowed" }, 405);
    if (!env.GITHUB_TOKEN) return json({ error: "Server misconfigured" }, 500);

    const ip = request.headers.get("CF-Connecting-IP") || "unknown";
    if (!checkRateLimit(ip)) return json({ error: "Too many reports, try again later" }, 429);

    const raw = await request.text();
    if (raw.length > MAX_BODY_BYTES) return json({ error: "Payload too large" }, 413);

    let payload;
    try {
      payload = JSON.parse(raw);
    } catch {
      return json({ error: "Invalid JSON" }, 400);
    }

    const fp = String(payload?.fingerprint || "").trim();
    const title = String(payload?.title || "").trim().slice(0, 180);
    const body = String(payload?.body || "").trim().slice(0, 30_000);
    if (!/^[a-f0-9]{8,64}$/i.test(fp) || !title || !body) {
      return json({ error: "Missing or invalid fingerprint, title, or body" }, 400);
    }

    const repo = String(env.GITHUB_REPO || "tomermayost/pdf2musescore").trim();
    const searchQ = `repo:${repo} is:issue is:open "fp:${fp}" in:body`;
    const searchRes = await ghFetch(
      `https://api.github.com/search/issues?q=${encodeURIComponent(searchQ)}&per_page=1`,
      env.GITHUB_TOKEN
    );

    if (searchRes.ok) {
      const found = await searchRes.json();
      const existing = found?.items?.[0];
      if (existing?.number) {
        const ver = String(payload.app_version || "unknown").slice(0, 32);
        const osInfo = String(payload.os || "Windows").slice(0, 80);
        const note = String(payload.user_note || "").trim().slice(0, 500);
        const comment = `+1 occurrence (v${ver}, ${osInfo})${note ? `\n\n> ${note}` : ""}`;
        await ghFetch(`https://api.github.com/repos/${repo}/issues/${existing.number}/comments`, env.GITHUB_TOKEN, {
          method: "POST",
          body: JSON.stringify({ body: comment }),
        });
        return json({ ok: true, duplicate: true, issue_number: existing.number, issue_url: existing.html_url }, 200);
      }
    }

    const finalBody = body.includes(`fp:${fp}`) ? body : `${body}\n\n<!-- fp:${fp} -->`;
    const createRes = await ghFetch(`https://api.github.com/repos/${repo}/issues`, env.GITHUB_TOKEN, {
      method: "POST",
      body: JSON.stringify({ title, body: finalBody, labels: ["bug", "auto-report"] }),
    });
    if (!createRes.ok) return json({ error: `GitHub API error (${createRes.status})` }, 502);

    const created = await createRes.json();
    return json({ ok: true, duplicate: false, issue_number: created.number, issue_url: created.html_url }, 201);
  },
};
