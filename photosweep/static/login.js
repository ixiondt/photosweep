"use strict";

const $ = (id) => document.getElementById(id);

async function post(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Photosweep": "1" },
    body: JSON.stringify(body),
  });
  let data = null;
  try { data = await resp.json(); } catch { /* not JSON */ }
  return { ok: resp.ok, data };
}

async function boot() {
  const form = $("signin");
  const msg = $("signin-msg");
  try {
    const state = await (await fetch("/api/auth")).json();
    if (state.signedIn) { location.replace("/"); return; }
    if (!state.passwordSet) {
      $("signin-note").textContent =
        "No password is set yet. On the server, run: docker compose exec web python -m photosweep.passwd";
      $("password").disabled = true;
      $("signin-btn").disabled = true;
      return;
    }
  } catch (err) {
    msg.textContent = "Could not reach the server.";
    console.warn("auth state", err.message);
  }
  $("password").focus();
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = $("signin-btn");
    btn.disabled = true;
    msg.textContent = "";
    try {
      const res = await post("/api/login", { password: $("password").value });
      if (res.ok) { location.replace("/"); return; }
      msg.textContent = (res.data && res.data.message) || "Sign-in failed.";
      $("password").select();
    } catch (err) {
      msg.textContent = "Could not reach the server.";
      console.warn("sign-in", err.message);
    } finally {
      btn.disabled = false;
    }
  });
}

document.addEventListener("DOMContentLoaded", boot);
