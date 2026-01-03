// Minimal UI logic: POST /ask and render answer + citations (+ debug details).
// No external dependencies, no frameworks.

function el(id) {
  return document.getElementById(id);
}

function setLoading(isLoading) {
  const btn = el("askBtn");
  const status = el("status");
  btn.disabled = isLoading;
  status.textContent = isLoading ? "Asking..." : "";
}

function showResult() {
  el("result").classList.remove("hidden");
}

function clearResult() {
  el("answer").textContent = "";
  el("citations").innerHTML = "";
  el("debugRetrieved").innerHTML = "";
  el("debugDetails").classList.add("hidden");
}

function renderCitations(citations) {
  const ul = el("citations");
  ul.innerHTML = "";

  for (const c of citations || []) {
    const li = document.createElement("li");
    const title = document.createElement("div");
    title.textContent = `${c.chunk_id}`;

    const snippet = document.createElement("div");
    snippet.className = "snippet";
    snippet.textContent = c.snippet || "";

    li.appendChild(title);
    li.appendChild(snippet);
    ul.appendChild(li);
  }
}

function renderDebug(debug) {
  if (!debug || !debug.retrieved) {
    el("debugDetails").classList.add("hidden");
    return;
  }

  const container = el("debugRetrieved");
  container.innerHTML = "";

  for (const r of debug.retrieved) {
    const pre = document.createElement("pre");
    pre.textContent = `[${r.chunk_id}] (score=${r.score})\n${r.text}`;
    container.appendChild(pre);
  }

  el("debugDetails").classList.remove("hidden");
}

async function ask() {
  const question = el("question").value.trim();
  const topK = Number(el("top_k").value || "4");
  const debug = el("debug").checked;
  const agentMode = el("agent_mode").checked;

  if (!question) {
    el("status").textContent = "Please enter a question.";
    return;
  }

  setLoading(true);
  clearResult();

  try {
    const endpoint = agentMode ? "/ask_agent" : "/ask";
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, top_k: topK, debug: debug }),
    });

    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data?.detail || `Request failed with status ${resp.status}`);
    }

    el("answer").textContent = data.answer || "";
    renderCitations(data.citations || []);
    renderDebug(data.debug);
    showResult();
  } catch (err) {
    el("status").textContent = `Error: ${err.message || String(err)}`;
  } finally {
    setLoading(false);
  }
}

function wireEvents() {
  el("askBtn").addEventListener("click", ask);
  el("question").addEventListener("keydown", (e) => {
    // Ctrl+Enter submits the question.
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      ask();
    }
  });
}

wireEvents();


