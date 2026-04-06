const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const statusText = document.getElementById("status-text");
const clearChatBtn = document.getElementById("clear-chat-btn");
const quickButtons = document.querySelectorAll(".quick-btn");

const API_BASE_URL = window.APP_CONFIG?.API_BASE_URL || "";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function addUserMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user";

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;

  wrapper.appendChild(content);
  chatMessages.appendChild(wrapper);
  scrollMessagesToBottom();
}

function addBotMessage(text, sources = []) {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot";

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;

  if (Array.isArray(sources) && sources.length > 0) {
    const sourcesBox = document.createElement("div");
    sourcesBox.className = "sources-box";

    let html = `<div class="sources-title">Sources consultées</div>`;

    for (const source of sources) {
      const title = escapeHtml(source.title || "Source");
      const url = source.url;
      const similarity = typeof source.similarity === "number"
        ? `${Math.round(source.similarity * 100)}%`
        : null;

      if (url) {
        html += `
          <div class="source-item">
            <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${title}</a>
            ${similarity ? `<div class="source-meta">Pertinence : ${similarity}</div>` : ""}
          </div>
        `;
      } else {
        html += `
          <div class="source-item">
            <div>${title}</div>
            ${similarity ? `<div class="source-meta">Pertinence : ${similarity}</div>` : ""}
          </div>
        `;
      }
    }

    sourcesBox.innerHTML = html;
    content.appendChild(sourcesBox);
  }

  wrapper.appendChild(content);
  chatMessages.appendChild(wrapper);
  scrollMessagesToBottom();
}

function addLoadingMessage() {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot";
  wrapper.id = "loading-message";

  const content = document.createElement("div");
  content.className = "message-content loading";
  content.textContent = "Recherche dans la documentation…";

  wrapper.appendChild(content);
  chatMessages.appendChild(wrapper);
  scrollMessagesToBottom();
}

function removeLoadingMessage() {
  const loading = document.getElementById("loading-message");
  if (loading) {
    loading.remove();
  }
}

function scrollMessagesToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setLoadingState(isLoading) {
  sendBtn.disabled = isLoading;
  chatInput.disabled = isLoading;
  statusText.textContent = isLoading ? "Réponse en cours..." : "Prêt";
}

async function sendQuestion(question) {
  const trimmedQuestion = question.trim();

  if (!trimmedQuestion) return;

  if (!API_BASE_URL) {
    addBotMessage("Configuration manquante : l’URL du backend n’est pas définie.");
    return;
  }

  addUserMessage(trimmedQuestion);
  chatInput.value = "";
  addLoadingMessage();
  setLoadingState(true);

  try {
    const response = await fetch(
      `${API_BASE_URL}/ask?question=${encodeURIComponent(trimmedQuestion)}`,
      {
        method: "GET",
        headers: {
          "Accept": "application/json"
        }
      }
    );

    const data = await response.json();
    removeLoadingMessage();

    if (data.status === "ok") {
      addBotMessage(data.answer || "Aucune réponse reçue.", data.sources || []);
    } else {
      addBotMessage(`Une erreur est survenue : ${data.message || "inconnue"}`);
    }
  } catch (error) {
    removeLoadingMessage();
    addBotMessage("Impossible de contacter l’assistant pour le moment.");
  } finally {
    setLoadingState(false);
    chatInput.focus();
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendQuestion(chatInput.value);
});

quickButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    await sendQuestion(button.textContent);
  });
});

clearChatBtn.addEventListener("click", () => {
  chatMessages.innerHTML = `
    <div class="message bot">
      <div class="message-content">
        Bonjour 👋<br><br>
        Je peux vous aider sur les jours fériés, les contrats, les absences, le préavis, la paie et d'autres sujets RH, sur base de la documentation disponible.
      </div>
    </div>
  `;
  chatInput.focus();
});