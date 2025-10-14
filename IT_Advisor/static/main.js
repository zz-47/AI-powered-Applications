const askBtn = document.getElementById("askBtn");
const promptBox = document.getElementById("prompt");
const styleSelect = document.getElementById("style");
const outputBox = document.getElementById("output");

askBtn.addEventListener("click", async () => {
  const prompt = promptBox.value.trim();
  const style = styleSelect.value;

  if (!prompt) {
    outputBox.textContent = "⚠️ Please enter a question.";
    return;
  }

  outputBox.textContent = "⏳ Thinking...";

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, style }),
    });
    const data = await res.json();
    typeWriter(data.answer, outputBox, 10);
  } catch (err) {
    console.error(err);
    outputBox.textContent = "❌ Error contacting backend.";
  }
});

function typeWriter(text, el, delay = 10) {
  el.textContent = "";
  let i = 0;
  (function type() {
    if (i < text.length) {
      el.textContent += text.charAt(i++);
      setTimeout(type, delay);
    }
  })();
}
