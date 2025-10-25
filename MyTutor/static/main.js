const steps = ["s1","s2","s3","s4"];
const el = id => document.getElementById(id);

function show(id) {
  steps.forEach(s => el(s).classList.remove("active"));
  el(id).classList.add("active");
}

async function api(path, opts={}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts
  });
  if (!res.ok) {
    const err = await res.json().catch(()=>({error:"Unknown"}));
    throw new Error(JSON.stringify(err));
  }
  return res.json();
}

async function loadSubjects() {
  const { subjects } = await api("/subjects");
  const s = el("subject");
  s.innerHTML = "";
  subjects.forEach(sub => {
    const opt = document.createElement("option");
    opt.value = sub; opt.textContent = sub;
    s.appendChild(opt);
  });
}

async function startFlow() {
  const subject = el("subject").value;
  if (!subject) return alert("Pick a subject first.");
  const data = await api("/start", { method:"POST", body: JSON.stringify({ subject }) });
  const form = el("diagForm");
  form.innerHTML = "";
  data.diagnostic.forEach((q, i) => {
    form.innerHTML += `<div class="q"><strong>Q${i+1}.</strong> ${q.question}<br/><input name="q${i}" /></div>`;
  });
  show("s2");
}

async function submitDiagnostic(e) {
  e.preventDefault();
  const next = await api("/submit_diagnostic", { method:"POST", body: JSON.stringify({}) });
  el("diffPill").textContent = `Difficulty: ${next.difficulty}`;
  const form = el("quizForm");
  form.innerHTML = "";
  next.questions.forEach((q, i) => {
    form.innerHTML += `<div class="q"><strong>Q${i+1}.</strong> ${q.question}<br/>` +
      q.options.map(opt => `<label><input type="radio" name="q${i}" value="${opt}"> ${opt}</label>`).join("<br/>") +
      `</div>`;
  });
  show("s3");
}

async function submitQuiz(e) {
  e.preventDefault();
  const form = el("quizForm");
  const formData = new FormData(form);
  const answers = {};
  for (const [k,v] of formData.entries()) answers[k] = v;

  const res = await api("/submit_quiz", { method:"POST", body: JSON.stringify({ answers }) });
  el("results").innerHTML = `
    <div class="kpi">
      <span class="pill">Score: ${res.score}/${res.total}</span>
      <span class="pill">Accuracy: ${Math.round(res.accuracy*100)}%</span>
      <span class="pill">Reward: ${res.reward}</span>
      <span class="pill">Next: ${res.next}</span>
    </div>
  `;

  const hintsDiv = el("hintsSection");
  hintsDiv.innerHTML = "";
  if (res.hints && res.hints.length) {
    hintsDiv.innerHTML = `<h4>🧠 Tips to Review</h4>` + res.hints.map(h => `<div class="hint">💡 ${h}</div>`).join("");
  }

  show("s4");
}

async function nextQuiz() {
  const data = await api("/quiz");
  el("diffPill").textContent = `Difficulty: ${data.difficulty}`;
  const form = el("quizForm");
  form.innerHTML = "";
  data.questions.forEach((q, i) => {
    form.innerHTML += `<div class="q"><strong>Q${i+1}.</strong> ${q.question}<br/>` +
      q.options.map(opt => `<label><input type="radio" name="q${i}" value="${opt}"> ${opt}</label>`).join("<br/>") +
      `</div>`;
  });
  show("s3");
}

el("startBtn").onclick = startFlow;
el("diagNext").onclick = submitDiagnostic;
el("quizSubmit").onclick = submitQuiz;
el("nextQuizBtn").onclick = nextQuiz;

loadSubjects().catch(console.error);
