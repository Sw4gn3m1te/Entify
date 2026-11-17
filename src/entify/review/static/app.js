"use strict";

const COLORS = [
  "#ffd400", "#00e5ff", "#ff4fd8", "#7cfc00", "#ff9f1a", "#b388ff",
  "#ff6b6b", "#ffffff", "#c6ff00", "#80cbc4", "#ffab91", "#f48fb1",
];
const HANDLE = 6;        // handle size in screen pixels
const MIN_BOX = 3;       // smallest new box in image pixels

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const el = (id) => document.getElementById(id);

const state = {
  session: null,
  index: -1,
  image: null,
  objects: [],          // {box: [x1, y1, x2, y2], category, source, score, rejected}
  complete: false,
  dirty: false,
  selected: -1,
  undo: [],
  view: { scale: 1, x: 0, y: 0 },
  drag: null,
  space: false,
  busy: false,
};

// ---------- server ----------

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function setStatus(text) {
  el("status").textContent = text;
}

// ---------- coordinates ----------

const toScreen = (x, y) => [x * state.view.scale + state.view.x, y * state.view.scale + state.view.y];
const toImage = (x, y) => [(x - state.view.x) / state.view.scale, (y - state.view.y) / state.view.scale];

function pointer(event) {
  const rect = canvas.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

function clampBox(box) {
  const { width, height } = currentImage();
  const [x1, y1, x2, y2] = box;
  return [
    Math.max(0, Math.min(x1, x2)), Math.max(0, Math.min(y1, y2)),
    Math.min(width, Math.max(x1, x2)), Math.min(height, Math.max(y1, y2)),
  ];
}

function fitView() {
  const { width, height } = currentImage();
  const scale = Math.min(canvas.width / width, canvas.height / height) * 0.98;
  state.view = { scale, x: (canvas.width - width * scale) / 2, y: (canvas.height - height * scale) / 2 };
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  if (state.image) fitView();
  draw();
}

// ---------- drawing ----------

const currentImage = () => state.session.images[state.index];
const colorOf = (category) => COLORS[Math.max(0, state.session.categories.indexOf(category)) % COLORS.length];

function handles(box) {
  const [x1, y1, x2, y2] = box;
  const xm = (x1 + x2) / 2, ym = (y1 + y2) / 2;
  return { nw: [x1, y1], n: [xm, y1], ne: [x2, y1], e: [x2, ym], se: [x2, y2], s: [xm, y2], sw: [x1, y2], w: [x1, ym] };
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.image) return;
  const { width, height } = currentImage();
  const [ox, oy] = toScreen(0, 0);
  ctx.drawImage(state.image, ox, oy, width * state.view.scale, height * state.view.scale);

  state.objects.forEach((item, index) => {
    const [x1, y1] = toScreen(item.box[0], item.box[1]);
    const [x2, y2] = toScreen(item.box[2], item.box[3]);
    const color = item.rejected ? "#8a8a8a" : colorOf(item.category);
    ctx.setLineDash(item.rejected ? [6, 4] : []);
    ctx.lineWidth = index === state.selected ? 3 : 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);

    const text = item.category + (item.score != null && item.source === "model" ? ` ${item.score.toFixed(2)}` : "")
      + (item.rejected ? " (rejected)" : "");
    ctx.font = "12px system-ui, sans-serif";
    const textWidth = ctx.measureText(text).width + 6;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 16, textWidth, 16);
    ctx.fillStyle = "#000";
    ctx.fillText(text, x1 + 3, y1 - 4);

    if (index === state.selected && !item.rejected) {
      ctx.fillStyle = "#fff";
      for (const [hx, hy] of Object.values(handles(item.box))) {
        const [sx, sy] = toScreen(hx, hy);
        ctx.fillRect(sx - HANDLE / 2, sy - HANDLE / 2, HANDLE, HANDLE);
        ctx.strokeRect(sx - HANDLE / 2, sy - HANDLE / 2, HANDLE, HANDLE);
      }
    }
  });

  if (state.drag && state.drag.kind === "new") {
    const [x1, y1] = toScreen(...state.drag.start);
    const [x2, y2] = toScreen(...state.drag.current);
    ctx.strokeStyle = colorOf(el("category").value);
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
  }
}

// ---------- editing ----------

function remember() {
  state.undo.push(JSON.stringify(state.objects));
  if (state.undo.length > 100) state.undo.shift();
  state.dirty = true;
  updateButtons();
}

function undo() {
  if (!state.undo.length) return;
  state.objects = JSON.parse(state.undo.pop());
  state.selected = -1;
  state.dirty = true;
  updateButtons();
  draw();
}

function updateButtons() {
  const item = state.objects[state.selected];
  el("snap").disabled = !item || item.rejected || state.busy;
  el("reject").disabled = !item;
  el("reject").textContent = item && item.rejected ? "Restore" : "Reject";
  el("undo").disabled = !state.undo.length;
  el("save").textContent = state.dirty ? "Save*" : "Save";
  if (item) el("category").value = item.category;
}

function hitTest(px, py) {
  const item = state.objects[state.selected];
  if (item && !item.rejected) {
    for (const [name, [hx, hy]] of Object.entries(handles(item.box))) {
      const [sx, sy] = toScreen(hx, hy);
      if (Math.abs(px - sx) <= HANDLE && Math.abs(py - sy) <= HANDLE) return { kind: "handle", index: state.selected, name };
    }
  }
  const [x, y] = toImage(px, py);
  let best = -1, bestArea = Infinity;
  state.objects.forEach((other, index) => {
    const [x1, y1, x2, y2] = other.box;
    const area = (x2 - x1) * (y2 - y1);
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && area < bestArea) {
      best = index;
      bestArea = area;
    }
  });
  return best >= 0 ? { kind: "box", index: best } : null;
}

function markChanged(item) {
  if (item.source === "model") item.source = "adjusted";
}

function toggleReject() {
  const item = state.objects[state.selected];
  if (!item) return;
  remember();
  item.rejected = !item.rejected;
  updateButtons();
  draw();
}

async function snapSelected() {
  const item = state.objects[state.selected];
  if (!item || item.rejected || state.busy) return;
  state.busy = true;
  updateButtons();
  setStatus("Snapping… the first snap on an image takes a few seconds");
  try {
    const result = await api(`/api/images/${state.index}/snap`, { method: "POST", body: JSON.stringify({ box: item.box }) });
    remember();
    item.box = result.box;
    item.source = "snapped";
    setStatus("Snapped");
  } catch (error) {
    setStatus(`Snap failed: ${error.message}`);
  } finally {
    state.busy = false;
    updateButtons();
    draw();
  }
}

// ---------- mouse ----------

canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("mousedown", (event) => {
  if (!state.image) return;
  const [px, py] = pointer(event);
  if (event.button === 2 || event.button === 1 || state.space) {
    state.drag = { kind: "pan", from: [px, py], view: { ...state.view } };
    return;
  }
  const hit = hitTest(px, py);
  if (hit && hit.kind === "handle") {
    remember();
    state.drag = { kind: "resize", index: hit.index, name: hit.name };
  } else if (hit && hit.index === state.selected && !state.objects[hit.index].rejected) {
    remember();
    state.drag = { kind: "move", index: hit.index, from: toImage(px, py), box: [...state.objects[hit.index].box] };
  } else if (hit) {
    state.selected = hit.index;
    updateButtons();
  } else {
    state.selected = -1;
    const start = toImage(px, py);
    state.drag = { kind: "new", start, current: start };
    updateButtons();
  }
  draw();
});

window.addEventListener("mousemove", (event) => {
  if (!state.drag) return;
  const [px, py] = pointer(event);
  const [x, y] = toImage(px, py);
  const drag = state.drag;
  if (drag.kind === "pan") {
    state.view.x = drag.view.x + px - drag.from[0];
    state.view.y = drag.view.y + py - drag.from[1];
  } else if (drag.kind === "new") {
    drag.current = [x, y];
  } else if (drag.kind === "move") {
    const item = state.objects[drag.index];
    const dx = x - drag.from[0], dy = y - drag.from[1];
    item.box = [drag.box[0] + dx, drag.box[1] + dy, drag.box[2] + dx, drag.box[3] + dy];
    markChanged(item);
  } else if (drag.kind === "resize") {
    const item = state.objects[drag.index];
    const box = [...item.box];
    if (drag.name.includes("w")) box[0] = x;
    if (drag.name.includes("e")) box[2] = x;
    if (drag.name.includes("n")) box[1] = y;
    if (drag.name.includes("s")) box[3] = y;
    item.box = box;
    markChanged(item);
  }
  draw();
});

window.addEventListener("mouseup", () => {
  const drag = state.drag;
  state.drag = null;
  if (!drag) return;
  if (drag.kind === "new") {
    const box = clampBox([...drag.start, ...drag.current]);
    if (box[2] - box[0] >= MIN_BOX && box[3] - box[1] >= MIN_BOX) {
      remember();
      state.objects.push({ box, category: el("category").value, source: "manual", score: null, rejected: false });
      state.selected = state.objects.length - 1;
    }
  } else if (drag.kind === "move" || drag.kind === "resize") {
    const item = state.objects[drag.index];
    item.box = clampBox(item.box);
  }
  updateButtons();
  draw();
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const [px, py] = pointer(event);
  const factor = Math.exp(-event.deltaY * 0.0015);
  state.view.x = px - (px - state.view.x) * factor;
  state.view.y = py - (py - state.view.y) * factor;
  state.view.scale *= factor;
  draw();
}, { passive: false });

// ---------- keyboard ----------

window.addEventListener("keydown", (event) => {
  if (event.target.tagName === "SELECT") return;
  const key = event.key;
  if (key === " ") { state.space = true; event.preventDefault(); return; }
  if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "z") { event.preventDefault(); undo(); return; }
  if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "s") { event.preventDefault(); save(); return; }
  if (key === "Delete" || key === "Backspace" || key.toLowerCase() === "x") toggleReject();
  else if (key.toLowerCase() === "s") snapSelected();
  else if (key.toLowerCase() === "c") { el("complete").checked = !el("complete").checked; setComplete(); }
  else if (key === "ArrowRight" || key.toLowerCase() === "d") openImage(state.index + 1);
  else if (key === "ArrowLeft" || key.toLowerCase() === "a") openImage(state.index - 1);
  else if (key === "Escape") { state.selected = -1; updateButtons(); draw(); }
});

window.addEventListener("keyup", (event) => {
  if (event.key === " ") state.space = false;
});

// ---------- toolbar ----------

el("category").addEventListener("change", () => {
  const item = state.objects[state.selected];
  if (item && item.category !== el("category").value) {
    remember();
    item.category = el("category").value;
    markChanged(item);
    draw();
  }
  el("category").blur();
});
el("snap").addEventListener("click", snapSelected);
el("reject").addEventListener("click", toggleReject);
el("undo").addEventListener("click", undo);
el("save").addEventListener("click", save);
el("complete").addEventListener("change", setComplete);

function setComplete() {
  state.complete = el("complete").checked;
  state.dirty = true;
  updateButtons();
}

// ---------- images ----------

function renderList() {
  const list = el("images");
  list.innerHTML = "";
  state.session.images.forEach((image, index) => {
    const item = document.createElement("li");
    item.textContent = image.name;
    item.title = image.name;
    item.className = [
      index === state.index ? "current" : "",
      image.complete ? "complete" : image.reviewed ? "reviewed" : "",
    ].join(" ");
    item.addEventListener("click", () => openImage(index));
    list.appendChild(item);
  });
  const reviewed = state.session.images.filter((image) => image.reviewed).length;
  const complete = state.session.images.filter((image) => image.complete).length;
  el("progress").textContent = `${reviewed} of ${state.session.images.length} reviewed, ${complete} complete`;
}

async function save() {
  if (state.index < 0) return;
  const objects = state.objects
    .filter((item) => !item.rejected)
    .map(({ box, category, source, score }) => ({ box, category, source, score }));
  try {
    const status = await api(`/api/images/${state.index}/annotations`, {
      method: "PUT",
      body: JSON.stringify({ objects, complete: state.complete }),
    });
    state.session.images[state.index] = status;
    state.dirty = false;
    setStatus(`Saved ${objects.length} objects`);
    renderList();
    updateButtons();
  } catch (error) {
    setStatus(`Saving failed: ${error.message}`);
    throw error;
  }
}

async function openImage(index) {
  if (index < 0 || index >= state.session.images.length || index === state.index) return;
  if (state.dirty) await save();
  state.index = index;
  state.objects = [];
  state.selected = -1;
  state.undo = [];
  state.image = null;
  renderList();
  draw();
  setStatus("Detecting…");
  const [annotations, image] = await Promise.all([
    api(`/api/images/${index}/annotations`),
    new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = `/api/images/${index}/file`;
    }),
  ]);
  if (state.index !== index) return;
  state.image = image;
  state.objects = annotations.objects.map((item) => ({ ...item, rejected: false }));
  state.complete = annotations.complete;
  state.dirty = false;
  el("complete").checked = state.complete;
  fitView();
  const found = annotations.saved ? "saved review" : "model proposals";
  setStatus(`${state.objects.length} objects (${found})`);
  updateButtons();
  draw();
}

window.addEventListener("beforeunload", (event) => {
  if (state.dirty) {
    event.preventDefault();
    event.returnValue = "";
  }
});

async function init() {
  state.session = await api("/api/session");
  el("model").textContent = `Model: ${state.session.model}`;
  el("category").innerHTML = state.session.categories
    .map((name) => `<option value="${name}">${name}</option>`)
    .join("");
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  if (state.session.images.length) await openImage(0);
}

init().catch((error) => setStatus(`Error: ${error.message}`));
