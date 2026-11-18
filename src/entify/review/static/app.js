"use strict";

const COLORS = [
  "#ffd400", "#00e5ff", "#ff4fd8", "#7cfc00", "#ff9f1a", "#b388ff",
  "#ff6b6b", "#ffffff", "#c6ff00", "#80cbc4", "#ffab91", "#f48fb1",
];
const HANDLE = 6;        // handle size in screen pixels
const MIN_BOX = 3;       // smallest new box in image pixels
const DRAG_START = 4;    // screen pixels the mouse must move before a press counts as drawing, not as a click
const RECENT = 6;        // recently used categories shown first in the category menu

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
  category: null,       // category of new boxes
  recent: [],           // recently picked categories, newest first
  hidden: new Set(),    // categories whose boxes are not shown (still saved)
  menu: { options: [], active: 0 },
};

// ---------- settings kept in the browser ----------

function loadSetting(name, fallback) {
  try {
    const value = localStorage.getItem(`entify-review:${name}`);
    return value === null ? fallback : JSON.parse(value);
  } catch {
    return fallback;
  }
}

function storeSetting(name, value) {
  try {
    localStorage.setItem(`entify-review:${name}`, JSON.stringify(value));
  } catch {
    // Private windows can block storage; the setting then lasts until the page is closed.
  }
}

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
const isVisible = (item) => !state.hidden.has(item.category);
// Model detections keep their score through every edit; boxes drawn by hand never have one.
const isDetection = (item) => item.score != null;

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
    if (!isVisible(item)) return;
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

  if (state.drag && state.drag.kind === "new" && state.drag.moved) {
    const [x1, y1] = toScreen(...state.drag.start);
    const [x2, y2] = toScreen(...state.drag.current);
    ctx.strokeStyle = colorOf(state.category);
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
  el("remove").disabled = !item;
  el("remove").textContent = !item || !isDetection(item) ? "Delete" : item.rejected ? "Restore" : "Reject";
  el("undo").disabled = !state.undo.length;
  el("save").textContent = state.dirty ? "Save*" : "Save";
  if (document.activeElement !== el("category")) {
    el("category-label").textContent = item ? "Selected box" : "New boxes";
    el("category").value = item ? item.category : state.category;
  }
  updateStatusCounts();
}

function updateStatusCounts() {
  const hidden = state.objects.filter((item) => !isVisible(item) && !item.rejected).length;
  el("filter-toggle").classList.toggle("active", state.hidden.size > 0);
  el("filter-toggle").title = `Choose which categories are shown (F)${hidden ? ` · ${hidden} boxes hidden here` : ""}`;
}

function hitTest(px, py) {
  const item = state.objects[state.selected];
  if (item && !item.rejected && isVisible(item)) {
    for (const [name, [hx, hy]] of Object.entries(handles(item.box))) {
      const [sx, sy] = toScreen(hx, hy);
      if (Math.abs(px - sx) <= HANDLE && Math.abs(py - sy) <= HANDLE) return { kind: "handle", index: state.selected, name };
    }
  }
  const [x, y] = toImage(px, py);
  let best = -1, bestArea = Infinity;
  state.objects.forEach((other, index) => {
    if (!isVisible(other)) return;
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

function removeSelected() {
  // A box drawn by hand is deleted outright; a model detection is marked as rejected, so it can be restored.
  const item = state.objects[state.selected];
  if (!item) return;
  remember();
  if (isDetection(item)) {
    item.rejected = !item.rejected;
  } else {
    state.objects.splice(state.selected, 1);
    state.selected = -1;
  }
  updateButtons();
  draw();
}

async function snapSelected() {
  const item = state.objects[state.selected];
  if (!item || item.rejected || state.busy) return;
  state.busy = true;
  updateButtons();
  setStatus("Snapping…");
  const started = performance.now();
  try {
    const result = await api(`/api/images/${state.index}/snap`, { method: "POST", body: JSON.stringify({ box: item.box }) });
    remember();
    item.box = result.box;
    item.source = "snapped";
    setStatus(`Snapped in ${((performance.now() - started) / 1000).toFixed(1)} s`);
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
  closeMenus();
  const [px, py] = pointer(event);
  if (event.button === 2 || event.button === 1 || state.space) {
    state.drag = { kind: "pan", from: [px, py], view: { ...state.view } };
    return;
  }
  const hit = hitTest(px, py);
  const start = toImage(px, py);
  if (!event.shiftKey && hit && hit.kind === "handle") {
    state.drag = { kind: "resize", index: hit.index, name: hit.name, changed: false };
  } else if (!event.shiftKey && hit && hit.index === state.selected && !state.objects[hit.index].rejected) {
    state.drag = { kind: "move", index: hit.index, from: start, box: [...state.objects[hit.index].box], changed: false };
  } else {
    // A press that does not move selects the box under the mouse; a drag draws a new box, also inside other boxes.
    state.drag = { kind: "new", start, current: start, from: [px, py], moved: false, click: hit ? hit.index : -1 };
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
    if (Math.hypot(px - drag.from[0], py - drag.from[1]) > DRAG_START) drag.moved = true;
  } else {
    const item = state.objects[drag.index];
    if (!drag.changed) {
      remember();
      drag.changed = true;
    }
    if (drag.kind === "move") {
      const dx = x - drag.from[0], dy = y - drag.from[1];
      item.box = [drag.box[0] + dx, drag.box[1] + dy, drag.box[2] + dx, drag.box[3] + dy];
    } else {
      const box = [...item.box];
      if (drag.name.includes("w")) box[0] = x;
      if (drag.name.includes("e")) box[2] = x;
      if (drag.name.includes("n")) box[1] = y;
      if (drag.name.includes("s")) box[3] = y;
      item.box = box;
    }
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
    if (drag.moved && box[2] - box[0] >= MIN_BOX && box[3] - box[1] >= MIN_BOX) {
      remember();
      state.objects.push({ box, category: state.category, source: "manual", score: null, rejected: false });
      state.selected = state.objects.length - 1;
      if (state.hidden.has(state.category)) {
        state.hidden.delete(state.category);
        saveHidden();
        renderFilter();
      }
    } else if (!drag.moved) {
      state.selected = drag.click;
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

// ---------- category menu ----------

function categoryOptions(query) {
  const all = state.session.categories;
  const text = query.trim().toLowerCase();
  if (text) {
    const matches = all.filter((name) => name.toLowerCase().includes(text));
    const starts = (name) => name.toLowerCase().startsWith(text) || name.toLowerCase().includes(` ${text}`);
    matches.sort((a, b) => starts(b) - starts(a) || a.localeCompare(b));
    return matches.map((name) => ({ name }));
  }
  const scope = state.session.scope || [];
  const recent = state.recent.filter((name) => all.includes(name) && !scope.includes(name));
  const inImage = [...new Set(state.objects.map((item) => item.category))]
    .filter((name) => !scope.includes(name) && !recent.includes(name)).sort();
  const rest = [...all].sort().filter((name) => !scope.includes(name) && !recent.includes(name) && !inImage.includes(name));
  return [
    ...scope.map((name) => ({ name, group: "Checked in this review" })),
    ...recent.map((name) => ({ name, group: "Recent" })),
    ...inImage.map((name) => ({ name, group: "In this image" })),
    ...rest.map((name) => ({ name, group: "All categories" })),
  ];
}

function renderCategoryMenu() {
  const list = el("category-list");
  const counts = categoryCounts();
  list.innerHTML = "";
  let group = null;
  state.menu.options.forEach((option, index) => {
    if (option.group && option.group !== group) {
      group = option.group;
      const heading = document.createElement("li");
      heading.className = "group";
      heading.textContent = group;
      list.appendChild(heading);
    }
    const item = document.createElement("li");
    item.setAttribute("role", "option");
    item.id = `category-option-${index}`;
    item.className = index === state.menu.active ? "active" : "";
    item.setAttribute("aria-selected", index === state.menu.active);
    item.innerHTML = `<span class="swatch" style="background:${colorOf(option.name)}"></span>`;
    item.append(option.name);
    if (counts[option.name]) {
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = counts[option.name];
      item.appendChild(count);
    }
    item.addEventListener("mousedown", (event) => {
      event.preventDefault();
      chooseCategory(option.name);
    });
    list.appendChild(item);
  });
  if (!state.menu.options.length) {
    const empty = document.createElement("li");
    empty.className = "group";
    empty.textContent = "No matching category";
    list.appendChild(empty);
  }
  const active = el(`category-option-${state.menu.active}`);
  if (active) active.scrollIntoView({ block: "nearest" });
  el("category").setAttribute("aria-activedescendant", active ? active.id : "");
}

function openCategoryMenu() {
  closeFilter();
  const input = el("category");
  const item = state.objects[state.selected];
  input.placeholder = item ? item.category : state.category;
  input.value = "";
  state.menu = { options: categoryOptions(""), active: 0 };
  el("category-list").hidden = false;
  input.setAttribute("aria-expanded", "true");
  renderCategoryMenu();
}

function closeCategoryMenu() {
  el("category-list").hidden = true;
  el("category").setAttribute("aria-expanded", "false");
  el("category").placeholder = "";
  updateButtons();
}

function chooseCategory(name) {
  state.recent = [name, ...state.recent.filter((other) => other !== name)].slice(0, RECENT);
  storeSetting("recent", state.recent);
  if (state.hidden.has(name)) {
    state.hidden.delete(name);
    saveHidden();
  }
  const item = state.objects[state.selected];
  if (item) {
    if (item.category !== name) {
      remember();
      item.category = name;
      markChanged(item);
    }
  } else {
    state.category = name;
  }
  el("category").blur();
  renderFilter();
  draw();
}

el("category").addEventListener("focus", openCategoryMenu);
el("category").addEventListener("blur", closeCategoryMenu);
el("category").addEventListener("input", () => {
  state.menu = { options: categoryOptions(el("category").value), active: 0 };
  renderCategoryMenu();
});
el("category").addEventListener("keydown", (event) => {
  const count = state.menu.options.length;
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    if (count) state.menu.active = (state.menu.active + (event.key === "ArrowDown" ? 1 : count - 1)) % count;
    renderCategoryMenu();
  } else if (event.key === "Enter") {
    event.preventDefault();
    const option = state.menu.options[state.menu.active];
    if (option) chooseCategory(option.name);
  } else if (event.key === "Escape") {
    event.preventDefault();
    el("category").blur();
  }
});

// ---------- shown categories ----------

function categoryCounts() {
  const counts = {};
  for (const item of state.objects) if (!item.rejected) counts[item.category] = (counts[item.category] || 0) + 1;
  return counts;
}

const hiddenKey = () => `hidden:${state.session.output}:${(state.session.scope || ["*"]).join(",")}`;

function saveHidden() {
  storeSetting(hiddenKey(), [...state.hidden]);
  const item = state.objects[state.selected];
  if (item && !isVisible(item)) state.selected = -1;
  updateButtons();
}

function renderFilter() {
  const all = state.session.categories;
  const shown = all.filter((name) => !state.hidden.has(name));
  el("filter-toggle").textContent = `Show: ${shown.length === all.length ? "all"
    : shown.length === 0 ? "none" : shown.length === 1 ? shown[0] : `${shown.length} of ${all.length}`}`;
  if (el("filter-panel").hidden) return;

  const counts = categoryCounts();
  const text = el("filter-search").value.trim().toLowerCase();
  const names = all.filter((name) => name.toLowerCase().includes(text))
    .sort((a, b) => !!counts[b] - !!counts[a] || a.localeCompare(b));
  const list = el("filter-list");
  list.innerHTML = "";
  for (const name of names) {
    const item = document.createElement("li");
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = !state.hidden.has(name);
    box.addEventListener("change", () => {
      if (box.checked) state.hidden.delete(name);
      else state.hidden.add(name);
      saveHidden();
      renderFilter();
      draw();
    });
    label.append(box);
    label.insertAdjacentHTML("beforeend", `<span class="swatch" style="background:${colorOf(name)}"></span>`);
    label.append(name);
    if (counts[name]) {
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = counts[name];
      label.appendChild(count);
    }
    item.appendChild(label);
    list.appendChild(item);
  }
}

function setShown(names) {
  state.hidden = new Set(state.session.categories.filter((name) => !names.includes(name)));
  saveHidden();
  renderFilter();
  draw();
}

function openFilter() {
  el("filter-panel").hidden = false;
  el("filter-toggle").setAttribute("aria-expanded", "true");
  el("filter-search").value = "";
  renderFilter();
  el("filter-search").focus();
}

function closeFilter() {
  el("filter-panel").hidden = true;
  el("filter-toggle").setAttribute("aria-expanded", "false");
}

function closeMenus() {
  closeFilter();
  if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
}

el("filter-toggle").addEventListener("click", () => (el("filter-panel").hidden ? openFilter() : closeFilter()));
el("filter-search").addEventListener("input", renderFilter);
el("filter-search").addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeFilter();
    el("filter-search").blur();
  }
});
el("filter-all").addEventListener("click", () => setShown(state.session.categories));
el("filter-none").addEventListener("click", () => setShown([]));
document.addEventListener("mousedown", (event) => {
  if (!el("filter").contains(event.target)) closeFilter();
});

// ---------- keyboard ----------

window.addEventListener("keydown", (event) => {
  const target = event.target;
  if (target.tagName === "INPUT" && target.type !== "checkbox") return;
  const key = event.key;
  if (key === " ") { state.space = true; event.preventDefault(); return; }
  if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "z") { event.preventDefault(); undo(); return; }
  if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "s") { event.preventDefault(); save(); return; }
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (key === "Delete" || key === "Backspace" || key.toLowerCase() === "x") removeSelected();
  else if (key.toLowerCase() === "s") snapSelected();
  else if (key.toLowerCase() === "c") { el("complete").checked = !el("complete").checked; setComplete(); }
  else if (key.toLowerCase() === "l" || key === "/") { event.preventDefault(); el("category").focus(); }
  else if (key.toLowerCase() === "f") { event.preventDefault(); el("filter-panel").hidden ? openFilter() : closeFilter(); }
  else if (key === "ArrowRight" || key.toLowerCase() === "d") openImage(state.index + 1);
  else if (key === "ArrowLeft" || key.toLowerCase() === "a") openImage(state.index - 1);
  else if (key === "Escape") { closeFilter(); state.selected = -1; updateButtons(); draw(); }
});

window.addEventListener("keyup", (event) => {
  if (event.key === " ") state.space = false;
});

// ---------- toolbar ----------

el("snap").addEventListener("click", snapSelected);
el("remove").addEventListener("click", removeSelected);
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
  const scope = state.session.scope;
  el("progress").textContent = `${reviewed} of ${state.session.images.length} reviewed, ${complete} `
    + (scope ? `checked for ${scope.join(", ")}` : "complete");
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
  const hidden = state.objects.filter((item) => !isVisible(item)).length;
  setStatus(`${state.objects.length} objects (${found})${hidden ? `, ${hidden} hidden` : ""}`);
  updateButtons();
  renderFilter();
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
  const categories = state.session.categories;
  el("model").textContent = `Model: ${state.session.model}`;
  const scope = state.session.scope;
  state.recent = loadSetting("recent", []).filter((name) => categories.includes(name));
  state.category = scope ? state.recent.find((name) => scope.includes(name)) || scope[0] : state.recent[0] || categories[0];
  // With a scope (entify review -s), only its categories are shown at first.
  const hidden = loadSetting(hiddenKey(), scope ? categories.filter((name) => !scope.includes(name)) : []);
  state.hidden = new Set(hidden.filter((name) => categories.includes(name)));
  if (scope) {
    const names = scope.join(", ");
    el("complete-text").textContent = `All ${names} objects checked`;
    el("complete-label").title = `Every ${names} object in this image is marked; other categories are not checked (C)`;
  }
  renderFilter();
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  if (state.session.images.length) await openImage(0);
}

init().catch((error) => setStatus(`Error: ${error.message}`));
