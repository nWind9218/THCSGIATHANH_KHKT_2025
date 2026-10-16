/* Mimi · Bàn hỗ trợ thầy cô — giao diện thầy cô (vanilla JS, không phụ thuộc bên ngoài).
 * Nguyên tắc: SSE chỉ báo "có thay đổi"; dữ liệu luôn được lấy lại qua REST.
 * Văn bản người dùng chỉ được đưa vào trang bằng textContent. */
(function () {
  "use strict";

  const API = "/console/api";
  const BASE_TITLE = "Mimi · Bàn hỗ trợ thầy cô";
  const MAX_LEN = 4000;
  const FALLBACK_MS = 5000;     // lưới an toàn: sự kiện trong tiến trình không tới được bản sao khác
  const DEBOUNCE_MS = 100;
  const TYPING_TTL_MS = 90000;  // tự tắt "đang trả lời" nếu không nhận được tín hiệu kết thúc

  const STATUS_LABEL = { pending: "Cần hỗ trợ", accepted: "Thầy cô đang trả lời", none: "Mimi đang trả lời" };
  const TABS = [
    { key: "pending", label: "Cần hỗ trợ" },
    { key: "accepted", label: "Thầy cô đang trả lời" },
    { key: "none", label: "Mimi đang trả lời" },
    { key: "all", label: "Tất cả" },
  ];
  const EMPTY_TEXT = {
    pending: "Hiện chưa có học sinh nào cần hỗ trợ.",
    accepted: "Chưa có cuộc trò chuyện nào do thầy cô đang trả lời.",
    none: "Chưa có cuộc trò chuyện nào do Mimi đang trả lời.",
    all: "Chưa có cuộc trò chuyện nào.",
  };

  // ---------- Trạng thái ----------
  const state = {
    user: null,
    tab: "pending",
    convs: [],
    counts: { pending: 0, accepted: 0, none: 0, all: 0 },
    selected: null,       // psid đang mở
    detail: null,         // { conversation, messages }
    drafts: {},           // psid -> nội dung đang soạn
    knownPending: null,   // Set psid đang chờ (null = chưa tải lần đầu)
    skew: 0,              // giây: server_time - giờ máy
    soundOn: true,
    es: null,
    esEverOpen: false,
    busy: new Set(),
    timers: { fallback: null, tick: null, debounce: null, reconnect: null },
    refreshing: false,
    again: null,          // yêu cầu lấy lại đến trong lúc đang lấy: { detail: bool }
    wantDetail: false,    // sự kiện đang chờ debounce có cần lấy lại chi tiết không
    typing: {},           // psid -> timer; có mặt = Mimi đang soạn trả lời
  };

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  };

  // ---------- Lưu tùy chọn (an toàn khi bị chặn) ----------
  function loadPrefs() {
    try { state.soundOn = localStorage.getItem("mimi_console_sound") !== "off"; } catch (e) { /* bỏ qua */ }
  }
  function savePrefs() {
    try { localStorage.setItem("mimi_console_sound", state.soundOn ? "on" : "off"); } catch (e) { /* bỏ qua */ }
  }

  // ---------- Gọi API ----------
  class ApiError extends Error {
    constructor(status, detail) { super(detail); this.status = status; }
  }

  async function api(path, opts) {
    opts = opts || {};
    const method = opts.method || "GET";
    const init = { method, credentials: "same-origin", headers: { Accept: "application/json" } };
    if (method !== "GET") {
      init.headers["Content-Type"] = "application/json";
      init.headers["X-Requested-With"] = "mimi-console";
      init.body = JSON.stringify(opts.body === undefined ? {} : opts.body);
    }
    let res;
    try {
      res = await fetch(API + path, init);
    } catch (e) {
      throw new ApiError(0, "Không kết nối được máy chủ, thầy/cô thử lại nhé");
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* có thể không có nội dung */ }
    if (!res.ok) {
      const detail = data && typeof data.detail === "string" ? data.detail : "Có lỗi xảy ra, thầy/cô thử lại nhé";
      if (res.status === 401 && !opts.noAuthRedirect) showLogin();
      throw new ApiError(res.status, detail);
    }
    return data;
  }

  // Chạy một thao tác và khóa các nút trong lúc chờ.
  async function withBusy(key, buttons, fn) {
    if (state.busy.has(key)) return;
    state.busy.add(key);
    buttons.forEach((b) => { if (b) b.disabled = true; });
    try { return await fn(); }
    catch (e) { if (e.status !== 401) toast(e.message, true); }
    finally {
      state.busy.delete(key);
      buttons.forEach((b) => { if (b) b.disabled = false; });
      renderDetail();
    }
  }

  // ---------- Thông báo nổi ----------
  function toast(message, isError) {
    const t = el("div", "toast" + (isError ? " toast-error" : ""), message);
    $("toasts").appendChild(t);
    setTimeout(() => t.remove(), 6000);
  }

  // ---------- Hiển thị các màn hình ----------
  function showView(name) {
    $("login-view").hidden = name !== "login";
    $("disabled-view").hidden = name !== "disabled";
    $("app-view").hidden = name !== "app";
  }

  function showLogin() {
    stopRealtime();
    state.user = null;
    state.selected = null;
    state.detail = null;
    state.convs = [];
    state.knownPending = null;
    document.title = BASE_TITLE;
    showView("login");
    $("login-pass").value = "";
    $("login-user").focus();
  }

  function showDisabled(detail) {
    stopRealtime();
    $("disabled-text").textContent = detail;
    showView("disabled");
  }

  async function showApp(username) {
    state.user = username;
    $("me-name").textContent = username;
    showView("app");
    renderTabs();
    renderThreadVisibility();
    updateNotifyButton();
    renderSound();
    startRealtime();
    await refreshList();
  }

  // ---------- Tiện ích hiển thị ----------
  const nowSec = () => Date.now() / 1000 + state.skew;

  function displayName(c) {
    if (c.alias) return c.alias;
    return "Học sinh ••" + String(c.psid).slice(-4);
  }

  function relTime(ts) {
    if (!ts) return "";
    const d = Math.max(0, nowSec() - ts);
    if (d < 60) return "vừa xong";
    if (d < 3600) return Math.floor(d / 60) + " phút trước";
    if (d < 86400) return Math.floor(d / 3600) + " giờ trước";
    return Math.floor(d / 86400) + " ngày trước";
  }

  function clock(ts) {
    const d = new Date(ts * 1000);
    const p = (n) => String(n).padStart(2, "0");
    const hm = p(d.getHours()) + ":" + p(d.getMinutes());
    return new Date().toDateString() === d.toDateString() ? hm : hm + " " + p(d.getDate()) + "/" + p(d.getMonth() + 1);
  }

  function badge(status) {
    const s = STATUS_LABEL[status] ? status : "none";
    return el("span", "badge badge-" + s, STATUS_LABEL[s]);
  }

  // ---------- Danh sách ----------
  function renderTabs() {
    const box = $("tabs");
    if (!box.children.length) {
      TABS.forEach((t) => {
        const b = el("button", "tab tab-" + t.key);
        b.type = "button";
        b.setAttribute("role", "tab");
        b.dataset.key = t.key;
        b.appendChild(el("span", "tab-label", t.label));
        b.appendChild(el("span", "count", "0"));
        b.addEventListener("click", () => { state.tab = t.key; renderTabs(); renderList(); });
        box.appendChild(b);
      });
    }
    Array.from(box.children).forEach((b) => {
      b.setAttribute("aria-selected", String(b.dataset.key === state.tab));
      b.querySelector(".count").textContent = String(state.counts[b.dataset.key] || 0);
    });
  }

  function renderList() {
    const list = $("conv-list");
    const focusedPsid = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.psid : null;
    const items = state.convs.filter((c) => state.tab === "all" || c.handoff_status === state.tab);
    list.replaceChildren();
    items.forEach((c) => {
      const li = document.createElement("li");
      const b = el("button", "conv-card" + (c.unread > 0 ? " unread" : ""));
      b.type = "button";
      b.dataset.psid = c.psid;
      if (c.psid === state.selected) b.setAttribute("aria-current", "true");
      const r1 = el("div", "row");
      r1.appendChild(el("span", "conv-name", displayName(c)));
      r1.appendChild(el("span", "conv-time", relTime(c.last_ts)));
      const r2 = el("div", "row");
      if (state.typing[c.psid]) r2.appendChild(el("span", "conv-preview conv-typing", "Mimi đang trả lời…"));
      else r2.appendChild(el("span", "conv-preview", c.last_preview || ""));
      if (c.unread > 0) {
        const dot = el("span", "unread-dot");
        dot.title = "Tin nhắn chưa đọc";
        r2.appendChild(dot);
      }
      const r3 = el("div", "row");
      r3.appendChild(badge(c.handoff_status));
      b.append(r1, r2, r3);
      b.addEventListener("click", () => openConversation(c.psid));
      li.appendChild(b);
      list.appendChild(li);
    });
    const empty = $("list-empty");
    empty.hidden = items.length > 0;
    empty.textContent = EMPTY_TEXT[state.tab] || "";
    if (focusedPsid) {
      const again = list.querySelector('[data-psid="' + CSS.escape(focusedPsid) + '"]');
      if (again) again.focus();
    }
  }

  async function refreshList() {
    try {
      const data = await api("/conversations?filter=all");
      state.convs = data.conversations || [];
      const open = state.convs.find((c) => c.psid === state.selected);
      if (open) open.unread = 0;        // đang mở là đang đọc; danh sách và chi tiết được lấy song song
      state.counts = data.counts || state.counts;
      if (typeof data.server_time === "number") state.skew = data.server_time - Date.now() / 1000;
      detectNewPending();
      renderTabs();
      renderList();
      updateTitle();
    } catch (e) {
      if (e.status === 503) showDisabled(e.message);
      else if (e.status !== 401 && e.status !== 0) toast(e.message, true);
    }
  }

  // ---------- Cảnh báo cuộc trò chuyện mới cần hỗ trợ ----------
  function detectNewPending() {
    const now = new Set(state.convs.filter((c) => c.handoff_status === "pending").map((c) => c.psid));
    if (state.knownPending) {
      const fresh = state.convs.filter((c) => now.has(c.psid) && !state.knownPending.has(c.psid));
      if (fresh.length) {
        if (state.soundOn) beep();
        notifyDesktop(fresh[0]);
      }
    }
    state.knownPending = now;
  }

  function updateTitle() {
    const n = state.counts.pending || 0;
    document.title = (n > 0 ? "(" + n + ") " : "") + BASE_TITLE;
  }

  let audioCtx = null;
  function ensureAudio() {
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      if (!audioCtx) audioCtx = new AC();
      if (audioCtx.state === "suspended") audioCtx.resume();
      return audioCtx;
    } catch (e) { return null; }
  }
  function beep() {
    const ctx = ensureAudio();
    if (!ctx) return;
    try {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.25, ctx.currentTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.45);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.5);
    } catch (e) { /* bỏ qua */ }
  }

  function renderSound() {
    const b = $("sound-btn");
    b.textContent = state.soundOn ? "🔔" : "🔕";
    b.setAttribute("aria-pressed", String(state.soundOn));
    b.title = state.soundOn ? "Âm báo đang bật (bấm để tắt)" : "Âm báo đang tắt (bấm để bật)";
    b.setAttribute("aria-label", b.title);
  }

  function updateNotifyButton() {
    $("notify-btn").hidden = !("Notification" in window) || Notification.permission !== "default";
  }

  function notifyDesktop(conv) {
    try {
      if ("Notification" in window && Notification.permission === "granted") {
        new Notification("Có học sinh cần thầy cô hỗ trợ", { body: displayName(conv) });
      }
    } catch (e) { /* bỏ qua */ }
  }

  // ---------- Cuộc trò chuyện đang mở ----------
  function renderThreadVisibility() {
    const open = !!state.selected;
    $("layout").classList.toggle("show-thread", open);
    $("layout").classList.toggle("show-list", !open);
    $("no-select").hidden = open;
    $("thread-view").hidden = !open;
  }

  function saveDraft() {
    if (state.selected) state.drafts[state.selected] = $("composer-text").value;
  }

  async function openConversation(psid) {
    saveDraft();
    state.selected = psid;
    state.detail = null;
    cancelAlias();
    const opened = state.convs.find((c) => c.psid === psid);
    if (opened && opened.unread > 0) opened.unread = 0;    // xóa chấm chưa đọc ngay
    $("composer-text").value = state.drafts[psid] || "";
    $("messages").replaceChildren();
    renderThreadVisibility();
    renderList();
    renderDetail();
    await loadDetail(psid, true);
  }

  function closeConversation() {
    saveDraft();
    state.selected = null;
    state.detail = null;
    renderThreadVisibility();
    renderList();
  }

  async function loadDetail(psid, scrollBottom) {
    try {
      const data = await api("/conversations/" + encodeURIComponent(psid));
      if (state.selected !== psid) return;           // đã chuyển sang cuộc khác
      state.detail = data;
      renderDetail(scrollBottom);
    } catch (e) {
      if (e.status === 404 && state.selected === psid) {
        toast("Không tìm thấy cuộc trò chuyện này", true);
        closeConversation();
      } else if (e.status !== 401 && e.status !== 0) toast(e.message, true);
    }
  }

  // Lấy lại danh sách (luôn) và chi tiết (nếu cần) song song. Yêu cầu đến lúc đang lấy sẽ được chạy lại ngay sau đó.
  async function refreshAll(detail) {
    detail = detail !== false;
    if (state.refreshing) {
      state.again = { detail: detail || !!(state.again && state.again.detail) };
      return;
    }
    state.refreshing = true;
    try {
      await Promise.all([
        detail && state.selected ? loadDetail(state.selected, false) : null,
        refreshList(),
      ]);
    } finally {
      state.refreshing = false;
      const next = state.again;
      state.again = null;
      if (next && state.user) refreshAll(next.detail);
    }
  }

  function windowInfo(conv) {
    const exp = conv.window_expires_at;
    if (typeof exp !== "number") return { open: false, text: "Chưa có tin nhắn của học sinh để trả lời" };
    const left = exp - nowSec();
    if (left <= 0) return { open: false, text: "Đã hết 24 giờ, không thể nhắn qua Facebook" };
    const h = Math.floor(left / 3600);
    const m = Math.floor((left % 3600) / 60);
    return { open: true, text: "Còn " + h + " giờ " + String(m).padStart(2, "0") + " phút để trả lời" };
  }

  function renderDetail(scrollBottom) {
    if (!state.selected) return;
    const conv = state.detail ? state.detail.conversation : null;
    const fallback = state.convs.find((c) => c.psid === state.selected);
    const info = conv || fallback;
    if (!info) return;
    const status = STATUS_LABEL[info.handoff_status] ? info.handoff_status : "none";
    const win = conv ? windowInfo(conv) : { open: false, text: "" };

    $("head-title").textContent = displayName(info);
    const hb = $("head-badge");
    hb.replaceChildren(badge(status));
    $("head-window").textContent = conv ? win.text : "";

    // Banner + nút chính
    const banner = $("banner");
    banner.className = "banner banner-" + status;
    const btn = $("action-btn");
    const busy = state.busy.has("action");
    if (status === "pending") {
      $("banner-text").textContent = "Mimi đã chuyển cuộc trò chuyện này cho thầy cô";
      btn.textContent = "Tiếp nhận cuộc trò chuyện";
      btn.className = "btn btn-primary btn-big";
      btn.dataset.action = "accept";
    } else if (status === "accepted") {
      $("banner-text").textContent = "Thầy/cô đang trả lời — Mimi tạm im lặng";
      btn.textContent = "Trả lại cho Mimi";
      btn.className = "btn btn-big";
      btn.dataset.action = "resume";
    } else {
      $("banner-text").textContent = "Mimi đang trả lời";
      btn.textContent = "Tiếp nhận";
      btn.className = "btn btn-big";
      btn.dataset.action = "accept";
    }
    btn.disabled = busy || !conv;

    // Tin nhắn
    if (state.detail) renderMessages(state.detail.messages || [], scrollBottom);

    // Khung soạn tin
    const canReply = !!(conv && conv.can_reply && win.open);
    const ta = $("composer-text");
    const sending = state.busy.has("send");
    ta.disabled = !canReply;
    const note = $("composer-note");
    note.classList.toggle("warn", !canReply && !!conv);
    if (!conv) note.textContent = "";
    else if (status !== "accepted") note.textContent = "Hãy bấm “Tiếp nhận” trước khi trả lời học sinh.";
    else if (!win.open) note.textContent = "Đã quá 24 giờ kể từ tin nhắn cuối của học sinh, Facebook không cho nhắn thêm.";
    else note.textContent = "Enter để gửi, Shift+Enter để xuống dòng.";
    updateComposerControls(canReply, sending);
  }

  function updateComposerControls(canReply, sending) {
    const ta = $("composer-text");
    $("counter").textContent = ta.value.length + "/" + MAX_LEN;
    $("send-btn").disabled = !canReply || sending || !ta.value.trim();
  }

  function renderMessages(messages, scrollBottom) {
    const box = $("messages");
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
    const prevTop = box.scrollTop;
    box.replaceChildren();
    messages.forEach((m) => {
      const role = ["student", "mimi", "teacher", "system"].includes(m.role) ? m.role : "system";
      const failed = m.status === "failed";
      const row = el("div", "msg msg-" + role + (failed ? " msg-failed" : ""));
      if (role === "system") {
        row.textContent = m.text;
      } else {
        const who = role === "student" ? "Học sinh" : role === "mimi" ? "Mimi" : (m.author || "Thầy/cô");
        row.appendChild(el("div", "who", who));
        row.appendChild(el("div", "bubble", m.text));
        if (failed) row.appendChild(el("div", "fail", "Gửi lỗi"));
        else if (m.ts) row.appendChild(el("div", "time", clock(m.ts)));
      }
      box.appendChild(row);
    });
    if (state.selected && state.typing[state.selected]) {
      const row = el("div", "msg msg-mimi msg-typing");
      row.appendChild(el("div", "who", "Mimi"));
      row.appendChild(el("div", "bubble", "Mimi đang trả lời…"));
      box.appendChild(row);
    }
    if (scrollBottom || nearBottom) box.scrollTop = box.scrollHeight;
    else box.scrollTop = prevTop;
  }

  // ---------- Hành động ----------
  function onAction() {
    const psid = state.selected;
    const btn = $("action-btn");
    if (!psid) return;
    if (btn.dataset.action === "resume") {
      $("resume-greet").checked = true;
      $("resume-dialog").showModal();
      return;
    }
    withBusy("action", [btn], async () => {
      const data = await api("/conversations/" + encodeURIComponent(psid) + "/accept", { method: "POST" });
      mergeConversation(psid, data.conversation);
      await loadDetail(psid, true);
      await refreshList();
      followStatus(psid);
      $("composer-text").focus();
    });
  }

  function doResume(ev) {
    ev.preventDefault();
    const psid = state.selected;
    const greet = $("resume-greet").checked;
    const dlg = $("resume-dialog");
    if (!psid) { dlg.close(); return; }
    withBusy("action", [$("resume-ok"), $("resume-cancel"), $("action-btn")], async () => {
      const data = await api("/conversations/" + encodeURIComponent(psid) + "/resume", { method: "POST", body: { send_greeting: greet } });
      dlg.close();
      if (data && data.warning) toast(data.warning, true);
      else toast("Đã trả lại cuộc trò chuyện cho Mimi");
      await loadDetail(psid, true);
      await refreshList();
      followStatus(psid);
    });
  }

  // Sau thao tác của chính thầy cô: chuyển sang tab của trạng thái mới để cuộc trò chuyện vẫn hiện trong danh sách.
  function followStatus(psid) {
    if (state.selected !== psid || !state.detail || !state.detail.conversation) return;
    const status = state.detail.conversation.handoff_status;
    if (state.tab === "all" || state.tab === status || !STATUS_LABEL[status]) return;
    state.tab = status;
    renderTabs();
    renderList();
  }

  function mergeConversation(psid, conv) {
    if (state.detail && state.detail.conversation && state.detail.conversation.psid === psid && conv) {
      state.detail.conversation = conv;
    }
  }

  function onSend(ev) {
    if (ev) ev.preventDefault();
    const psid = state.selected;
    const ta = $("composer-text");
    const text = ta.value.trim();
    if (!psid || !text || ta.disabled) return;
    withBusy("send", [$("send-btn")], async () => {
      try {
        await api("/conversations/" + encodeURIComponent(psid) + "/messages", { method: "POST", body: { text } });
        ta.value = "";
        delete state.drafts[psid];
      } finally {
        // Dù thành công hay lỗi (tin lỗi cũng được lưu), lấy lại để thấy kết quả.
        await loadDetail(psid, true);
        refreshList();
      }
    });
  }

  function startAlias() {
    if (!state.detail) return;
    $("alias-input").value = state.detail.conversation.alias || "";
    $("alias-btn").hidden = true;
    $("alias-form").hidden = false;
    $("head-title").hidden = true;
    $("alias-input").focus();
  }
  function cancelAlias() {
    $("alias-btn").hidden = false;
    $("alias-form").hidden = true;
    $("head-title").hidden = false;
  }
  function saveAlias(ev) {
    ev.preventDefault();
    const psid = state.selected;
    const alias = $("alias-input").value.trim();
    withBusy("alias", [$("alias-form").querySelector("button[type=submit]")], async () => {
      const data = await api("/conversations/" + encodeURIComponent(psid) + "/alias", { method: "PUT", body: { alias } });
      mergeConversation(psid, data.conversation);
      cancelAlias();
      await refreshList();
    });
  }

  // ---------- Thời gian thực ----------
  function setConn(on) {
    const c = $("conn");
    c.className = "conn " + (on ? "conn-on" : "conn-off");
    $("conn-text").textContent = on ? "Đã kết nối" : "Đang kết nối lại…";
  }

  // Danh sách luôn được lấy lại; chi tiết chỉ khi sự kiện thuộc cuộc trò chuyện đang mở.
  function scheduleRefresh(psid) {
    if (psid && psid === state.selected) state.wantDetail = true;
    clearTimeout(state.timers.debounce);
    state.timers.debounce = setTimeout(() => {
      const detail = state.wantDetail;
      state.wantDetail = false;
      refreshAll(detail);
    }, DEBOUNCE_MS);
  }

  // "Mimi đang trả lời": chỉ là cờ bật/tắt, không lấy lại dữ liệu.
  function setTyping(psid, active) {
    if (!psid) return;
    clearTimeout(state.typing[psid]);
    if (active) state.typing[psid] = setTimeout(() => setTyping(psid, false), TYPING_TTL_MS);
    else delete state.typing[psid];
    renderList();
    if (psid === state.selected) renderDetail(false);
  }

  function startRealtime() {
    stopRealtime();
    setConn(false);
    state.esEverOpen = false;
    connectEvents();
    state.timers.fallback = setInterval(() => refreshAll(true), FALLBACK_MS);
    // Mỗi 30 giây: cập nhật đồng hồ 24 giờ và thời gian tương đối.
    state.timers.tick = setInterval(() => { renderDetail(); renderList(); }, 30000);
  }

  function connectEvents() {
    if (state.es) state.es.close();
    const es = new EventSource(API + "/events");
    state.es = es;
    es.onopen = () => {
      const reconnect = state.esEverOpen;
      state.esEverOpen = true;
      setConn(true);
      if (reconnect) refreshAll();       // bù những thay đổi đã lỡ
    };
    es.addEventListener("hello", () => setConn(true));
    es.addEventListener("conversation", (ev) => {
      let psid = null;
      try { psid = JSON.parse(ev.data).psid; } catch (e) { /* bỏ qua */ }
      scheduleRefresh(psid);
    });
    es.addEventListener("typing", (ev) => {
      try { const d = JSON.parse(ev.data); setTyping(d.psid, !!d.active); } catch (e) { /* bỏ qua */ }
    });
    es.addEventListener("resync", () => refreshAll(true));
    es.onerror = () => {
      setConn(false);
      if (es.readyState === EventSource.CLOSED && state.es === es) {
        // Trình duyệt không tự nối lại (ví dụ 401): kiểm tra phiên rồi thử lại.
        clearTimeout(state.timers.reconnect);
        state.timers.reconnect = setTimeout(async () => {
          try { await api("/me"); if (state.user) connectEvents(); }
          catch (e) { if (e.status === 503) showDisabled(e.message); }
        }, 3000);
      }
    };
  }

  function stopRealtime() {
    if (state.es) { state.es.close(); state.es = null; }
    Object.keys(state.typing).forEach((k) => clearTimeout(state.typing[k]));
    state.typing = {};
    Object.keys(state.timers).forEach((k) => {
      clearTimeout(state.timers[k]);
      clearInterval(state.timers[k]);
      state.timers[k] = null;
    });
  }

  // ---------- Đăng nhập / đăng xuất ----------
  async function onLogin(ev) {
    ev.preventDefault();
    const err = $("login-error");
    const btn = $("login-btn");
    err.hidden = true;
    btn.disabled = true;
    try {
      const data = await api("/login", {
        method: "POST", noAuthRedirect: true,
        body: { username: $("login-user").value.trim(), password: $("login-pass").value },
      });
      $("login-pass").value = "";
      await showApp(data.username);
    } catch (e) {
      if (e.status === 503) { showDisabled(e.message); }
      else { err.textContent = e.message; err.hidden = false; }
    } finally { btn.disabled = false; }
  }

  async function onLogout() {
    const btn = $("logout-btn");
    btn.disabled = true;
    try { await api("/logout", { method: "POST", noAuthRedirect: true }); } catch (e) { /* vẫn thoát */ }
    btn.disabled = false;
    showLogin();
  }

  // ---------- Khởi tạo ----------
  function bind() {
    $("login-form").addEventListener("submit", onLogin);
    $("logout-btn").addEventListener("click", onLogout);
    $("sound-btn").addEventListener("click", () => {
      state.soundOn = !state.soundOn;
      savePrefs();
      renderSound();
      if (state.soundOn) { ensureAudio(); beep(); }
    });
    $("notify-btn").addEventListener("click", async () => {
      try { await Notification.requestPermission(); } catch (e) { /* bỏ qua */ }
      updateNotifyButton();
    });
    $("back-btn").addEventListener("click", closeConversation);
    $("action-btn").addEventListener("click", onAction);
    $("resume-form").addEventListener("submit", doResume);
    $("resume-cancel").addEventListener("click", () => $("resume-dialog").close());
    $("alias-btn").addEventListener("click", startAlias);
    $("alias-cancel").addEventListener("click", cancelAlias);
    $("alias-form").addEventListener("submit", saveAlias);
    $("alias-input").addEventListener("keydown", (e) => { if (e.key === "Escape") cancelAlias(); });
    $("composer").addEventListener("submit", onSend);
    const ta = $("composer-text");
    ta.addEventListener("input", () => {
      saveDraft();
      const conv = state.detail && state.detail.conversation;
      updateComposerControls(!ta.disabled, state.busy.has("send"));
      if (!conv) return;
    });
    ta.addEventListener("keydown", (e) => {
      // Không gửi khi đang gõ tiếng Việt bằng bộ gõ (IME).
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
        e.preventDefault();
        onSend();
      }
    });
    // Trình duyệt chỉ cho phát âm sau khi người dùng đã bấm vào trang.
    document.addEventListener("pointerdown", ensureAudio, { once: true });
    document.addEventListener("visibilitychange", () => { if (!document.hidden && state.user) refreshAll(); });
  }

  async function init() {
    loadPrefs();
    bind();
    try {
      const me = await api("/me", { noAuthRedirect: true });
      await showApp(me.username);
    } catch (e) {
      if (e.status === 503) showDisabled(e.message);
      else showLogin();
    }
  }

  init();
})();
