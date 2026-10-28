/* TIER CHECK — frontend controller */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const screens = {
    home: $("screenHome"),
    camera: $("screenCamera"),
    scanning: $("screenScanning"),
    result: $("screenResult"),
    error: $("screenError"),
  };

  const video = $("video");
  const canvas = $("captureCanvas");
  const fileInput = $("fileInput");
  const scanImage = $("scanImage");
  const resultPhoto = $("resultPhoto");
  const confettiCanvas = $("confetti");
  const toastEl = $("toast");

  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const BEST_KEY = "tiercheck:best:v1";
  const SOUND_KEY = "tiercheck:sound:v1";
  const MAX_FILE_BYTES = 8 * 1024 * 1024;
  const SCAN_MIN_MS = REDUCED ? 1100 : 2450;

  const TIER_RANK = {
    sub3: 0,
    sub5: 1,
    ltn: 2,
    mtn: 3,
    htn: 4,
    chad: 5,
    true_adam: 6,
  };

  const state = {
    screen: "home",
    stream: null,
    facingMode: "user",
    busy: false,
    sound: readSound(),
    lastDataUrl: null,
    lastResult: null,
    best: readBest(),
    countdownToken: 0,
    audio: null,
    confettiRaf: 0,
    toastTimer: 0,
  };

  /* ------------------------------------------------------------------ */
  /* Utilities                                                          */
  /* ------------------------------------------------------------------ */

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  function readBest() {
    try {
      const raw = localStorage.getItem(BEST_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (typeof parsed?.score !== "number" || typeof parsed?.tier !== "string") {
        return null;
      }
      return parsed;
    } catch {
      return null;
    }
  }

  function readSound() {
    try {
      return localStorage.getItem(SOUND_KEY) !== "off";
    } catch {
      return true;
    }
  }

  function store(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* storage can be unavailable in private mode */
    }
  }

  function showToast(message, duration = 2600) {
    toastEl.textContent = message;
    toastEl.classList.add("is-visible");
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => {
      toastEl.classList.remove("is-visible");
    }, duration);
  }

  function setText(element, text) {
    if (element) element.textContent = text;
  }

  function updateClock() {
    const clock = $("phoneClock");
    if (!clock) return;
    clock.textContent = new Intl.DateTimeFormat("ru-RU", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(new Date());
  }

  function updateBestLine() {
    const line = $("bestLine");
    if (!line) return;
    if (state.best) {
      line.textContent = `РЕКОРД ${Math.round(state.best.score)} · ${state.best.tier} · фото не сохраняется`;
    } else {
      line.textContent = "JPG / PNG / WEBP · до 8 МБ · без регистрации";
    }
  }

  /* ------------------------------------------------------------------ */
  /* Sound                                                              */
  /* ------------------------------------------------------------------ */

  function audioContext() {
    if (!state.sound) return null;
    const AudioCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtor) return null;
    if (!state.audio) {
      try {
        state.audio = new AudioCtor();
      } catch {
        return null;
      }
    }
    if (state.audio.state === "suspended") {
      state.audio.resume().catch(() => {});
    }
    return state.audio;
  }

  function beep(frequency, duration = 0.08, type = "sine", volume = 0.05, delay = 0) {
    const ctx = audioContext();
    if (!ctx) return;
    const start = ctx.currentTime + delay;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(frequency, start);
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(volume, start + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
    osc.connect(gain).connect(ctx.destination);
    osc.start(start);
    osc.stop(start + duration + 0.02);
  }

  function tickSound() {
    beep(620, 0.055, "triangle", 0.045);
  }

  function successSound(tierId) {
    if (!state.sound) return;
    if ((TIER_RANK[tierId] ?? 0) >= 4) {
      [523.25, 659.25, 783.99, 1046.5].forEach((note, index) => {
        beep(note, 0.22, "sine", 0.045, index * 0.09);
      });
    } else {
      beep(392, 0.14, "triangle", 0.035);
      beep(311, 0.2, "triangle", 0.03, 0.11);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Screens                                                           */
  /* ------------------------------------------------------------------ */

  function showScreen(name) {
    if (!screens[name]) return;
    Object.entries(screens).forEach(([key, element]) => {
      const active = key === name;
      element.classList.toggle("is-active", active);
      element.setAttribute("aria-hidden", active ? "false" : "true");
    });
    state.screen = name;
    if (name !== "camera") stopStream();
    if (name === "home") updateBestLine();
  }

  function goHome() {
    state.countdownToken += 1;
    state.busy = false;
    stopStream();
    showScreen("home");
  }

  /* ------------------------------------------------------------------ */
  /* Camera                                                            */
  /* ------------------------------------------------------------------ */

  function stopStream() {
    if (!state.stream) return;
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    if (video) {
      video.pause();
      video.srcObject = null;
      video.classList.remove("is-mirrored");
    }
  }

  function cameraErrorMessage(error) {
    const name = error?.name || "";
    if (name === "NotAllowedError" || name === "PermissionDeniedError" || name === "SecurityError") {
      return "Доступ к камере запрещён. Разреши камеру в настройках браузера или просто загрузи фото.";
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      return "Камера не найдена. Подключи её или загрузи фото с устройства.";
    }
    if (name === "NotReadableError" || name === "TrackStartError") {
      return "Камера занята другим приложением. Закрой его и попробуй снова.";
    }
    if (name === "OverconstrainedError") {
      return "Камера не поддерживает нужный режим. Попробуй другую камеру.";
    }
    return "Не удалось включить камеру. Проверь разрешения браузера или загрузи фото.";
  }

  function showCameraError(error) {
    stopStream();
    state.busy = false;
    setText($("errorTitle"), "КАМЕРА\nНЕ ОТКРЫЛАСЬ");
    setText($("errorText"), cameraErrorMessage(error));
    showScreen("error");
  }

  async function requestStream(facingMode, relaxed = false) {
    if (!navigator.mediaDevices?.getUserMedia) {
      const error = new Error("Insecure context");
      error.name = "SecurityError";
      throw error;
    }
    const videoConstraints = relaxed
      ? { facingMode }
      : {
          facingMode,
          width: { ideal: 1280 },
          height: { ideal: 960 },
        };
    try {
      return await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });
    } catch (error) {
      if (!relaxed && (error.name === "OverconstrainedError" || error.name === "NotFoundError")) {
        return requestStream(undefined, true);
      }
      throw error;
    }
  }

  async function startCamera() {
    if (state.busy) return;
    state.busy = true;
    state.countdownToken += 1;
    const token = state.countdownToken;
    stopStream();

    setText($("cameraTip"), "включаем камеру… разреши доступ");
    setText($("camHint"), "подключение");
    $("shutter").disabled = true;
    showScreen("camera");

    try {
      const stream = await requestStream(state.facingMode);
      if (token !== state.countdownToken || state.screen !== "camera") {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      state.stream = stream;
      video.srcObject = stream;
      video.classList.toggle("is-mirrored", state.facingMode === "user");
      await video.play().catch(() => {});
      setText($("cameraTip"), "нажми на кнопку — начнётся отсчёт");
      setText($("camHint"), "ровно · на уровне глаз");
      $("shutter").disabled = false;
      state.busy = false;
      beep(440, 0.07, "sine", 0.03);
    } catch (error) {
      if (token === state.countdownToken) showCameraError(error);
    }
  }

  function capturePhoto() {
    if (!video.videoWidth || !video.videoHeight) return null;
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    const targetRatio = 0.8;
    let sx = 0;
    let sy = 0;
    let sw = sourceWidth;
    let sh = sourceHeight;

    if (sourceWidth / sourceHeight > targetRatio) {
      sw = Math.round(sourceHeight * targetRatio);
      sx = Math.round((sourceWidth - sw) / 2);
    } else {
      sh = Math.round(sourceWidth / targetRatio);
      sy = Math.round((sourceHeight - sh) / 2);
    }

    canvas.width = 900;
    canvas.height = Math.round(900 / targetRatio);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.save();
    if (state.facingMode === "user") {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    ctx.restore();
    return canvas.toDataURL("image/jpeg", 0.9);
  }

  async function runCountdownAndCapture() {
    if (state.busy || !state.stream) return;
    state.busy = true;
    const token = ++state.countdownToken;
    $("shutter").disabled = true;
    setText($("cameraTip"), "смотри в объектив · не двигайся");

    const countdownEl = $("countdown");
    const numbers = [3, 2, 1];
    for (const number of numbers) {
      if (token !== state.countdownToken || state.screen !== "camera") return;
      countdownEl.textContent = String(number);
      countdownEl.classList.remove("is-ticking");
      void countdownEl.offsetWidth;
      countdownEl.classList.add("is-ticking");
      tickSound();
      await sleep(REDUCED ? 320 : 900);
    }
    if (token !== state.countdownToken || state.screen !== "camera") return;

    const flash = $("flash");
    flash.classList.remove("is-firing");
    void flash.offsetWidth;
    flash.classList.add("is-firing");
    beep(880, 0.09, "square", 0.03);

    const dataUrl = capturePhoto();
    countdownEl.textContent = "";
    countdownEl.classList.remove("is-ticking");

    if (!dataUrl) {
      state.busy = false;
      $("shutter").disabled = false;
      showToast("Кадр не получился. Попробуй ещё раз.");
      return;
    }

    state.lastDataUrl = dataUrl;
    await sleep(REDUCED ? 60 : 180);
    stopStream();
    state.busy = false;
    $("shutter").disabled = false;
    await scanPhoto(dataUrl, "camera");
  }

  /* ------------------------------------------------------------------ */
  /* Upload                                                            */
  /* ------------------------------------------------------------------ */

  function loadImage(source) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("Не удалось прочитать изображение."));
      image.src = source;
    });
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error("Не удалось прочитать файл."));
      reader.readAsDataURL(file);
    });
  }

  async function prepareImage(dataUrl, maxSide = 1400) {
    const image = await loadImage(dataUrl);
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    if (!width || !height) throw new Error("Пустое изображение.");
    const scale = Math.min(1, maxSide / Math.max(width, height));
    const targetWidth = Math.max(1, Math.round(width * scale));
    const targetHeight = Math.max(1, Math.round(height * scale));
    const offscreen = document.createElement("canvas");
    offscreen.width = targetWidth;
    offscreen.height = targetHeight;
    const ctx = offscreen.getContext("2d");
    if (!ctx) throw new Error("Браузер не поддерживает обработку изображений.");
    ctx.fillStyle = "#0b0b0d";
    ctx.fillRect(0, 0, targetWidth, targetHeight);
    ctx.drawImage(image, 0, 0, targetWidth, targetHeight);
    return offscreen.toDataURL("image/jpeg", 0.88);
  }

  async function handleFile(file) {
    if (!file || state.busy) return;
    if (!/^image\/(jpeg|png|webp)$/i.test(file.type)) {
      showToast("Нужен JPG, PNG или WEBP.");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      showToast("Файл больше 8 МБ. Сожми фото и попробуй снова.", 3400);
      return;
    }

    state.busy = true;
    setText($("errorTitle"), "ФОТО\nНЕ ЗАГРУЗИЛОСЬ");
    setText($("errorText"), "Не удалось открыть файл. Попробуй другой JPG или PNG.");
    try {
      const rawDataUrl = await readFileAsDataUrl(file);
      const dataUrl = await prepareImage(rawDataUrl);
      state.lastDataUrl = dataUrl;
      state.busy = false;
      await scanPhoto(dataUrl, "upload");
    } catch (error) {
      state.busy = false;
      setText($("errorText"), error?.message || "Не удалось прочитать изображение.");
      showScreen("error");
    }
  }

  /* ------------------------------------------------------------------ */
  /* Scanning + API                                                    */
  /* ------------------------------------------------------------------ */

  const SCAN_STEPS = [
    { at: 4, status: "ЗАХВАТ КАДРА", log: "FRAME 900×1125 · JPEG" },
    { at: 14, status: "ПОИСК ЛИЦА", log: "YUNET · FACE 01 FOUND" },
    { at: 27, status: "КАРТИРОВАНИЕ", log: "LANDMARKS · EYES / NOSE / MOUTH" },
    { at: 43, status: "ЗАМЕР ЧЕЛЮСТИ", log: "JAWLINE · ARC PROFILE" },
    { at: 58, status: "АНАЛИЗ СИММЕТРИИ", log: "SYMMETRY · PIXEL + MESH" },
    { at: 72, status: "СВЕТ И ТЕКСТУРА", log: "SKIN / HAIR / LIGHT" },
    { at: 86, status: "СВЕРКА С TIER LIST", log: "TIER LADDER · 7 LEVELS" },
  ];

  function resetScanHud() {
    setText($("scanPercent"), "0%");
    setText($("scanStatus"), "ИНИЦИАЛИЗАЦИЯ…");
    $("scanLog").replaceChildren();
    $("scanProgress").style.width = "0%";
  }

  function updateScanHud(percent) {
    const clamped = Math.max(0, Math.min(100, Math.round(percent)));
    setText($("scanPercent"), `${clamped}%`);
    $("scanProgress").style.width = `${clamped}%`;

    const log = $("scanLog");
    for (const step of SCAN_STEPS) {
      if (clamped >= step.at && !log.querySelector(`[data-at="${step.at}"]`)) {
        const line = document.createElement("span");
        line.dataset.at = String(step.at);
        line.textContent = step.log;
        log.append(line);
        if (log.childElementCount > 5) log.firstElementChild?.remove();
      }
    }

    const active = SCAN_STEPS.filter((step) => clamped >= step.at).pop();
    if (active) setText($("scanStatus"), `${active.status}…`);
    if (clamped >= 97) setText($("scanStatus"), "ФИНАЛЬНЫЙ ВЕРДИКТ…");
  }

  async function requestRating(dataUrl) {
    const response = await fetch("/api/rate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });

    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }

    if (!response.ok) {
      let detail = "Сканер не ответил. Попробуй другое фото.";
      if (payload && typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (payload && Array.isArray(payload.detail)) {
        detail = payload.detail.map((item) => item?.msg).filter(Boolean).join(" ") || detail;
      }
      const error = new Error(detail);
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  async function scanPhoto(dataUrl, source) {
    const token = ++state.countdownToken;
    state.busy = true;
    scanImage.src = dataUrl;
    resetScanHud();
    showScreen("scanning");

    const started = performance.now();
    let percent = 0;
    let apiResult = null;
    let apiError = null;
    let finished = false;

    const apiPromise = requestRating(dataUrl)
      .then((payload) => {
        apiResult = payload;
      })
      .catch((error) => {
        apiError = error;
      })
      .finally(() => {
        finished = true;
      });

    const ticker = window.setInterval(() => {
      const elapsed = performance.now() - started;
      const timePart = Math.min(1, elapsed / SCAN_MIN_MS);
      const target = finished ? 100 : Math.min(96, timePart * 94);
      percent += (target - percent) * 0.22;
      if (target - percent < 0.5) percent = target;
      updateScanHud(percent);
    }, 90);

    await Promise.all([apiPromise, sleep(SCAN_MIN_MS)]);
    window.clearInterval(ticker);

    if (token !== state.countdownToken) return;

    percent = 100;
    updateScanHud(100);
    setText($("scanStatus"), "ГОТОВО");
    await sleep(REDUCED ? 120 : 520);

    state.busy = false;

    if (apiError || !apiResult) {
      const isFace = /Лицо не найдено|лицо/i.test(apiError?.message || "");
      setText($("errorTitle"), isFace ? "ЛИЦО\nНЕ НАЙДЕНО" : "СКАН\nСОРВАЛСЯ");
      setText(
        $("errorText"),
        apiError?.message ||
          "Сервер не ответил. Проверь, что приложение запущено, и попробуй ещё раз.",
      );
      showScreen("error");
      return;
    }

    state.lastResult = apiResult;
    state.lastDataUrl = dataUrl;
    renderResult(apiResult, dataUrl, source);
  }

  /* ------------------------------------------------------------------ */
  /* Result                                                            */
  /* ------------------------------------------------------------------ */

  function animateNumber(element, target, duration = 850) {
    if (!element) return;
    const end = Math.max(0, Math.round(target));
    if (REDUCED) {
      element.textContent = String(end);
      return;
    }
    const start = performance.now();
    const from = 0;
    const step = (now) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      element.textContent = String(Math.round(from + (end - from) * eased));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function animateScoreFloat(element, target, duration = 850) {
    if (!element) return;
    const end = Number(target) || 0;
    if (REDUCED) {
      element.textContent = String(Math.round(end));
      return;
    }
    const start = performance.now();
    const step = (now) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      element.textContent = String(Math.round(end * eased));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function renderResult(result, dataUrl, source) {
    const tier = result.tier || {};
    const color = tier.color || "#25f4ee";
    screens.result.style.setProperty("--tier", color);
    screens.result.dataset.tier = tier.id || "";

    resultPhoto.src = dataUrl;
    setText($("resultTier"), tier.name || "—");
    setText($("resultTag"), tier.tag || "");
    setText($("resultRarity"), tier.rarity ? `РЕДКОСТЬ ${tier.rarity}` : "—");

    const now = new Date();
    setText(
      $("resultDate"),
      now.toLocaleDateString("ru-RU", {
        day: "2-digit",
        month: "2-digit",
        year: "2-digit",
      }),
    );

    animateScoreFloat($("resultScore"), result.score);

    const stats = Array.isArray(result.stats) ? result.stats : [];
    const statsWrap = $("resultStats");
    statsWrap.replaceChildren();
    stats.forEach((stat) => {
      const item = document.createElement("div");
      item.className = "stat";

      const head = document.createElement("div");
      head.className = "stat-head";
      const label = document.createElement("span");
      label.className = "stat-label";
      label.textContent = stat.label;
      const value = document.createElement("span");
      value.className = "stat-value";
      value.textContent = String(Math.round(stat.value));
      head.append(label, value);

      const track = document.createElement("div");
      track.className = "stat-track";
      const fill = document.createElement("span");
      fill.className = "stat-fill";
      track.append(fill);

      item.append(head, track);
      statsWrap.append(item);

      requestAnimationFrame(() => {
        fill.style.width = `${Math.max(0, Math.min(100, stat.value))}%`;
      });
    });

    const note = $("resultNote");
    note.replaceChildren();
    const notes = Array.isArray(result.notes) ? result.notes : [];
    notes.forEach((text) => {
      const paragraph = document.createElement("p");
      paragraph.textContent = text;
      note.append(paragraph);
    });
    note.hidden = notes.length === 0;

    const stamp = $("resultStamp");
    stamp.classList.remove("is-stamped");
    void stamp.offsetWidth;
    if (!REDUCED) stamp.classList.add("is-stamped");

    updateBest(result);
    showScreen("result");
    $("resultScroll").scrollTop = 0;
    successSound(tier.id);
    if (!REDUCED && (TIER_RANK[tier.id] ?? 0) >= 3) {
      launchConfetti(color, (TIER_RANK[tier.id] ?? 0) >= 5);
    }
    if (source === "upload") showToast("Фото просканировано. На сервере оно не осталось.");
  }

  function updateBest(result) {
    const score = Number(result.score) || 0;
    if (!state.best || score > state.best.score) {
      state.best = {
        score,
        tier: result.tier?.name || "—",
        at: Date.now(),
      };
      store(BEST_KEY, JSON.stringify(state.best));
    }
    updateBestLine();
  }

  /* ------------------------------------------------------------------ */
  /* Confetti                                                          */
  /* ------------------------------------------------------------------ */

  function launchConfetti(tierColor, big) {
    const ctx = confettiCanvas.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = window.innerWidth;
    const height = window.innerHeight;
    confettiCanvas.width = width * dpr;
    confettiCanvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const colors = [tierColor, "#fe2c55", "#25f4ee", "#b8ff3c", "#ffd76a", "#f7f4ec"];
    const count = big ? 130 : 80;
    const particles = Array.from({ length: count }, () => ({
      x: width * (0.16 + Math.random() * 0.68),
      y: height * (0.12 + Math.random() * 0.22),
      vx: (Math.random() - 0.5) * 7,
      vy: Math.random() * -6 - 2,
      size: 4 + Math.random() * 7,
      color: colors[Math.floor(Math.random() * colors.length)],
      rotation: Math.random() * Math.PI,
      spin: (Math.random() - 0.5) * 0.22,
      life: 0,
      ttl: 120 + Math.random() * 80,
    }));

    const started = performance.now();
    cancelAnimationFrame(state.confettiRaf);

    const frame = (now) => {
      const elapsed = now - started;
      ctx.clearRect(0, 0, width, height);
      let alive = false;
      particles.forEach((particle) => {
        particle.life += 1;
        particle.vy += 0.14;
        particle.vx *= 0.992;
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.rotation += particle.spin;
        const alpha = Math.max(0, 1 - particle.life / particle.ttl);
        if (alpha <= 0 || particle.y > height + 30) return;
        alive = true;
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.translate(particle.x, particle.y);
        ctx.rotate(particle.rotation);
        ctx.fillStyle = particle.color;
        ctx.fillRect(-particle.size / 2, -particle.size / 4, particle.size, particle.size / 2);
        ctx.restore();
      });
      if (alive && elapsed < 5200) {
        state.confettiRaf = requestAnimationFrame(frame);
      } else {
        ctx.clearRect(0, 0, width, height);
      }
    };
    state.confettiRaf = requestAnimationFrame(frame);
  }

  /* ------------------------------------------------------------------ */
  /* Share card                                                        */
  /* ------------------------------------------------------------------ */

  function wrapText(ctx, text, maxWidth) {
    const words = String(text).split(/\s+/);
    const lines = [];
    let line = "";
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (ctx.measureText(candidate).width > maxWidth && line) {
        lines.push(line);
        line = word;
      } else {
        line = candidate;
      }
    }
    if (line) lines.push(line);
    return lines;
  }

  function fitFont(ctx, text, maxWidth, startSize, family, weight = 900) {
    let size = startSize;
    while (size > 22) {
      ctx.font = `${weight} ${size}px ${family}`;
      if (ctx.measureText(text).width <= maxWidth) break;
      size -= 3;
    }
    return size;
  }

  async function buildShareCard(dataUrl, result) {
    const width = 1080;
    const height = 1350;
    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    const ctx = offscreen.getContext("2d");
    if (!ctx) throw new Error("Canvas недоступен.");

    const tier = result.tier || {};
    const color = tier.color || "#25f4ee";
    const score = Math.round(Number(result.score) || 0);

    try {
      await document.fonts.ready;
      await document.fonts.load('900 80px "Unbounded"');
      await document.fonts.load('600 40px "Caveat"');
      await document.fonts.load('500 30px "Onest"');
    } catch {
      /* fall back to system fonts */
    }

    ctx.fillStyle = "#0b0b0d";
    ctx.fillRect(0, 0, width, height);

    try {
      const image = await loadImage(dataUrl);
      const ratio = Math.max(width / image.width, 760 / image.height);
      const drawWidth = image.width * ratio;
      const drawHeight = image.height * ratio;
      ctx.save();
      ctx.filter = "grayscale(0.75) contrast(1.06) brightness(0.62)";
      ctx.drawImage(
        image,
        (width - drawWidth) / 2,
        0,
        drawWidth,
        drawHeight,
      );
      ctx.restore();
    } catch {
      ctx.fillStyle = "#19191e";
      ctx.fillRect(0, 0, width, 760);
    }

    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, "rgba(11,11,13,0.5)");
    gradient.addColorStop(0.34, "rgba(11,11,13,0.12)");
    gradient.addColorStop(0.58, "rgba(11,11,13,0.9)");
    gradient.addColorStop(0.8, "#0b0b0d");
    gradient.addColorStop(1, "#0b0b0d");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = color;
    ctx.fillRect(0, 0, width, 12);

    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "rgba(247,244,236,0.65)";
    ctx.font = '700 24px "Courier New", monospace';
    ctx.fillText("TIER CHECK", 64, 72);
    ctx.textAlign = "right";
    ctx.fillText("MEME SCANNER / 2026", width - 64, 72);
    ctx.textAlign = "left";

    ctx.strokeStyle = "rgba(247,244,236,0.15)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(64, 96);
    ctx.lineTo(width - 64, 96);
    ctx.stroke();

    const labelY = 700;
    ctx.fillStyle = "rgba(247,244,236,0.5)";
    ctx.font = '700 26px "Courier New", monospace';
    ctx.fillText("ТВОЙ ТИР", 64, labelY);

    const tierName = String(tier.name || "—");
    const tierSize = fitFont(ctx, tierName, 900, 128, '"Unbounded", "Arial Black", sans-serif');
    const tierY = labelY + 20 + tierSize * 0.82;
    ctx.fillStyle = color;
    ctx.font = `900 ${tierSize}px "Unbounded", "Arial Black", sans-serif`;
    ctx.fillText(tierName, 60, tierY);

    const tagY = tierY + 50;
    ctx.fillStyle = "rgba(247,244,236,0.7)";
    ctx.font = '600 46px "Caveat", cursive';
    ctx.fillText(String(tier.tag || ""), 64, tagY);

    const scoreY = tagY + 150;
    ctx.fillStyle = "#f7f4ec";
    ctx.font = '900 140px "Unbounded", "Arial Black", sans-serif';
    ctx.fillText(String(score), 60, scoreY);
    const scoreWidth = ctx.measureText(String(score)).width;
    ctx.fillStyle = "rgba(247,244,236,0.4)";
    ctx.font = '700 32px "Courier New", monospace';
    ctx.fillText("/100", 66 + scoreWidth, scoreY - 14);

    ctx.textAlign = "right";
    ctx.fillStyle = "rgba(247,244,236,0.55)";
    ctx.font = '700 22px "Courier New", monospace';
    ctx.fillText("РЕДКОСТЬ", width - 64, scoreY - 104);
    ctx.fillStyle = "#b8ff3c";
    ctx.font = '700 30px "Courier New", monospace';
    ctx.fillText(String(tier.rarity || "—"), width - 64, scoreY - 64);
    ctx.textAlign = "left";

    const topStats = (Array.isArray(result.stats) ? result.stats : [])
      .slice()
      .sort((a, b) => b.value - a.value)
      .slice(0, 3);
    const statsY = scoreY + 58;
    const columnWidth = (width - 128) / 3;
    topStats.forEach((stat, index) => {
      const x = 64 + columnWidth * index;
      ctx.fillStyle = "rgba(247,244,236,0.5)";
      ctx.font = '700 20px "Courier New", monospace';
      ctx.fillText(String(stat.label), x, statsY);
      ctx.fillStyle = "#f7f4ec";
      ctx.font = '900 34px "Unbounded", "Arial Black", sans-serif';
      ctx.fillText(String(Math.round(stat.value)), x, statsY + 46);

      ctx.fillStyle = "rgba(247,244,236,0.14)";
      ctx.fillRect(x, statsY + 64, columnWidth - 28, 5);
      ctx.fillStyle = color;
      ctx.fillRect(x, statsY + 64, (columnWidth - 28) * Math.max(0, Math.min(100, stat.value)) / 100, 5);
    });

    const noteText = Array.isArray(result.notes) && result.notes[0] ? result.notes[0] : "";
    if (noteText) {
      ctx.fillStyle = "rgba(247,244,236,0.72)";
      ctx.font = '500 26px "Onest", "Segoe UI", sans-serif';
      const lines = wrapText(ctx, noteText, 920).slice(0, 2);
      lines.forEach((line, index) => {
        ctx.fillText(line, 64, statsY + 103 + index * 34);
      });
    }

    ctx.strokeStyle = "rgba(247,244,236,0.16)";
    ctx.beginPath();
    ctx.moveTo(64, height - 120);
    ctx.lineTo(width - 64, height - 120);
    ctx.stroke();

    ctx.fillStyle = "rgba(247,244,236,0.4)";
    ctx.font = '500 22px "Onest", "Segoe UI", sans-serif';
    ctx.fillText("Это мем, а не диагноз. Оценка не отражает реальную привлекательность.", 64, height - 76);
    ctx.fillStyle = color;
    ctx.font = '700 20px "Courier New", monospace';
    ctx.fillText("TIERCHECK · FASTAPI + OPENCV", 64, height - 40);

    return offscreen;
  }

  async function downloadCard() {
    if (!state.lastResult || !state.lastDataUrl) return;
    const button = $("shareBtn");
    const original = button.textContent;
    button.disabled = true;
    button.textContent = "РИСУЕМ КАРТОЧКУ…";
    try {
      const card = await buildShareCard(state.lastDataUrl, state.lastResult);
      const blob = await new Promise((resolve) => card.toBlob(resolve, "image/png"));
      if (!blob) throw new Error("Не удалось собрать карточку.");
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `tier-check-${state.lastResult.tier?.id || "scan"}-${Math.round(state.lastResult.score)}.png`;
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 4000);
      showToast("Карточка сохранена. Можно постить.");
      beep(660, 0.08, "sine", 0.035);
      beep(880, 0.12, "sine", 0.03, 0.09);
    } catch (error) {
      showToast(error?.message || "Не удалось сохранить карточку.", 3200);
    } finally {
      button.disabled = false;
      button.textContent = original;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Sound button                                                      */
  /* ------------------------------------------------------------------ */

  function updateSoundButton() {
    const button = $("soundBtn");
    if (!button) return;
    button.setAttribute("aria-pressed", state.sound ? "true" : "false");
    button.title = state.sound ? "Выключить звук" : "Включить звук";
    if (state.sound) {
      beep(520, 0.06, "sine", 0.03);
    }
  }

  function toggleSound() {
    state.sound = !state.sound;
    store(SOUND_KEY, state.sound ? "on" : "off");
    updateSoundButton();
  }

  /* ------------------------------------------------------------------ */
  /* Wire up                                                           */
  /* ------------------------------------------------------------------ */

  $("heroStart")?.addEventListener("click", () => startCamera());
  $("startCamera")?.addEventListener("click", () => startCamera());
  $("uploadTrigger")?.addEventListener("click", () => fileInput.click());
  $("errorUpload")?.addEventListener("click", () => fileInput.click());
  $("errorRetry")?.addEventListener("click", () => startCamera());
  $("closeCamera")?.addEventListener("click", goHome);
  $("shutter")?.addEventListener("click", runCountdownAndCapture);
  $("retakeBtn")?.addEventListener("click", () => startCamera());
  $("shareBtn")?.addEventListener("click", downloadCard);
  $("soundBtn")?.addEventListener("click", toggleSound);

  $("flipCamera")?.addEventListener("click", async () => {
    if (state.busy) return;
    state.facingMode = state.facingMode === "user" ? "environment" : "user";
    beep(520, 0.05, "sine", 0.025);
    await startCamera();
  });

  fileInput?.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    fileInput.value = "";
    if (file) await handleFile(file);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (state.screen === "camera") {
        goHome();
      } else if (state.screen === "result" || state.screen === "error") {
        state.countdownToken += 1;
        goHome();
      }
      return;
    }
    if (event.code === "Space" && state.screen === "camera" && !state.busy) {
      const target = event.target;
      if (target instanceof HTMLElement && /^(BUTTON|A|INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) {
        return;
      }
      event.preventDefault();
      runCountdownAndCapture();
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && state.stream) {
      state.countdownToken += 1;
      state.busy = false;
      stopStream();
      showScreen("home");
    }
  });

  document.addEventListener("dragover", (event) => {
    if (state.screen !== "home") return;
    event.preventDefault();
  });

  document.addEventListener("drop", (event) => {
    if (state.screen !== "home") return;
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    event.preventDefault();
    handleFile(file);
  });

  window.addEventListener("pagehide", stopStream);

  updateClock();
  updateSoundButton();
  updateBestLine();
  setInterval(updateClock, 20000);

  // Preload the font used by the share card so downloads look right.
  if (document.fonts?.load) {
    document.fonts.load('900 40px "Unbounded"').catch(() => {});
  }
})();
