(() => {
  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");

  function resize() {
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
  }
  window.addEventListener("resize", resize);

  const scoreEl = document.getElementById("score");
  const bestEl  = document.getElementById("best");
  const speedEl = document.getElementById("speed");
  const restartBtn = document.getElementById("restartBtn");
  const leftBtn = document.getElementById("leftBtn");
  const rightBtn = document.getElementById("rightBtn");
  const modeBtn = document.getElementById("modeBtn");

  let best = Number(localStorage.getItem("best_lane_car") || 0);
  bestEl.textContent = best;

  const lanes = 3;
  const roadMargin = 0.18;
  let keyLeft = false, keyRight = false;

  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  // ---------------- MODE ----------------
  let mode = "MANUAL"; // MANUAL | AI
  function setMode(m) {
    mode = m;
    if (modeBtn) modeBtn.textContent = `Mode: ${mode}`;
  }
  setMode("MANUAL");
  if (modeBtn) modeBtn.addEventListener("click", () => setMode(mode === "MANUAL" ? "AI" : "MANUAL"));

  // ---------------- POLICY (DQN MLP) ----------------
  // policy.json schema:
  // { "type":"mlp", "state_dim":6, "W1":[[...]], "b1":[...], "W2":[[...]], "b2":[...] }
  let policy = null;

  async function loadPolicy() {
    try {
      const res = await fetch("policy.json", { cache: "no-store" });
      if (!res.ok) throw new Error();
      policy = await res.json();
      console.log("Policy loaded:", policy?.type || "unknown");
    } catch {
      policy = null;
      console.log("No policy.json, using heuristic");
    }
  }
  loadPolicy();

  function argmax(a) {
    let i = 0;
    for (let j = 1; j < a.length; j++) if (a[j] > a[i]) i = j;
    return i;
  }

  // NEW state (6 dims):
  // [lane, ob1_lane, ob1_dist, ob2_lane, ob2_dist, speed_norm]
  function getState() {
    const rr = roadRect();
    const py = rr.height * 0.84;

    // collect obstacles in front (dy >= 0), sort by dy (closest first)
    const ahead = [];
    for (const ob of obstacles) {
      const dy = py - ob.y;
      if (dy >= 0) ahead.push({ dy, lane: ob.lane });
    }
    ahead.sort((a,b) => a.dy - b.dy);

    const ob1 = ahead[0] || { dy: rr.height, lane: targetLane };
    const ob2 = ahead[1] || { dy: rr.height, lane: targetLane };

    const ob1Dist = clamp(ob1.dy / rr.height, 0, 1);
    const ob2Dist = clamp(ob2.dy / rr.height, 0, 1);
    const speedNorm = clamp((speed - 1.0) / 2.2, 0, 1);

    return [targetLane, ob1.lane, ob1Dist, ob2.lane, ob2Dist, speedNorm];
  }

  function relu(x){ return x > 0 ? x : 0; }

  function mlpForward(state) {
    // returns logits length 3
    const W1 = policy.W1, b1 = policy.b1, W2 = policy.W2, b2 = policy.b2;
    const h = new Array(b1.length).fill(0);
    for (let i=0;i<h.length;i++){
      let s = b1[i] || 0;
      for (let j=0;j<state.length;j++) s += (W1[i][j] || 0) * state[j];
      h[i] = relu(s);
    }
    const out = new Array(b2.length).fill(0);
    for (let k=0;k<out.length;k++){
      let s = b2[k] || 0;
      for (let i=0;i<h.length;i++) s += (W2[k][i] || 0) * h[i];
      out[k] = s;
    }
    return out;
  }

  // action: -1,0,+1
  function policyAction(state) {
    if (policy && policy.type === "mlp") {
      const logits = mlpForward(state);
      const idx = argmax(logits); // 0,1,2
      return idx === 0 ? -1 : (idx === 1 ? 0 : +1);
    }

    // heuristic fallback
    const lane = state[0], ob1Lane = state[1], ob1Dist = state[2];
    if (lane === ob1Lane && ob1Dist < 0.35) {
      if (lane === 0) return +1;
      if (lane === 2) return -1;
      return Math.random() < 0.5 ? -1 : +1;
    }
    return 0;
  }

  // ---------------- GAME STATE ----------------
  let t = 0, score = 0, speed = 1.0, alive = true;
  let targetLane = 1, playerX = 0;
  const obstacles = [];
  let spawnTimer = 0;

  function reset() {
    resize();
    t = 0; score = 0;
    speed = 1.0;
    alive = true;
    targetLane = 1;
    playerX = 0;
    obstacles.length = 0;
    spawnTimer = 0;
    scoreEl.textContent = "0";
    speedEl.textContent = "1.0";
  }

  function roadRect() {
    const W = canvas.width, H = canvas.height;
    const left = W * roadMargin, right = W * (1 - roadMargin);
    return { left, right, width: right - left, height: H };
  }

  function laneCenterX(l) {
    const rr = roadRect();
    return rr.left + rr.width * (l + 0.5) / lanes;
  }

  function spawnObstacle() {
    const rr = roadRect();
    const lane = Math.floor(Math.random() * lanes);
    obstacles.push({
      lane,
      x: laneCenterX(lane),
      y: -rr.height * 0.06,
      w: rr.width / lanes * 0.55,
      h: rr.height * 0.06
    });
  }

  function setLane(d) {
    if (!alive) return;
    targetLane = clamp(targetLane + d, 0, lanes - 1);
  }

  // -------- INPUT --------
  window.addEventListener("keydown", e => {
    const k = e.key.toLowerCase();
    if (k === "arrowleft" || k === "a") keyLeft = true;
    if (k === "arrowright" || k === "d") keyRight = true;
    if (k === "r") reset();
    if (k === "m") setMode("MANUAL");
    if (k === "t") setMode("AI");
  });

  window.addEventListener("keyup", e => {
    const k = e.key.toLowerCase();
    if (k === "arrowleft" || k === "a") keyLeft = false;
    if (k === "arrowright" || k === "d") keyRight = false;
  });

  function bindHold(btn, on, off) {
    btn.addEventListener("pointerdown", e => { e.preventDefault(); on(); });
    btn.addEventListener("pointerup", off);
    btn.addEventListener("pointercancel", off);
  }
  bindHold(leftBtn, () => keyLeft = true, () => keyLeft = false);
  bindHold(rightBtn, () => keyRight = true, () => keyRight = false);
  restartBtn.addEventListener("click", reset);

  // -------- LOOP --------
  let last = performance.now();
  function loop(now) {
    const dt = Math.min(0.033, (now - last) / 1000);
    last = now;
    update(dt);
    render();
    requestAnimationFrame(loop);
  }

  function update(dt) {
    if (!alive) return;
    t += dt;
    speed = 1.0 + Math.min(2.2, score / 2500);
    speedEl.textContent = speed.toFixed(1);

    if (mode === "AI") {
      const act = policyAction(getState());
      if (act) setLane(act);
    } else {
      if (keyLeft) setLane(-1), keyLeft = false;
      if (keyRight) setLane(1), keyRight = false;
    }

    const rr = roadRect();
    const targetX = laneCenterX(targetLane);
    const curX = rr.left + rr.width/2 + playerX * (rr.width/2);
    const lerp = 1 - Math.pow(0.001, dt * 8);
    const nx = curX + (targetX - curX) * lerp;
    playerX = (nx - (rr.left + rr.width/2)) / (rr.width/2);

    spawnTimer -= dt * speed;
    if (spawnTimer <= 0) {
      spawnObstacle();
      spawnTimer = Math.max(0.25, 0.65 - score / 6000);
    }

    const fall = rr.height * 0.55 * speed;
    for (const ob of obstacles) ob.y += fall * dt;

    for (let i = obstacles.length - 1; i >= 0; i--) {
      if (obstacles[i].y > rr.height) {
        obstacles.splice(i, 1);
        score += 25;
        scoreEl.textContent = score;
      }
    }

    const p = playerRect();
    for (const ob of obstacles) {
      if (rectOverlap(p, {x:ob.x-ob.w/2,y:ob.y-ob.h/2,w:ob.w,h:ob.h})) {
        alive = false;
        best = Math.max(best, score);
        localStorage.setItem("best_lane_car", best);
        bestEl.textContent = best;
      }
    }
  }

  function playerRect() {
    const rr = roadRect();
    const w = rr.width / lanes * 0.55;
    const h = rr.height * 0.07;
    const x = rr.left + rr.width/2 + playerX * (rr.width/2);
    const y = rr.height * 0.84;
    return { x:x-w/2, y:y-h/2, w, h };
  }

  function rectOverlap(a,b) {
    return a.x < b.x+b.w && a.x+a.w > b.x && a.y < b.y+b.h && a.y+a.h > b.y;
  }

  function render() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0,0,W,H);
    const rr = roadRect();

    ctx.fillStyle="#0b0f14"; ctx.fillRect(0,0,W,H);
    ctx.fillStyle="#111827"; ctx.fillRect(rr.left,0,rr.width,rr.height);

    ctx.strokeStyle="rgba(219,231,255,.18)";
    ctx.lineWidth=Math.max(2,W*0.004);
    for(let i=1;i<lanes;i++){
      const x=rr.left+rr.width*i/lanes;
      ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();
    }

    for(const ob of obstacles){
      ctx.fillStyle="#ef4444";
      ctx.fillRect(ob.x-ob.w/2,ob.y-ob.h/2,ob.w,ob.h);
    }

    const p=playerRect();
    ctx.fillStyle=alive?"#22c55e":"#94a3b8";
    ctx.fillRect(p.x,p.y,p.w,p.h);
  }

  reset();
  requestAnimationFrame(loop);
})();