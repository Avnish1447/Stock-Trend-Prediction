/**
 * PULSE 3D // AWARD-WINNING THREE.JS INTERACTIVE SCROLL EXPERIENCE
 * Built with Three.js WebGL, GSAP ScrollTrigger, and Lenis Smooth Scroll
 */

document.addEventListener("DOMContentLoaded", () => {
  // Enforce manual scroll restoration and start unconditionally at top (0, 0)
  if ('scrollRestoration' in history) {
    history.scrollRestoration = 'manual';
  }
  window.scrollTo(0, 0);
  if (window.location.hash) {
    history.replaceState(null, null, window.location.pathname);
  }

  const isReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ==========================================
  // 1. LENIS SMOOTH SCROLLING
  // ==========================================
  let lenis = null;
  if (!isReducedMotion && typeof Lenis !== "undefined") {
    lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: "vertical",
      gestureOrientation: "vertical",
      smoothWheel: true,
      wheelMultiplier: 1.1,
      touchMultiplier: 2,
    });

    lenis.scrollTo(0, { immediate: true });

    lenis.on("scroll", ScrollTrigger.update);
    gsap.ticker.add((time) => {
      lenis.raf(time * 1000);
    });
    gsap.ticker.lagSmoothing(0);
  }

  if (typeof gsap !== "undefined" && typeof ScrollTrigger !== "undefined") {
    gsap.registerPlugin(ScrollTrigger);
  }

  // ==========================================
  // 2. THREE.JS 3D WEBGEL SCENE INITIALIZATION
  // ==========================================
  const canvas = document.getElementById("webgl-canvas");
  if (!canvas || typeof THREE === "undefined") return;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x030305, 0.035);

  const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.set(0, 0, 32);

  const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance"
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;

  // Ambient & Directional Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  const pointLightPrimary = new THREE.PointLight(0x00d2ff, 3, 60);
  pointLightPrimary.position.set(10, 15, 15);
  scene.add(pointLightPrimary);

  const pointLightSecondary = new THREE.PointLight(0x9d4edd, 2.5, 60);
  pointLightSecondary.position.set(-15, -10, 10);
  scene.add(pointLightSecondary);

  // Group to hold all 3D components for mouse tilt
  const worldGroup = new THREE.Group();
  scene.add(worldGroup);

  // 2.1 Neural Particle Sphere (The Core)
  const sphereGeo = new THREE.SphereGeometry(7, 48, 48);
  const sphereCount = sphereGeo.attributes.position.count;
  const sphereColors = new Float32Array(sphereCount * 3);
  const baseColor = new THREE.Color(0x00d2ff);

  for (let i = 0; i < sphereCount; i++) {
    sphereColors[i * 3] = baseColor.r;
    sphereColors[i * 3 + 1] = baseColor.g;
    sphereColors[i * 3 + 2] = baseColor.b;
  }
  sphereGeo.setAttribute("color", new THREE.BufferAttribute(sphereColors, 3));

  const sphereMat = new THREE.PointsMaterial({
    size: 0.16,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending
  });
  const neuralSphere = new THREE.Points(sphereGeo, sphereMat);
  worldGroup.add(neuralSphere);

  // Wireframe Cage around Sphere
  const cageGeo = new THREE.IcosahedronGeometry(7.2, 2);
  const cageMat = new THREE.MeshBasicMaterial({
    color: 0x00d2ff,
    wireframe: true,
    transparent: true,
    opacity: 0.15
  });
  const neuralCage = new THREE.Mesh(cageGeo, cageMat);
  worldGroup.add(neuralCage);

  // 2.2 Deep 3D Candlestick Corridor
  const candleGroup = new THREE.Group();
  worldGroup.add(candleGroup);

  const candleBoxGeo = new THREE.BoxGeometry(0.5, 1, 0.5);
  const wickGeo = new THREE.CylinderGeometry(0.04, 0.04, 1, 6);

  const bullMat = new THREE.MeshStandardMaterial({
    color: 0x00ff66,
    roughness: 0.2,
    metalness: 0.8,
    emissive: 0x00ff66,
    emissiveIntensity: 0.4
  });

  const bearMat = new THREE.MeshStandardMaterial({
    color: 0xff3b30,
    roughness: 0.2,
    metalness: 0.8,
    emissive: 0xff3b30,
    emissiveIntensity: 0.4
  });

  // Spawn 60 3D floating candlesticks along the Z corridor
  for (let i = 0; i < 60; i++) {
    const isBull = Math.random() > 0.45;
    const mat = isBull ? bullMat : bearMat;

    const candleMesh = new THREE.Mesh(candleBoxGeo, mat);
    const height = Math.random() * 3 + 1;
    candleMesh.scale.set(1, height, 1);

    const x = (Math.random() - 0.5) * 22;
    const y = (Math.random() - 0.5) * 14;
    const z = -i * 1.5;

    candleMesh.position.set(x, y, z);
    candleMesh.rotation.y = Math.random() * Math.PI;

    // Wick
    const wick = new THREE.Mesh(wickGeo, mat);
    wick.scale.set(1, height + 1.2, 1);
    wick.position.set(x, y, z);

    candleGroup.add(candleMesh);
    candleGroup.add(wick);
  }

  // 2.3 Concentric 3D Gate Rings (LSTM Forget, Input, Output)
  const ringGroup = new THREE.Group();
  worldGroup.add(ringGroup);

  const ringRadii = [8.5, 10, 11.5, 13];
  const ringColors = [0x00d2ff, 0x9d4edd, 0x00ff66, 0xff9e00];
  const gateRings = [];

  ringRadii.forEach((radius, idx) => {
    const torusGeo = new THREE.TorusGeometry(radius, 0.06, 16, 100);
    const torusMat = new THREE.MeshBasicMaterial({
      color: ringColors[idx],
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending
    });
    const ring = new THREE.Mesh(torusGeo, torusMat);
    ring.rotation.x = Math.PI / 2;
    ringGroup.add(ring);
    gateRings.push(ring);
  });

  // 2.4 Floating Volumetric Starfield Particles
  const particleCount = 1200;
  const particleGeo = new THREE.BufferGeometry();
  const particlePositions = new Float32Array(particleCount * 3);

  for (let i = 0; i < particleCount * 3; i += 3) {
    particlePositions[i] = (Math.random() - 0.5) * 70;
    particlePositions[i + 1] = (Math.random() - 0.5) * 70;
    particlePositions[i + 2] = (Math.random() - 0.5) * 90;
  }
  particleGeo.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));

  const particleMat = new THREE.PointsMaterial({
    size: 0.12,
    color: 0xffffff,
    transparent: true,
    opacity: 0.45,
    blending: THREE.AdditiveBlending
  });
  const starfield = new THREE.Points(particleGeo, particleMat);
  scene.add(starfield);

  // ==========================================
  // 3. MOUSE PARALLAX WITH PHYSICAL INERTIA
  // ==========================================
  let mouseX = 0, mouseY = 0;
  let targetTiltX = 0, targetTiltY = 0;
  let currentTiltX = 0, currentTiltY = 0;

  window.addEventListener("mousemove", (e) => {
    mouseX = (e.clientX / window.innerWidth) - 0.5;
    mouseY = (e.clientY / window.innerHeight) - 0.5;
    targetTiltX = mouseY * 0.45;
    targetTiltY = mouseX * 0.45;
  });

  // ==========================================
  // 4. GSAP SCROLLTRIGGER 3D CAMERA ORCHESTRATION
  // ==========================================
  if (!isReducedMotion && typeof ScrollTrigger !== "undefined") {
    // Stage 1, 2, 3 Card elements
    const card1 = document.getElementById("card-stage-1");
    const card2 = document.getElementById("card-stage-2");
    const card3 = document.getElementById("card-stage-3");

    // Camera Flight Controller
    const cameraTimeline = gsap.timeline({
      scrollTrigger: {
        trigger: "#dive",
        start: "top top",
        end: "bottom bottom",
        scrub: 1,
        onUpdate: (self) => {
          const p = self.progress;
          // Interpolate 3D camera coordinates along custom trajectory
          if (p < 0.33) {
            // Act 1 to Act 2: Dive toward the core
            const t = p / 0.33;
            camera.position.z = THREE.MathUtils.lerp(32, 16, t);
            camera.position.y = THREE.MathUtils.lerp(0, 4, t);
            camera.position.x = THREE.MathUtils.lerp(0, -3, t);
          } else if (p < 0.66) {
            // Act 2 to Act 3: Lateral orbital perspective through gate rings
            const t = (p - 0.33) / 0.33;
            camera.position.z = THREE.MathUtils.lerp(16, 6, t);
            camera.position.y = THREE.MathUtils.lerp(4, -2, t);
            camera.position.x = THREE.MathUtils.lerp(-3, 14, t);
          } else {
            // Act 3 to Act 4: Extreme close up into the singularity
            const t = (p - 0.66) / 0.34;
            camera.position.z = THREE.MathUtils.lerp(6, 1.5, t);
            camera.position.y = THREE.MathUtils.lerp(-2, 0, t);
            camera.position.x = THREE.MathUtils.lerp(14, 0, t);
          }
        }
      }
    });

    // Card 1 Appearance & Exit
    cameraTimeline.to(card1, { opacity: 1, y: 0, duration: 0.25, ease: "power2.out" }, 0.05)
                  .to(card1, { opacity: 0, y: -40, duration: 0.2, ease: "power2.in" }, 0.28);

    // Card 2 Appearance & Exit
    cameraTimeline.to(card2, { opacity: 1, y: 0, duration: 0.25, ease: "power2.out" }, 0.38)
                  .to(card2, { opacity: 0, y: -40, duration: 0.2, ease: "power2.in" }, 0.60);

    // Card 3 Appearance & Exit
    cameraTimeline.to(card3, { opacity: 1, y: 0, duration: 0.25, ease: "power2.out" }, 0.70)
                  .to(card3, { opacity: 0, y: -40, duration: 0.2, ease: "power2.in" }, 0.94);
  }

  // ==========================================
  // 5. HORIZONTAL 3D ENGINE SPECS (GSAP SCRUB)
  // ==========================================
  const horizontalStrip = document.getElementById("horizontal-strip");
  if (horizontalStrip && !isReducedMotion && typeof ScrollTrigger !== "undefined") {
    const totalDist = horizontalStrip.scrollWidth - window.innerWidth + 200;

    gsap.to(horizontalStrip, {
      x: -totalDist,
      ease: "none",
      scrollTrigger: {
        trigger: "#engines",
        start: "top top",
        end: "bottom bottom",
        scrub: 1
      }
    });
  }

  // ==========================================
  // 6. KEYNOTE MANIFESTO WORD SCRUB REVEAL
  // ==========================================
  const manifesto = document.getElementById("manifesto-text");
  if (manifesto) {
    const words = manifesto.textContent.trim().split(/\s+/);
    manifesto.innerHTML = "";
    words.forEach((w) => {
      const span = document.createElement("span");
      span.className = "scrub-token";
      span.textContent = w;
      manifesto.appendChild(span);
    });

    const tokens = manifesto.querySelectorAll(".scrub-token");
    if (!isReducedMotion && typeof ScrollTrigger !== "undefined") {
      gsap.to(tokens, {
        opacity: 1,
        color: "#ffffff",
        stagger: 0.06,
        ease: "power1.out",
        scrollTrigger: {
          trigger: ".chapter-manifesto",
          start: "top 75%",
          end: "bottom 35%",
          scrub: 0.8
        }
      });
    } else {
      tokens.forEach(t => t.style.opacity = "1");
    }
  }

  // ==========================================
  // 7. REAL API PREDICTION ENGINE & DUAL CHART
  // ==========================================
  const searchForm = document.getElementById("stock-search-form");
  const tickerInput = document.getElementById("ticker-input");
  const searchBtn = document.getElementById("search-btn");
  const quickChips = document.querySelectorAll(".quick-chip");

  const cockpitTicker = document.getElementById("cockpit-ticker");
  const cockpitLatestPrice = document.getElementById("cockpit-latest-price");
  const cockpitTodayChange = document.getElementById("cockpit-today-change");
  const cockpit52wHigh = document.getElementById("cockpit-52w-high");
  const cockpit52wLow = document.getElementById("cockpit-52w-low");
  const valNextDay = document.getElementById("val-next-day");
  const valNextDelta = document.getElementById("val-next-delta");
  const valRmse = document.getElementById("val-rmse");
  const valMape = document.getElementById("val-mape");

  const chartViewport = document.getElementById("chart-viewport");
  const actualCurve = document.getElementById("int-actual-curve");
  const predCurve = document.getElementById("int-pred-curve");
  const ma100Curve = document.getElementById("int-ma100-curve");
  const ma200Curve = document.getElementById("int-ma200-curve");
  const crosshairDot = document.getElementById("int-crosshair-dot");
  const crosshairV = document.getElementById("crosshair-v");
  const chartTooltip = document.getElementById("chart-tooltip");
  const tipDate = document.getElementById("tip-date");
  const tipActual = document.getElementById("tip-actual");
  const tipPred = document.getElementById("tip-pred");
  const tipError = document.getElementById("tip-error");

  const toggleMa100 = document.getElementById("toggle-ma100");
  const toggleMa200 = document.getElementById("toggle-ma200");

  let showMa100 = true;
  let showMa200 = true;
  let currentPredictionData = null;

  if (toggleMa100) {
    toggleMa100.addEventListener("click", () => {
      showMa100 = !showMa100;
      toggleMa100.classList.toggle("active", showMa100);
      if (ma100Curve) ma100Curve.style.display = showMa100 ? "block" : "none";
    });
  }

  if (toggleMa200) {
    toggleMa200.addEventListener("click", () => {
      showMa200 = !showMa200;
      toggleMa200.classList.toggle("active", showMa200);
      if (ma200Curve) ma200Curve.style.display = showMa200 ? "block" : "none";
    });
  }

  async function runPrediction(symbol, scrollToResults = false) {
    if (!symbol) symbol = "AAPL";
    symbol = symbol.trim().toUpperCase();

    // Button loading state
    if (searchBtn) {
      searchBtn.disabled = true;
      searchBtn.querySelector(".btn-text").style.display = "none";
      searchBtn.querySelector(".btn-loader").style.display = "inline";
    }

    // Active chip highlight
    quickChips.forEach(chip => {
      chip.classList.toggle("active", chip.dataset.symbol === symbol);
    });

    try {
      const response = await fetch(`/api/predict?ticker=${encodeURIComponent(symbol)}`);
      const data = await response.json();

      if (data.error) {
        alert("Stock Prediction Error: " + data.error);
        return;
      }

      currentPredictionData = data;

      // Update Hero Live Preview Badge
      const pillTicker = document.getElementById("hero-pill-ticker");
      const pillClose = document.getElementById("hero-pill-close");
      const pillPred = document.getElementById("hero-pill-pred");
      const pillDelta = document.getElementById("hero-pill-delta");
      const heroPill = document.getElementById("hero-live-pill");

      if (pillTicker) pillTicker.textContent = data.ticker;
      if (pillClose) pillClose.textContent = `$${data.latest_close.toFixed(2)}`;
      if (pillPred) pillPred.textContent = `$${data.next_day_pred.toFixed(2)}`;
      if (pillDelta) {
        const dSign = data.next_day_pct >= 0 ? "+" : "";
        pillDelta.textContent = `(${dSign}${data.next_day_pct.toFixed(2)}%)`;
        pillDelta.style.color = data.next_day_pct >= 0 ? "var(--neon-emerald)" : "#ff3b30";
      }
      if (heroPill) {
        heroPill.style.borderColor = data.next_day_pct >= 0 ? "rgba(0, 255, 102, 0.4)" : "rgba(255, 59, 48, 0.4)";
      }

      // Sync all input fields across sections
      if (tickerInput) tickerInput.value = data.ticker;
      const cockpitInput = document.getElementById("cockpit-ticker-input");
      if (cockpitInput) cockpitInput.value = data.ticker;

      // Update Cockpit Header
      if (cockpitTicker) cockpitTicker.textContent = data.ticker;
      if (cockpitLatestPrice) cockpitLatestPrice.textContent = `$${data.latest_close.toFixed(2)}`;
      if (cockpitTodayChange) {
        const sign = data.pct_change >= 0 ? "+" : "";
        cockpitTodayChange.textContent = `${sign}${data.price_change.toFixed(2)} (${sign}${data.pct_change.toFixed(2)}%)`;
        cockpitTodayChange.style.color = data.pct_change >= 0 ? "var(--neon-emerald)" : "#ff3b30";
      }
      if (cockpit52wHigh) cockpit52wHigh.textContent = `$${data.high_52w.toFixed(2)}`;
      if (cockpit52wLow) cockpit52wLow.textContent = `$${data.low_52w.toFixed(2)}`;

      // Update KPI Cards
      if (valNextDay) valNextDay.textContent = `$${data.next_day_pred.toFixed(2)}`;
      if (valNextDelta) {
        const dSign = data.next_day_pct >= 0 ? "+" : "";
        valNextDelta.textContent = `${dSign}${data.next_day_change.toFixed(2)} (${dSign}${data.next_day_pct.toFixed(2)}%)`;
        valNextDelta.style.color = data.next_day_pct >= 0 ? "var(--neon-emerald)" : "#ff3b30";
      }
      if (valRmse) valRmse.textContent = `$${data.rmse.toFixed(2)}`;
      if (valMape) valMape.textContent = `${data.mape.toFixed(2)}%`;

      // Render the SVG Charts with real test data
      renderRealDualChart(data);

      // Morph 3D WebGL scene to match the stock
      morph3DSceneForStock(data);

      // Gradually scroll down to where the predictions are displayed
      if (scrollToResults) {
        const targetEl = document.getElementById("main-cockpit-card") || document.getElementById("dashboard");
        if (targetEl) {
          if (lenis) {
            lenis.scrollTo(targetEl, {
              offset: -85,
              duration: 1.4,
              easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t))
            });
          } else {
            const targetY = targetEl.getBoundingClientRect().top + window.pageYOffset - 85;
            window.scrollTo({ top: targetY, behavior: "smooth" });
          }

          // Subtle glow pulse on the cockpit card to celebrate fresh prediction
          gsap.fromTo(targetEl, 
            { boxShadow: "0 0 50px rgba(0, 210, 255, 0.45), 0 30px 80px rgba(0, 0, 0, 0.7)" },
            { boxShadow: "0 0 0px rgba(0, 210, 255, 0), 0 30px 80px rgba(0, 0, 0, 0.7)", duration: 1.8, ease: "power2.out" }
          );
        }
      }

    } catch (err) {
      console.error("Failed to fetch prediction:", err);
    } finally {
      if (searchBtn) {
        searchBtn.disabled = false;
        searchBtn.querySelector(".btn-text").style.display = "inline";
        searchBtn.querySelector(".btn-loader").style.display = "none";
      }
    }
  }

  function renderRealDualChart(data) {
    const comparison = data.test_comparison;
    const history = data.recent_history;
    if (!comparison || comparison.length === 0) return;

    const width = 1000;
    const height = 320;
    const padding = 20;

    // Determine min and max price across comparison & history
    const allPrices = [
      ...comparison.map(c => c.actual),
      ...comparison.map(c => c.predicted)
    ];
    if (history) {
      history.forEach(h => {
        if (h.ma100) allPrices.push(h.ma100);
        if (h.ma200) allPrices.push(h.ma200);
      });
    }

    const minPrice = Math.min(...allPrices) * 0.96;
    const maxPrice = Math.max(...allPrices) * 1.04;

    const scaleX = (idx, total) => (idx / (total - 1)) * (width - padding * 2) + padding;
    const scaleY = (price) => height - padding - ((price - minPrice) / (maxPrice - minPrice)) * (height - padding * 2);

    // 1. Build Actual Path
    let actualD = `M ${scaleX(0, comparison.length)} ${scaleY(comparison[0].actual)}`;
    for (let i = 1; i < comparison.length; i++) {
      actualD += ` L ${scaleX(i, comparison.length)} ${scaleY(comparison[i].actual)}`;
    }
    if (actualCurve) actualCurve.setAttribute("d", actualD);

    // 2. Build Predicted Path
    let predD = `M ${scaleX(0, comparison.length)} ${scaleY(comparison[0].predicted)}`;
    for (let j = 1; j < comparison.length; j++) {
      predD += ` L ${scaleX(j, comparison.length)} ${scaleY(comparison[j].predicted)}`;
    }
    if (predCurve) predCurve.setAttribute("d", predD);

    // 3. Build MA100 & MA200 if history available
    if (history && history.length > 0) {
      let ma100D = "";
      let ma200D = "";
      let first100 = true;
      let first200 = true;

      for (let k = 0; k < history.length; k++) {
        const x = scaleX(k, history.length);
        if (history[k].ma100) {
          const y100 = scaleY(history[k].ma100);
          ma100D += first100 ? `M ${x} ${y100}` : ` L ${x} ${y100}`;
          first100 = false;
        }
        if (history[k].ma200) {
          const y200 = scaleY(history[k].ma200);
          ma200D += first200 ? `M ${x} ${y200}` : ` L ${x} ${y200}`;
          first200 = false;
        }
      }
      if (ma100Curve) ma100Curve.setAttribute("d", ma100D);
      if (ma200Curve) ma200Curve.setAttribute("d", ma200D);
    }
  }

  // Chart hover crosshair interaction
  if (chartViewport) {
    chartViewport.addEventListener("mousemove", (e) => {
      if (!currentPredictionData || !currentPredictionData.test_comparison) return;

      const rect = chartViewport.getBoundingClientRect();
      const relX = e.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, relX / rect.width));

      const comparison = currentPredictionData.test_comparison;
      const targetIdx = Math.floor(pct * (comparison.length - 1));
      const pt = comparison[targetIdx];
      if (!pt) return;

      const width = 1000;
      const height = 320;
      const padding = 20;

      const allPrices = [
        ...comparison.map(c => c.actual),
        ...comparison.map(c => c.predicted)
      ];
      const minPrice = Math.min(...allPrices) * 0.96;
      const maxPrice = Math.max(...allPrices) * 1.04;
      const scaleX = (idx, total) => (idx / (total - 1)) * (width - padding * 2) + padding;
      const scaleY = (price) => height - padding - ((price - minPrice) / (maxPrice - minPrice)) * (height - padding * 2);

      const mappedX = scaleX(targetIdx, comparison.length);
      const mappedY = scaleY(pt.actual);

      if (crosshairV) {
        crosshairV.style.display = "block";
        crosshairV.setAttribute("x1", mappedX);
        crosshairV.setAttribute("x2", mappedX);
      }

      if (crosshairDot) {
        crosshairDot.style.display = "block";
        crosshairDot.setAttribute("cx", mappedX);
        crosshairDot.setAttribute("cy", mappedY);
      }

      if (chartTooltip) {
        chartTooltip.style.display = "block";
        tipDate.textContent = pt.date;
        tipActual.textContent = `ACTUAL: $${pt.actual.toFixed(2)}`;
        tipPred.textContent = `PRED: $${pt.predicted.toFixed(2)}`;
        const err = Math.abs(pt.actual - pt.predicted).toFixed(2);
        tipError.textContent = `DEV: $${err}`;
      }
    });

    chartViewport.addEventListener("mouseleave", () => {
      if (crosshairV) crosshairV.style.display = "none";
      if (crosshairDot) crosshairDot.style.display = "none";
      if (chartTooltip) chartTooltip.style.display = "none";
    });
  }

  function morph3DSceneForStock(data) {
    const isBull = data.pct_change >= 0;
    const targetColor = isBull ? new THREE.Color(0x00ff66) : new THREE.Color(0xff3b30);

    // Pulse point lights
    gsap.to(pointLightPrimary.color, { r: targetColor.r, g: targetColor.g, b: targetColor.b, duration: 1.2 });
    gsap.to(neuralCage.material.color, { r: targetColor.r, g: targetColor.g, b: targetColor.b, duration: 1.2 });

    // Pulse sphere scale
    gsap.to(neuralSphere.scale, { x: 1.3, y: 1.3, z: 1.3, duration: 0.4, yoyo: true, repeat: 1 });
  }

  // Bind Hero Search Form Submit
  if (searchForm) {
    searchForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const sym = tickerInput ? tickerInput.value.trim() : "AAPL";
      runPrediction(sym, true);
    });
  }

  // Bind Quick Chips Click
  quickChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const sym = chip.dataset.symbol;
      if (tickerInput) tickerInput.value = sym;
      runPrediction(sym, true);
    });
  });

  // Bind Cockpit Search Form Submit
  const cockpitForm = document.getElementById("cockpit-search-form");
  const cockpitInput = document.getElementById("cockpit-ticker-input");
  if (cockpitForm) {
    cockpitForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const sym = cockpitInput ? cockpitInput.value.trim() : "AAPL";
      if (tickerInput) tickerInput.value = sym;
      runPrediction(sym, true);
    });
  }

  // Run initial prediction for AAPL
  runPrediction("AAPL", false);



  // ==========================================
  // 9. AMBIENT 432Hz SYNTHESIZED KEYNOTE HUM (WEB AUDIO API)
  // ==========================================
  const audioToggle = document.getElementById("audio-toggle");
  const audioLabel = document.getElementById("audio-label");
  let audioCtx = null;
  let oscillator = null;
  let gainNode = null;
  let isAudioPlaying = false;

  if (audioToggle) {
    audioToggle.addEventListener("click", () => {
      if (!isAudioPlaying) {
        try {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          oscillator = audioCtx.createOscillator();
          gainNode = audioCtx.createGain();

          const filter = audioCtx.createBiquadFilter();
          filter.type = "lowpass";
          filter.frequency.setValueAtTime(320, audioCtx.currentTime);

          oscillator.type = "sine";
          oscillator.frequency.setValueAtTime(432, audioCtx.currentTime); // 432 Hz soothing harmonic

          gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
          gainNode.gain.linearRampToValueAtTime(0.04, audioCtx.currentTime + 2); // Soft volume

          oscillator.connect(filter);
          filter.connect(gainNode);
          gainNode.connect(audioCtx.destination);

          oscillator.start();
          isAudioPlaying = true;
          audioLabel.textContent = "AUDIO: 432Hz ON";
          audioToggle.style.borderColor = "var(--neon-cyan)";
        } catch (e) {
          console.warn("Web Audio API not allowed without user gesture:", e);
        }
      } else {
        if (gainNode && audioCtx) {
          gainNode.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.5);
          setTimeout(() => {
            if (oscillator) oscillator.stop();
            isAudioPlaying = false;
            audioLabel.textContent = "AUDIO: OFF";
            audioToggle.style.borderColor = "var(--border-glass)";
          }, 500);
        }
      }
    });
  }

  // ==========================================
  // 10. CLI COPY BUTTON
  // ==========================================
  const cliCopy = document.getElementById("cli-copy");
  if (cliCopy) {
    cliCopy.addEventListener("click", () => {
      navigator.clipboard.writeText("streamlit run app.py");
      cliCopy.textContent = "COPIED TO CLIPBOARD!";
      setTimeout(() => {
        cliCopy.textContent = "COPY COMMAND";
      }, 2000);
    });
  }

  // ==========================================
  // 11. MAIN RENDER LOOP & PERFORMANCE TELEMETRY
  // ==========================================
  let frameCount = 0;
  let lastTime = performance.now();
  const hudFps = document.getElementById("hud-fps");

  const clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();
    const elapsedTime = clock.getElapsedTime();

    // Mouse tilt lerp
    currentTiltX += (targetTiltX - currentTiltX) * 0.05;
    currentTiltY += (targetTiltY - currentTiltY) * 0.05;
    worldGroup.rotation.x = currentTiltX;
    worldGroup.rotation.y = currentTiltY;

    // Continuous 3D rotations
    neuralSphere.rotation.y = elapsedTime * 0.12;
    neuralCage.rotation.y = -elapsedTime * 0.08;
    neuralCage.rotation.x = elapsedTime * 0.05;

    // Rotate LSTM gate rings at varying orbital speeds
    gateRings.forEach((ring, idx) => {
      ring.rotation.z = elapsedTime * (0.15 + idx * 0.08);
      ring.rotation.x = Math.PI / 2 + Math.sin(elapsedTime * 0.5 + idx) * 0.1;
    });

    // Starfield drift
    starfield.rotation.y = -elapsedTime * 0.02;

    // Render WebGL
    renderer.render(scene, camera);

    // Telemetry FPS Counter
    frameCount++;
    const now = performance.now();
    if (now - lastTime >= 1000) {
      if (hudFps) hudFps.textContent = frameCount;
      frameCount = 0;
      lastTime = now;
    }
  }

  animate();

  // Resize handler
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  });

  // Final safeguard: guarantee viewport is at top (0, 0) once assets finish loading
  window.addEventListener("load", () => {
    window.scrollTo(0, 0);
    if (lenis) lenis.scrollTo(0, { immediate: true });
  });
});
