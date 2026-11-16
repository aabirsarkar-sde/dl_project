// Builds reports/phase2_presentation.pptx from reports/phase2_results.json and the figures.
// Run from the repo root:  node reports/scripts/build_phase2_deck.js
// Style follows the team's Phase 1 "RUL LAB" deck (dark navy, indigo panels, cyan / pink accents).

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "reports", "figures");
const DECK = path.join(FIG, "deck");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "reports", "phase2_results.json"), "utf8"));
const TEXT = JSON.parse(fs.readFileSync(path.join(ROOT, "reports", "scripts", "deck_text.json"), "utf8"));

// ---- palette (from the Phase 1 team deck) ----
const BG = "050B10", PANEL = "0D1424", LINE = "253440";
const FG = "F4F6FB", SOFT = "CDD5DF", MUTED = "9CA9B7";
const VIOLET = "9DA5FF", PINK = "F04ADB", CYAN = "84D8FF", INDIGO = "5362D1";
const FONT = "Calibri";

const W = 13.333, H = 7.5, M = 0.6;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "RUL Prediction - Phase 2";
pres.author = "Team T017";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

// ---- helpers ----
const f2 = (x) => Number(x).toFixed(2);
const cv = (k) => R.cv[k];
const te = (k) => R.test[k];

function txt(slide, text, o) {
  slide.addText(text, Object.assign({ isTextBox: true, fontFace: FONT, color: FG, margin: 0, valign: "top" }, o));
}

let pageNo = 0;
function chrome(slide, label, title, opts = {}) {
  pageNo += 1;
  slide.background = { color: BG };
  // logo
  slide.addShape(pres.shapes.DIAMOND, { x: M, y: 0.42, w: 0.16, h: 0.16, fill: { color: FG }, line: { color: FG } });
  txt(slide, "RUL LAB", { x: M + 0.26, y: 0.38, w: 2, h: 0.25, fontSize: 10, bold: true, charSpacing: 1 });
  if (label) txt(slide, label.toUpperCase(), { x: M, y: 0.95, w: opts.titleW || 7.5, h: 0.25, fontSize: 10, color: CYAN, charSpacing: 1 });
  if (title) txt(slide, title, { x: M, y: 1.22, w: opts.titleW || 7.5, h: 1.0, fontSize: 30, color: FG });
  txt(slide, "PHASE 2  /  C-MAPSS FD001", { x: M, y: H - 0.45, w: 4, h: 0.22, fontSize: 8, color: MUTED, charSpacing: 1 });
  txt(slide, String(pageNo).padStart(2, "0"), { x: W - M - 1, y: H - 0.45, w: 1, h: 0.22, fontSize: 8, color: MUTED, align: "right" });
}

function panel(slide, x) {
  // right-hand indigo gradient panel, as in the team deck
  slide.addImage({ path: path.join(DECK, "gradient_square.png"), x, y: 0, w: W - x, h: H, sizing: { type: "cover", w: W - x, h: H } });
}

function card(slide, x, y, w, h, fill = PANEL) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: LINE, width: 0.75 } });
}

function lightFigure(slide, file, x, y, w, h) {
  // notebook figures have white backgrounds; frame them on a white card
  const s0 = imgSize(file);
  let iw = w - 0.16, ih = (iw * s0.h) / s0.w;
  if (ih > h - 0.16) { ih = h - 0.16; iw = (ih * s0.w) / s0.h; }
  const cx = x + (w - iw - 0.16) / 2;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y, w: iw + 0.16, h: ih + 0.16, rectRadius: 0.06, fill: { color: "FFFFFF" }, line: { color: "FFFFFF" } });
  slide.addImage({ path: file, x: cx + 0.08, y: y + 0.08, w: iw, h: ih });
}

function stat(slide, x, y, value, label, color = FG, size = 40) {
  txt(slide, value, { x, y, w: 3, h: 0.75, fontSize: size, color });
  txt(slide, label.toUpperCase(), { x, y: y + 0.78, w: 3.2, h: 0.4, fontSize: 9, color: MUTED, charSpacing: 1 });
}

function imgSize(file) {
  // read PNG width/height from the header
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function fitImage(slide, file, x, y, maxW, maxH) {
  const s = imgSize(file);
  let w = maxW, h = (maxW * s.h) / s.w;
  if (h > maxH) { h = maxH; w = (maxH * s.w) / s.h; }
  slide.addImage({ path: file, x, y, w, h });
  return { w, h };
}

const RF = "Random Forest (Phase 1)", LSTM = "LSTM (W=30)", CNN = "Temporal CNN (W=30)",
  MLP = "MLP on 57 features", HYB = "Hybrid LSTM + features (W=30)", HCAP = "Hybrid, capped at 125";

// =====================================================================
// 1. Title
{
  const s = pres.addSlide();
  s.background = { color: BG };
  s.addImage({ path: path.join(DECK, "gradient_wide.png"), x: 0, y: 0, w: 8.0, h: H, sizing: { type: "cover", w: 8.0, h: H } });
  s.addShape(pres.shapes.DIAMOND, { x: M, y: 0.42, w: 0.16, h: 0.16, fill: { color: FG }, line: { color: FG } });
  txt(s, "RUL LAB", { x: M + 0.26, y: 0.38, w: 2, h: 0.25, fontSize: 10, bold: true, charSpacing: 1 });
  txt(s, "PREDICTIVE MAINTENANCE  /  PHASE 2", { x: M, y: 3.0, w: 7, h: 0.3, fontSize: 10, color: SOFT, charSpacing: 1 });
  txt(s, "Remaining Useful Life\nPrediction with Sequence Models", { x: M, y: 3.35, w: 7.2, h: 1.3, fontSize: 32, color: FG });
  txt(s, "LSTM, temporal CNN and a hybrid model on raw sensor windows", { x: M, y: 4.8, w: 7, h: 0.35, fontSize: 14, color: SOFT });
  txt(s, "NASA C-MAPSS  /  SUBSET FD001", { x: M, y: 5.35, w: 7, h: 0.3, fontSize: 10, color: CYAN, charSpacing: 1 });
  txt(s, "TEAM T017: Kush Puri, Aadit Vachher, Vachan Gupta, Aabir Sarkar, Kushagra Maheshwari   |   COURSE: CSAI",
    { x: M, y: 5.8, w: 7.2, h: 0.3, fontSize: 9, color: SOFT });
  txt(s, "02", { x: 8.3, y: 1.6, w: 4.6, h: 3.6, fontSize: 220, color: INDIGO, align: "center", valign: "middle" });
  txt(s, "PHASE 2  /  C-MAPSS FD001", { x: M, y: H - 0.45, w: 4, h: 0.22, fontSize: 8, color: SOFT, charSpacing: 1 });
  pageNo = 1;
  s.addNotes(TEXT.notes.title);
}

// 2. Where Phase 1 left us
{
  const s = pres.addSlide();
  chrome(s, "01 / Starting point", "Can a network learn\ndegradation from raw sensors?", { titleW: 5.4 });
  const rf = te(RF);
  stat(s, M, 2.75, f2(rf.RMSE.mean), "RF test RMSE (re-run, cycles)", CYAN);
  stat(s, M + 2.6, 2.75, f2(rf.MAE.mean), "RF test MAE (re-run)", CYAN);
  stat(s, M, 4.15, "0.559", "Importance of cycle number", PINK);
  stat(s, M + 2.6, 4.15, "57", "Hand-built features", FG);
  txt(s, "Phase 1 depended on hand-built features, and its top feature was engine age. Phase 2 asks whether a sequence model can find the signal in the raw readings by itself.",
    { x: M, y: 5.5, w: 5.0, h: 1.0, fontSize: 13, color: SOFT });
  card(s, 6.35, 2.2, 6.4, 4.4);
  txt(s, "SENSOR S11, ALL 100 TRAINING ENGINES", { x: 6.6, y: 2.4, w: 6, h: 0.25, fontSize: 9, color: MUTED, charSpacing: 1 });
  fitImage(s, path.join(DECK, "phase2_s11_all_engines.png"), 6.55, 2.8, 6.0, 2.6);
  txt(s, "Near failure the drift is clear. Far from failure every engine looks the same, so early-life RUL is hard for any model.",
    { x: 6.6, y: 5.65, w: 5.9, h: 0.8, fontSize: 12, color: SOFT });
  s.addNotes(TEXT.notes.start);
}

// 3. What changes, what stays fixed
{
  const s = pres.addSlide();
  chrome(s, "02 / Experimental design", "Same evaluation, new input");
  const rows = [
    ["Target", "RUL = last cycle - current cycle, uncapped"],
    ["Sensors", "Same 14 (std < 0.01 dropped)"],
    ["Split", "Same 5 GroupKFold folds on engine ID"],
    ["Metrics", "RMSE, MAE, MAE early (RUL >= 50) / late (< 50)"],
    ["Test", "Last cycle of 100 engines, used once"],
  ];
  card(s, M, 2.45, 6.0, 4.3);
  txt(s, "KEPT IDENTICAL TO PHASE 1", { x: M + 0.3, y: 2.65, w: 5, h: 0.25, fontSize: 10, color: CYAN, charSpacing: 1 });
  rows.forEach((r, i) => {
    const y = 3.1 + i * 0.68;
    txt(s, r[0], { x: M + 0.3, y, w: 1.3, h: 0.5, fontSize: 14, bold: true });
    txt(s, r[1], { x: M + 1.65, y, w: 4.1, h: 0.6, fontSize: 13, color: SOFT });
  });
  card(s, 6.95, 2.45, 5.8, 4.3);
  txt(s, "CHANGED IN PHASE 2", { x: 7.25, y: 2.65, w: 5, h: 0.25, fontSize: 10, color: PINK, charSpacing: 1 });
  const ch = [
    ["Input", "Before", "1 row of 57 engineered features", "After", "30 cycles x 14 raw sensors"],
    ["Model", "Before", "Random Forest", "After", "LSTM, temporal CNN, MLP, hybrid"],
  ];
  ch.forEach((r, i) => {
    const y = 3.15 + i * 1.65;
    txt(s, r[0], { x: 7.25, y, w: 2, h: 0.4, fontSize: 14, bold: true });
    txt(s, [{ text: "Phase 1  ", options: { color: MUTED } }, { text: r[2], options: { color: SOFT } }], { x: 7.25, y: y + 0.45, w: 5.3, h: 0.4, fontSize: 13 });
    txt(s, [{ text: "Phase 2  ", options: { color: MUTED } }, { text: r[4], options: { color: FG, bold: true } }], { x: 7.25, y: y + 0.88, w: 5.3, h: 0.4, fontSize: 13 });
  });
  txt(s, "So any change in score comes from the model and its input, not from a different test.", { x: M, y: 1.95, w: 11, h: 0.35, fontSize: 13, color: SOFT });
  s.addNotes(TEXT.notes.design);
}

// 4. Windows
{
  const s = pres.addSlide();
  chrome(s, "03 / Sequence windows", "One window per row, inside one engine");
  const sz = fitImage(s, path.join(DECK, "phase2_windows.png"), M, 2.25, 8.4, 4.4);
  const x = M + sz.w + 0.35, w = W - M - x;
  const items = [
    ["30 x 14", "cycles x sensors per window. 30 is below the shortest test history (31), so every test engine gets a full real window."],
    ["Padding", "Rows before cycle 30 repeat the engine's first reading. Every row is still predicted, as in Phase 1."],
    ["Causal", "A window ending at cycle t holds nothing after t and nothing from another engine (checked in code)."],
  ];
  items.forEach((it, i) => {
    const y = 2.3 + i * 1.45;
    txt(s, it[0], { x, y, w, h: 0.4, fontSize: 18, color: i === 2 ? PINK : CYAN });
    txt(s, it[1], { x, y: y + 0.42, w, h: 0.95, fontSize: 12, color: SOFT });
  });
  s.addNotes(TEXT.notes.windows);
}

// 5. What the LSTM sees
{
  const s = pres.addSlide();
  chrome(s, "04 / Model input", "What the LSTM actually sees", { titleW: 7.5 });
  panel(s, 9.0);
  fitImage(s, path.join(DECK, "phase2_window_heatmap.png"), M, 2.35, 8.1, 4.4);
  const x = 9.35, w = 3.4;
  txt(s, "ENGINE 1, TWO WINDOWS", { x, y: 1.0, w, h: 0.25, fontSize: 10, color: FG, charSpacing: 1 });
  const pts = [
    ["Early in life", "Readings sit near the training mean. Mostly noise."],
    ["Near failure", "s2, s3, s4, s8, s11, s13, s15, s17 run high. s7, s12, s20, s21 run low."],
    ["Scaling", "z-score with mean and std from the training engines of each fold only."],
  ];
  pts.forEach((p, i) => {
    const y = 1.55 + i * 1.65;
    txt(s, p[0], { x, y, w, h: 0.4, fontSize: 16, bold: true });
    txt(s, p[1], { x, y: y + 0.42, w, h: 1.1, fontSize: 12, color: SOFT });
  });
  s.addNotes(TEXT.notes.heatmap);
}

// 6. Four models
{
  const s = pres.addSlide();
  chrome(s, "05 / Models", "Four models, one regression head");
  const models = [
    ["LSTM", "MAIN MODEL", "2 stacked LSTM layers, hidden 64. Last hidden state goes to the head.", R.params.LSTM, CYAN],
    ["Temporal CNN", "ARCHITECTURE CHECK", "Two 1D convolutions over time (kernel 5), then average pooling.", R.params.CNN, VIOLET],
    ["MLP", "CONTROL", "No sequence. Only the 57 Phase 1 features at the current row.", R.params.MLP, MUTED],
    ["Hybrid", "LSTM + FEATURES", "LSTM state joined with the 57 engineered features before the head.", R.params.Hybrid, PINK],
  ];
  const cw = (W - 2 * M - 3 * 0.3) / 4;
  models.forEach((m, i) => {
    const x = M + i * (cw + 0.3), y = 2.45;
    card(s, x, y, cw, 3.55);
    txt(s, m[1], { x: x + 0.25, y: y + 0.25, w: cw - 0.5, h: 0.25, fontSize: 9, color: m[4], charSpacing: 1 });
    txt(s, m[0], { x: x + 0.25, y: y + 0.55, w: cw - 0.5, h: 0.5, fontSize: 22, bold: true });
    txt(s, m[2], { x: x + 0.25, y: y + 1.2, w: cw - 0.5, h: 1.3, fontSize: 12, color: SOFT });
    txt(s, m[3].toLocaleString("en-US"), { x: x + 0.25, y: y + 2.6, w: cw - 0.5, h: 0.45, fontSize: 22, color: m[4] });
    txt(s, "PARAMETERS", { x: x + 0.25, y: y + 3.05, w: cw - 0.5, h: 0.25, fontSize: 9, color: MUTED, charSpacing: 1 });
  });
  txt(s, "Shared head: Linear 32, ReLU, dropout, Linear 1. Output is RUL / 100, multiplied back for scoring. Models are kept small: about 16,500 training rows per fold, highly correlated within each engine.",
    { x: M, y: 6.25, w: W - 2 * M, h: 0.6, fontSize: 12, color: MUTED });
  s.addNotes(TEXT.notes.models);
}

// 7. Hybrid architecture
{
  const s = pres.addSlide();
  chrome(s, "06 / Hybrid design", "Hybrid: sequence plus long-term context");
  { const a = imgSize(path.join(DECK, "phase2_hybrid_architecture.png")); const ah = 3.4, aw = ah * a.w / a.h; s.addImage({ path: path.join(DECK, "phase2_hybrid_architecture.png"), x: (W - aw) / 2, y: 2.2, w: aw, h: ah }); }
  const y = 5.75, cw = (W - 2 * M - 0.6) / 3;
  const why = [
    ["The gap", "A 30-cycle window cannot tell how old the engine is or what it looked like when healthy."],
    ["The evidence", "In Phase 1, cycle number (0.559) and drift-from-start features carried most of the RF's signal."],
    ["The fix", "Give the network both: learned short-term dynamics plus long-term context from Phase 1."],
  ];
  why.forEach((w, i) => {
    const x = M + i * (cw + 0.3);
    txt(s, w[0].toUpperCase(), { x, y, w: cw, h: 0.25, fontSize: 10, color: [CYAN, PINK, VIOLET][i], charSpacing: 1 });
    txt(s, w[1], { x, y: y + 0.3, w: cw, h: 0.9, fontSize: 12, color: SOFT });
  });
  s.addNotes(TEXT.notes.hybrid);
}

// 8. Training protocol
{
  const s = pres.addSlide();
  chrome(s, "07 / Training protocol", "Nested split: the test set stays untouched");
  // outer folds diagram
  const x0 = M, y0 = 2.3;
  txt(s, "OUTER: 5 FOLDS BY ENGINE", { x: x0, y: y0, w: 4, h: 0.25, fontSize: 9, color: MUTED, charSpacing: 1 });
  for (let k = 0; k < 5; k++) {
    const x = x0 + k * 0.72;
    for (let j = 0; j < 5; j++) {
      const isTest = j === k;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y: y0 + 0.4 + j * 0.3, w: 0.55, h: 0.22, rectRadius: 0.04,
        fill: { color: isTest ? PINK : INDIGO }, line: { color: isTest ? PINK : INDIGO },
      });
    }
    txt(s, `F${k + 1}`, { x, y: y0 + 1.95, w: 0.55, h: 0.25, fontSize: 9, color: MUTED, align: "center" });
  }
  // inner split
  const xi = M + 4.0;
  txt(s, "INSIDE EACH TRAINING FOLD (80 ENGINES)", { x: xi, y: y0, w: 4.5, h: 0.25, fontSize: 9, color: MUTED, charSpacing: 1 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: xi, y: y0 + 0.45, w: 3.0, h: 0.5, rectRadius: 0.05, fill: { color: INDIGO }, line: { color: INDIGO } });
  txt(s, "64 engines: fit", { x: xi, y: y0 + 0.45, w: 3.0, h: 0.5, fontSize: 12, align: "center", valign: "middle" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: xi + 3.1, y: y0 + 0.45, w: 1.0, h: 0.5, rectRadius: 0.05, fill: { color: "2C3978" }, line: { color: CYAN } });
  txt(s, "16: stop", { x: xi + 3.1, y: y0 + 0.45, w: 1.0, h: 0.5, fontSize: 11, align: "center", valign: "middle", color: CYAN });
  txt(s, [
    { text: "Pink", options: { color: PINK, bold: true } },
    { text: " = 20 held-out engines, scored.  ", options: { color: SOFT } },
    { text: "Every 5th", options: { color: CYAN, bold: true } },
    { text: " training engine picks the early-stopping epoch. Scalers fit on the 64 only.", options: { color: SOFT } },
  ], { x: xi, y: y0 + 1.15, w: 4.3, h: 1.0, fontSize: 12 });
  // settings
  const xs = 9.1;
  card(s, xs, 2.25, W - M - xs, 2.15);
  const set = [["Optimiser", "Adam, lr 1e-3"], ["Weight decay", "1e-4"], ["Batch / epochs", "256 / up to 40"], ["Early stop", "patience 6, best weights"], ["Grad clip", "norm 1.0"]];
  set.forEach((r, i) => {
    txt(s, r[0], { x: xs + 0.2, y: 2.4 + i * 0.37, w: 1.6, h: 0.3, fontSize: 11, color: MUTED });
    txt(s, r[1], { x: xs + 1.8, y: 2.4 + i * 0.37, w: 2.2, h: 0.3, fontSize: 11, color: FG });
  });
  lightFigure(s, path.join(FIG, "nb_learning_curves.png"), M, 4.7, W - 2 * M, 2.2);
  s.addNotes(TEXT.notes.protocol);
}

// 9. CV results chart
{
  const s = pres.addSlide();
  chrome(s, "08 / Cross-validation", "Raw sequences alone do not beat the features", { titleW: 8 });
  const order = [RF, LSTM, "LSTM (W=10)", "LSTM (W=50)", "LSTM (W=30, no dropout)", CNN, MLP, HYB, HCAP];
  const labels = order.map((k) => k.replace(" (Phase 1)", " (P1)"));
  const vals = order.map((k) => cv(k).RMSE.mean);
  s.addChart(pres.charts.BAR, [{ name: "CV RMSE", labels, values: vals }], {
    x: M, y: 2.2, w: 8.0, h: 4.7, barDir: "bar", catAxisOrientation: "maxMin",
    chartColors: order.map((k) => (k === RF ? "9CA9B7" : k === HYB ? "F04ADB" : k === LSTM ? "84D8FF" : "5362D1")),
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: "F4F6FB", dataLabelFontSize: 11, dataLabelFormatCode: "0.00",
    dataLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelColor: "CDD5DF", catAxisLabelFontSize: 11, valAxisLabelColor: "9CA9B7", valAxisLabelFontSize: 10,
    valGridLine: { color: "253440", size: 0.5 }, catGridLine: { style: "none" }, valAxisMinVal: 0,
    catAxisLineShow: false, valAxisLineShow: false, showLegend: false,
    showTitle: true, title: "CV RMSE in cycles (mean of 5 folds, lower is better)", titleColor: "CDD5DF", titleFontSize: 12, titleFontFace: "+mn-lt",
  });
  const x = 9.1, w = W - M - x;
  stat(s, x, 2.3, f2(cv(LSTM).RMSE.mean), "Raw LSTM CV RMSE", CYAN);
  stat(s, x, 3.6, f2(cv(HYB).RMSE.mean), "Hybrid CV RMSE", PINK);
  stat(s, x, 4.9, f2(cv(RF).RMSE.mean), "Random Forest CV RMSE", MUTED);
  txt(s, TEXT.cv_caption, { x, y: 6.15, w, h: 0.75, fontSize: 11, color: SOFT });
  s.addNotes(TEXT.notes.cv);
}

// 10. Ablations
{
  const s = pres.addSlide();
  chrome(s, "09 / Ablations", "One change at a time, same five folds", { titleW: 7.6 });
  panel(s, 8.6);
  const rows = TEXT.ablations; // [{q, a, b, verdict}] built from results
  rows.forEach((r, i) => {
    const y = 2.35 + i * 1.12;
    txt(s, r.q.toUpperCase(), { x: M, y, w: 3.0, h: 0.25, fontSize: 9, color: CYAN, charSpacing: 1 });
    txt(s, r.detail, { x: M, y: y + 0.3, w: 4.3, h: 0.7, fontSize: 13, color: FG });
    txt(s, r.verdict, { x: M + 4.5, y: y + 0.05, w: 3.2, h: 0.95, fontSize: 12, color: SOFT });
  });
  txt(s, "READING THE NUMBERS", { x: 8.95, y: 1.0, w: 3.8, h: 0.25, fontSize: 10, color: FG, charSpacing: 1 });
  txt(s, "Values are CV RMSE in cycles, mean over the same 5 folds, one seed.", { x: 8.95, y: 1.45, w: 3.8, h: 0.7, fontSize: 13, color: FG });
  txt(s, "A difference smaller than the fold-to-fold standard deviation is treated as noise, not as a win.",
    { x: 8.95, y: 2.4, w: 3.8, h: 1.0, fontSize: 13, color: SOFT });
  txt(s, "Ablations reuse the comparison folds, so the best setting's score is slightly optimistic. The test set is the independent check.",
    { x: 8.95, y: 3.75, w: 3.8, h: 1.2, fontSize: 12, color: SOFT });
  s.addNotes(TEXT.notes.ablations);
}

// 11. Error by RUL range
{
  const s = pres.addSlide();
  chrome(s, "10 / Error analysis", "Where in an engine's life each model fails", { titleW: 9 });
  const bins = Object.keys(R.err_by_bin);
  const E = (k) => bins.map((b) => R.err_by_bin[b][k]);
  s.addChart(pres.charts.LINE, [
    { name: "Random Forest (P1)", labels: bins, values: E("rf") },
    { name: "LSTM raw", labels: bins, values: E("lstm") },
    { name: "Hybrid", labels: bins, values: E("hybrid") },
  ], {
    x: M, y: 2.2, w: 8.4, h: 4.7, chartColors: ["9CA9B7", "84D8FF", "F04ADB"], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
    catAxisLabelColor: "CDD5DF", catAxisLabelFontSize: 10, catAxisLabelRotate: -45, valAxisLabelColor: "9CA9B7", valAxisLabelFontSize: 10,
    valGridLine: { color: "253440", size: 0.5 }, catGridLine: { style: "none" }, valAxisMinVal: 0,
    showLegend: true, legendPos: "t", legendColor: "CDD5DF", legendFontSize: 11,
    showValAxisTitle: true, valAxisTitle: "MAE (cycles)", valAxisTitleColor: "9CA9B7", valAxisTitleFontSize: 10,
    showCatAxisTitle: true, catAxisTitle: "True RUL range (cycles), out of fold", catAxisTitleColor: "9CA9B7", catAxisTitleFontSize: 10,
  });
  const x = 9.4, w = W - M - x;
  const b0 = R.err_by_bin["0-25"], b1 = R.err_by_bin["25-50"];
  txt(s, "LAST 25 CYCLES (MAE)", { x, y: 2.3, w, h: 0.25, fontSize: 9, color: MUTED, charSpacing: 1 });
  txt(s, [
    { text: b0.rf.toFixed(1), options: { color: MUTED } }, { text: "  RF    ", options: { color: MUTED, fontSize: 11 } },
    { text: b0.lstm.toFixed(1), options: { color: CYAN } }, { text: "  LSTM    ", options: { color: MUTED, fontSize: 11 } },
    { text: b0.hybrid.toFixed(1), options: { color: PINK } }, { text: "  Hyb", options: { color: MUTED, fontSize: 11 } },
  ], { x, y: 2.6, w, h: 0.5, fontSize: 20 });
  txt(s, "25-50 CYCLES LEFT (MAE)", { x, y: 3.35, w, h: 0.25, fontSize: 9, color: MUTED, charSpacing: 1 });
  txt(s, [
    { text: b1.rf.toFixed(1), options: { color: MUTED } }, { text: "  RF    ", options: { color: MUTED, fontSize: 11 } },
    { text: b1.lstm.toFixed(1), options: { color: CYAN } }, { text: "  LSTM    ", options: { color: MUTED, fontSize: 11 } },
    { text: b1.hybrid.toFixed(1), options: { color: PINK } }, { text: "  Hyb", options: { color: MUTED, fontSize: 11 } },
  ], { x, y: 3.65, w, h: 0.5, fontSize: 20 });
  txt(s, TEXT.err_caption, { x, y: 4.5, w, h: 2.3, fontSize: 12, color: SOFT });
  s.addNotes(TEXT.notes.errors);
}

// 12. Test results
{
  const s = pres.addSlide();
  chrome(s, "11 / Test set", "100 test engines, scored once", { titleW: 8 });
  const hdr = ["Model", "RMSE", "MAE", "MAE early", "MAE late"].map((h) => ({ text: h, options: { bold: true, color: CYAN, fontSize: 11 } }));
  const keys = [RF, LSTM, CNN, MLP, HYB];
  const short = { [RF]: "Random Forest (Phase 1)", [LSTM]: "LSTM, raw windows", [CNN]: "Temporal CNN", [MLP]: "MLP, 57 features", [HYB]: "Hybrid LSTM + features" };
  const fmt = (m) => (m.std > 0 ? `${f2(m.mean)} ± ${f2(m.std)}` : f2(m.mean));
  const best = Math.min(...keys.map((k) => te(k).RMSE.mean));
  const rows = [hdr].concat(keys.map((k) => {
    const t = te(k), hl = t.RMSE.mean === best;
    const o = { color: hl ? FG : SOFT, bold: hl, fontSize: 11 };
    return [short[k], fmt(t.RMSE), fmt(t.MAE), fmt(t["MAE early"]), fmt(t["MAE late"])].map((v) => ({ text: v, options: o }));
  }));
  s.addTable(rows, {
    x: M, y: 2.25, w: W - 2 * M, colW: [3.7, 2.1, 2.1, 2.1, 2.133], fontFace: FONT, color: SOFT,
    border: { type: "solid", color: LINE, pt: 0.75 }, fill: { color: BG }, rowH: 0.45, valign: "middle",
  });
  txt(s, TEXT.test_caption, { x: M, y: 5.3, w: 6.4, h: 1.0, fontSize: 14, color: FG });
  txt(s, `Deep models: mean ± std over 3 seeds, trained on all 100 engines for the median CV best epoch. ${R.n_test_early} early and ${R.n_test_late} late test engines. Not comparable to CV (one cycle per engine).`,
    { x: 7.4, y: 5.3, w: W - M - 7.4, h: 1.0, fontSize: 11, color: MUTED });
  s.addNotes(TEXT.notes.test);
}

// 12b. Test predictions per engine
{
  const s = pres.addSlide();
  chrome(s, "11 / Test set", "Engine by engine: good near failure, noisy early", { titleW: 10 });
  lightFigure(s, path.join(FIG, "nb_test_predictions.png"), M, 2.15, W - 2 * M, 3.8);
  txt(s, TEXT.testplot_caption, { x: M, y: 6.2, w: W - 2 * M, h: 0.5, fontSize: 12, color: SOFT });
  s.addNotes(TEXT.notes.testplot);
}

// 13. Takeaways + limits
{
  const s = pres.addSlide();
  chrome(s, "12 / Conclusions", "What Phase 2 shows", { titleW: 7.5 });
  panel(s, 8.6);
  TEXT.findings.forEach((f, i) => {
    const y = 2.35 + i * 1.12;
    txt(s, String(i + 1).padStart(2, "0"), { x: M, y, w: 0.6, h: 0.4, fontSize: 16, color: [CYAN, PINK, VIOLET, CYAN][i % 4] });
    txt(s, f, { x: M + 0.7, y, w: 7.1, h: 1.0, fontSize: 13, color: SOFT });
  });
  txt(s, "LIMITATIONS", { x: 8.95, y: 1.0, w: 3.8, h: 0.25, fontSize: 10, color: FG, charSpacing: 1 });
  const lim = ["No hyperparameter search; one seed for ablations", "Ablations reuse the comparison folds", "Test = 100 engines, one cycle each", "FD001 only: one condition, one fault mode", "Uncapped target: not comparable to published RMSE"];
  lim.forEach((l, i) => txt(s, l, { x: 8.95, y: 1.5 + i * 0.75, w: 3.8, h: 0.65, fontSize: 13, color: FG }));
  s.addNotes(TEXT.notes.conclusions);
}

// 14. Thank you
{
  const s = pres.addSlide();
  s.background = { color: BG };
  s.addImage({ path: path.join(DECK, "gradient_wide.png"), x: 0, y: 0, w: W, h: H, sizing: { type: "cover", w: W, h: H } });
  s.addShape(pres.shapes.DIAMOND, { x: M, y: 0.42, w: 0.16, h: 0.16, fill: { color: FG }, line: { color: FG } });
  txt(s, "RUL LAB", { x: M + 0.26, y: 0.38, w: 2, h: 0.25, fontSize: 10, bold: true, charSpacing: 1 });
  txt(s, "PHASE 2 COMPLETE", { x: M, y: 3.0, w: 6, h: 0.3, fontSize: 10, color: SOFT, charSpacing: 1 });
  txt(s, "Thank you.", { x: M, y: 3.35, w: 6, h: 0.9, fontSize: 44 });
  txt(s, "Questions?", { x: M, y: 4.3, w: 6, h: 0.5, fontSize: 20, color: CYAN });
  txt(s, "Notebook: notebooks/02_phase2_deep_learning.ipynb", { x: M, y: 5.3, w: 7, h: 0.3, fontSize: 11, color: SOFT });
  txt(s, "PHASE 2  /  C-MAPSS FD001", { x: M, y: H - 0.45, w: 4, h: 0.22, fontSize: 8, color: SOFT, charSpacing: 1 });
  s.addNotes(TEXT.notes.end);
}

const out = path.join(ROOT, "reports", "phase2_presentation.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("Wrote", out));
