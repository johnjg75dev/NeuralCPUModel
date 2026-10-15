/* ------------------------------------------------------------------
   micro-tape: a tiny reverse-mode autodiff engine over dense 2-D
   Float32Array tensors. Every op records a backward closure onto a
   module-level tape; backward() replays it in reverse.
------------------------------------------------------------------ */

export class T {
  data: Float32Array;
  grad: Float32Array;
  r: number;
  c: number;
  name?: string;
  constructor(r: number, c: number, name?: string) {
    this.r = r;
    this.c = c;
    this.data = new Float32Array(r * c);
    this.grad = new Float32Array(r * c);
    this.name = name;
  }
  get n() {
    return this.r * this.c;
  }
  at(i: number, j: number) {
    return this.data[i * this.c + j];
  }
}

type Op = { backward: () => void };
let tape: Op[] = [];
let gradEnabled = true;

export function newGraph() {
  tape = [];
}
export function tapeLength() {
  return tape.length;
}
export function setGradEnabled(v: boolean) {
  gradEnabled = v;
}
function rec(b: () => void) {
  if (gradEnabled) tape.push({ backward: b });
}

export function zeroGrad(params: T[]) {
  for (const p of params) p.grad.fill(0);
}

export function backward(loss: T) {
  loss.grad.fill(0);
  loss.grad[0] = 1;
  for (let i = tape.length - 1; i >= 0; i--) tape[i].backward();
}

/* ---------------------------- helpers ---------------------------- */

let gaussSpare: number | null = null;
export function randn(): number {
  if (gaussSpare !== null) {
    const v = gaussSpare;
    gaussSpare = null;
    return v;
  }
  let u = 0,
    v = 0,
    s = 0;
  do {
    u = Math.random() * 2 - 1;
    v = Math.random() * 2 - 1;
    s = u * u + v * v;
  } while (s === 0 || s >= 1);
  const m = Math.sqrt((-2 * Math.log(s)) / s);
  gaussSpare = v * m;
  return u * m;
}

export function param(r: number, c: number, std: number, init: 'gauss' | 'ones' | 'zeros' = 'gauss', name?: string): T {
  const t = new T(r, c, name);
  if (init === 'ones') t.data.fill(1);
  else if (init === 'gauss') for (let i = 0; i < t.data.length; i++) t.data[i] = randn() * std;
  return t;
}

/* ------------------------------ ops ------------------------------ */

/** C = A @ B   [r,k] x [k,c] */
export function matmul(A: T, B: T): T {
  const r = A.r,
    k = A.c,
    c = B.c;
  const out = new T(r, c);
  const a = A.data,
    b = B.data,
    o = out.data;
  for (let i = 0; i < r; i++) {
    const ai = i * k,
      oi = i * c;
    for (let p = 0; p < k; p++) {
      const av = a[ai + p];
      if (av === 0) continue;
      const bp = p * c;
      for (let j = 0; j < c; j++) o[oi + j] += av * b[bp + j];
    }
  }
  rec(() => {
    const go = out.grad,
      dA = A.grad,
      dB = B.grad;
    for (let i = 0; i < r; i++) {
      const gi = i * c,
        ai = i * k;
      for (let p = 0; p < k; p++) {
        let s = 0;
        const bp = p * c;
        for (let j = 0; j < c; j++) s += go[gi + j] * b[bp + j];
        dA[ai + p] += s;
      }
    }
    for (let p = 0; p < k; p++) {
      const bp = p * c;
      for (let i = 0; i < r; i++) {
        const av = a[i * k + p];
        if (av === 0) continue;
        const gi = i * c;
        for (let j = 0; j < c; j++) dB[bp + j] += av * go[gi + j];
      }
    }
  });
  return out;
}

/** C = A @ B^T   [r,k] x [c,k] -> [r,c] */
export function matmulT(A: T, B: T): T {
  const r = A.r,
    k = A.c,
    c = B.r;
  const out = new T(r, c);
  const a = A.data,
    b = B.data,
    o = out.data;
  for (let i = 0; i < r; i++) {
    const ai = i * k,
      oi = i * c;
    for (let j = 0; j < c; j++) {
      const bj = j * k;
      let s = 0;
      for (let p = 0; p < k; p++) s += a[ai + p] * b[bj + p];
      o[oi + j] = s;
    }
  }
  rec(() => {
    const go = out.grad,
      dA = A.grad,
      dB = B.grad;
    // dA = go @ B   [r,c] x [c,k]
    for (let i = 0; i < r; i++) {
      const gi = i * c,
        ai = i * k;
      for (let p = 0; p < k; p++) {
        let s = 0;
        for (let j = 0; j < c; j++) s += go[gi + j] * b[j * k + p];
        dA[ai + p] += s;
      }
    }
    // dB = go^T @ A  [c,r] x [r,k]
    for (let j = 0; j < c; j++) {
      const bj = j * k;
      for (let i = 0; i < r; i++) {
        const g = go[i * c + j];
        if (g === 0) continue;
        const ai = i * k;
        for (let p = 0; p < k; p++) dB[bj + p] += g * a[ai + p];
      }
    }
  });
  return out;
}

/** C = A^T @ B   [k,r] x [k,c] -> [r,c] */
export function matmulAT(A: T, B: T): T {
  const k = A.r,
    r = A.c,
    c = B.c;
  const out = new T(r, c);
  const a = A.data,
    b = B.data,
    o = out.data;
  for (let p = 0; p < k; p++) {
    const ap = p * r,
      bp = p * c;
    for (let i = 0; i < r; i++) {
      const av = a[ap + i];
      if (av === 0) continue;
      const oi = i * c;
      for (let j = 0; j < c; j++) o[oi + j] += av * b[bp + j];
    }
  }
  rec(() => {
    const go = out.grad,
      dA = A.grad,
      dB = B.grad;
    // dA = go @ B^T -> dA[p,i] = sum_j go[i,j] B[p,j]
    for (let p = 0; p < k; p++) {
      const ap = p * r,
        bp = p * c;
      for (let i = 0; i < r; i++) {
        let s = 0;
        const gi = i * c;
        for (let j = 0; j < c; j++) s += go[gi + j] * b[bp + j];
        dA[ap + i] += s;
      }
    }
    // dB = A @ go
    for (let p = 0; p < k; p++) {
      const ap = p * r,
        bp = p * c;
      for (let i = 0; i < r; i++) {
        const av = a[ap + i];
        if (av === 0) continue;
        const gi = i * c;
        for (let j = 0; j < c; j++) dB[bp + j] += av * go[gi + j];
      }
    }
  });
  return out;
}

export function addBias(X: T, b: T): T {
  const out = new T(X.r, X.c);
  const x = X.data,
    bb = b.data,
    o = out.data,
    c = X.c;
  for (let i = 0; i < X.r; i++) {
    const oi = i * c;
    for (let j = 0; j < c; j++) o[oi + j] = x[oi + j] + bb[j];
  }
  rec(() => {
    const go = out.grad,
      dX = X.grad,
      db = b.grad;
    for (let i = 0; i < go.length; i++) dX[i] += go[i];
    for (let i = 0; i < X.r; i++) {
      const gi = i * c;
      for (let j = 0; j < c; j++) db[j] += go[gi + j];
    }
  });
  return out;
}

export function add(A: T, B: T): T {
  const out = new T(A.r, A.c);
  for (let i = 0; i < out.data.length; i++) out.data[i] = A.data[i] + B.data[i];
  rec(() => {
    const go = out.grad;
    for (let i = 0; i < go.length; i++) {
      A.grad[i] += go[i];
      B.grad[i] += go[i];
    }
  });
  return out;
}

/** X + M where M is a constant (mask) that never receives gradient */
export function addConst(X: T, M: T): T {
  const out = new T(X.r, X.c);
  for (let i = 0; i < out.data.length; i++) out.data[i] = X.data[i] + M.data[i];
  rec(() => {
    const go = out.grad;
    for (let i = 0; i < go.length; i++) X.grad[i] += go[i];
  });
  return out;
}

/** element-wise product (Hadamard) */
export function mul(A: T, B: T): T {
  const out = new T(A.r, A.c);
  const a = A.data,
    b = B.data,
    o = out.data;
  for (let i = 0; i < o.length; i++) o[i] = a[i] * b[i];
  rec(() => {
    const go = out.grad;
    for (let i = 0; i < go.length; i++) {
      A.grad[i] += go[i] * b[i];
      B.grad[i] += go[i] * a[i];
    }
  });
  return out;
}

export function scale(X: T, s: number): T {
  const out = new T(X.r, X.c);
  for (let i = 0; i < out.data.length; i++) out.data[i] = X.data[i] * s;
  rec(() => {
    const go = out.grad;
    for (let i = 0; i < go.length; i++) X.grad[i] += go[i] * s;
  });
  return out;
}

export function relu(X: T): T {
  const out = new T(X.r, X.c);
  const x = X.data,
    o = out.data;
  for (let i = 0; i < x.length; i++) o[i] = x[i] > 0 ? x[i] : 0;
  rec(() => {
    const go = out.grad;
    for (let i = 0; i < go.length; i++) if (x[i] > 0) X.grad[i] += go[i];
  });
  return out;
}

/** fast gelu: x * sigmoid(1.702 x) */
export function gelu(X: T): T {
  const out = new T(X.r, X.c);
  const x = X.data,
    o = out.data;
  const sig = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const s = 1 / (1 + Math.exp(-1.702 * x[i]));
    sig[i] = s;
    o[i] = x[i] * s;
  }
  rec(() => {
    const go = out.grad;
    for (let i = 0; i < go.length; i++) {
      const s = sig[i];
      X.grad[i] += go[i] * (s + x[i] * 1.702 * s * (1 - s));
    }
  });
  return out;
}

export function sigmoid(X: T): T {
  const out = new T(X.r, X.c);
  const o = out.data;
  for (let i = 0; i < X.data.length; i++) o[i] = 1 / (1 + Math.exp(-X.data[i]));
  rec(() => {
    const go = out.grad;
    for (let i = 0; i < go.length; i++) X.grad[i] += go[i] * o[i] * (1 - o[i]);
  });
  return out;
}

export function tanh(X: T): T {
  const out = new T(X.r, X.c);
  const o = out.data;
  for (let i = 0; i < X.data.length; i++) o[i] = Math.tanh(X.data[i]);
  rec(() => {
    const go = out.grad;
    for (let i = 0; i < go.length; i++) X.grad[i] += go[i] * (1 - o[i] * o[i]);
  });
  return out;
}

/** row-wise softmax */
export function softmax(X: T): T {
  const out = new T(X.r, X.c);
  const x = X.data,
    o = out.data,
    c = X.c;
  for (let i = 0; i < X.r; i++) {
    const base = i * c;
    let m = -Infinity;
    for (let j = 0; j < c; j++) if (x[base + j] > m) m = x[base + j];
    let s = 0;
    for (let j = 0; j < c; j++) {
      const e = Math.exp(x[base + j] - m);
      o[base + j] = e;
      s += e;
    }
    const inv = 1 / (s || 1);
    for (let j = 0; j < c; j++) o[base + j] *= inv;
  }
  rec(() => {
    const go = out.grad;
    for (let i = 0; i < X.r; i++) {
      const base = i * c;
      let dot = 0;
      for (let j = 0; j < c; j++) dot += go[base + j] * o[base + j];
      for (let j = 0; j < c; j++) X.grad[base + j] += o[base + j] * (go[base + j] - dot);
    }
  });
  return out;
}

export function layerNorm(X: T, g: T, b: T, eps = 1e-5): T {
  const r = X.r,
    c = X.c;
  const out = new T(r, c);
  const x = X.data,
    o = out.data,
    gg = g.data,
    bb = b.data;
  const xhat = new Float32Array(r * c);
  const inv = new Float32Array(r);
  for (let i = 0; i < r; i++) {
    const base = i * c;
    let mean = 0;
    for (let j = 0; j < c; j++) mean += x[base + j];
    mean /= c;
    let v = 0;
    for (let j = 0; j < c; j++) {
      const d = x[base + j] - mean;
      v += d * d;
    }
    v /= c;
    const s = 1 / Math.sqrt(v + eps);
    inv[i] = s;
    for (let j = 0; j < c; j++) {
      const h = (x[base + j] - mean) * s;
      xhat[base + j] = h;
      o[base + j] = h * gg[j] + bb[j];
    }
  }
  rec(() => {
    const go = out.grad,
      dX = X.grad,
      dg = g.grad,
      db = b.grad;
    for (let i = 0; i < r; i++) {
      const base = i * c,
        s = inv[i];
      let s1 = 0,
        s2 = 0;
      for (let j = 0; j < c; j++) {
        const dxh = go[base + j] * gg[j];
        s1 += dxh;
        s2 += dxh * xhat[base + j];
        dg[j] += go[base + j] * xhat[base + j];
        db[j] += go[base + j];
      }
      const invC = 1 / c;
      for (let j = 0; j < c; j++) {
        const dxh = go[base + j] * gg[j];
        dX[base + j] += s * (dxh - s1 * invC - s2 * invC * xhat[base + j]);
      }
    }
  });
  return out;
}

export function concatCols(list: T[]): T {
  const r = list[0].r;
  let c = 0;
  for (const t of list) c += t.c;
  const out = new T(r, c);
  const offs: number[] = [];
  let acc = 0;
  for (const t of list) {
    offs.push(acc);
    acc += t.c;
    for (let i = 0; i < r; i++) {
      const ob = i * c + offs[offs.length - 1];
      for (let j = 0; j < t.c; j++) out.data[ob + j] = t.data[i * t.c + j];
    }
  }
  rec(() => {
    const go = out.grad;
    list.forEach((t, ti) => {
      const off = offs[ti];
      for (let i = 0; i < r; i++) {
        const ob = i * c + off;
        for (let j = 0; j < t.c; j++) t.grad[i * t.c + j] += go[ob + j];
      }
    });
  });
  return out;
}

export function concatRows(list: T[]): T {
  const c = list[0].c;
  let r = 0;
  for (const t of list) r += t.r;
  const out = new T(r, c);
  let acc = 0;
  for (const t of list) {
    out.data.set(t.data, acc * c);
    acc += t.r;
  }
  rec(() => {
    let off = 0;
    for (const t of list) {
      const slice = out.grad.subarray(off * c, (off + t.r) * c);
      for (let i = 0; i < slice.length; i++) t.grad[i] += slice[i];
      off += t.r;
    }
  });
  return out;
}

/** take rows [i0,i1) of X (differentiable view) */
export function sliceRows(X: T, i0: number, i1: number): T {
  const c = X.c;
  const out = new T(i1 - i0, c);
  out.data.set(X.data.subarray(i0 * c, i1 * c));
  rec(() => {
    const go = out.grad,
      dX = X.grad;
    const base = i0 * c;
    for (let i = 0; i < go.length; i++) dX[base + i] += go[i];
  });
  return out;
}

/** embedding lookup: rows of W selected by idx */
export function embLookup(W: T, idx: Int32Array): T {
  const r = idx.length,
    c = W.c;
  const out = new T(r, c);
  for (let i = 0; i < r; i++) {
    const src = idx[i] * c;
    for (let j = 0; j < c; j++) out.data[i * c + j] = W.data[src + j];
  }
  rec(() => {
    const go = out.grad,
      dW = W.grad;
    for (let i = 0; i < r; i++) {
      const dst = idx[i] * c,
        src = i * c;
      for (let j = 0; j < c; j++) dW[dst + j] += go[src + j];
    }
  });
  return out;
}

/** masked mean cross-entropy. logits [S,V], targets [S], weights [S] */
export function crossEntropy(logits: T, targets: Int32Array, weights: Float32Array): T {
  const S = logits.r,
    V = logits.c;
  const p = softmax(logits);
  const out = new T(1, 1);
  let sum = 0,
    wsum = 0;
  for (let i = 0; i < S; i++) {
    const w = weights[i];
    if (w === 0) continue;
    const pv = Math.max(1e-9, p.data[i * V + targets[i]]);
    sum += -w * Math.log(pv);
    wsum += w;
  }
  out.data[0] = wsum > 0 ? sum / wsum : 0;
  rec(() => {
    const g = out.grad[0] / (wsum > 0 ? wsum : 1);
    const dL = logits.grad;
    for (let i = 0; i < S; i++) {
      const w = weights[i];
      if (w === 0) continue;
      const base = i * V;
      for (let j = 0; j < V; j++) dL[base + j] += g * w * p.data[base + j];
      dL[base + targets[i]] -= g * w;
    }
  });
  return out;
}

/** softmax probabilities without touching the tape (inference helper) */
export function softmaxRowsRaw(x: Float32Array, r: number, c: number): Float32Array {
  const o = new Float32Array(r * c);
  for (let i = 0; i < r; i++) {
    const base = i * c;
    let m = -Infinity;
    for (let j = 0; j < c; j++) if (x[base + j] > m) m = x[base + j];
    let s = 0;
    for (let j = 0; j < c; j++) {
      const e = Math.exp(x[base + j] - m);
      o[base + j] = e;
      s += e;
    }
    for (let j = 0; j < c; j++) o[base + j] /= s || 1;
  }
  return o;
}

export function argmaxRow(x: Float32Array, row: number, c: number): number {
  let best = 0,
    bv = -Infinity;
  for (let j = 0; j < c; j++) {
    const v = x[row * c + j];
    if (v > bv) {
      bv = v;
      best = j;
    }
  }
  return best;
}

/* ------------------------- constant caches ------------------------ */

const maskCache = new Map<string, T>();
export function causalMask(n: number, m?: number): T {
  const key = n + ':' + (m ?? n);
  let t = maskCache.get(key);
  if (t) return t;
  const cols = m ?? n;
  t = new T(n, cols);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < cols; j++) if (j > i + (cols - n)) t.data[i * cols + j] = -1e9;
  maskCache.set(key, t);
  return t;
}

const posCache = new Map<string, T>();
export function sinusoidalPos(n: number, d: number): T {
  const key = n + ':' + d;
  let t = posCache.get(key);
  if (t) return t;
  t = new T(n, d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j += 2) {
      const f = 1 / Math.pow(10000, j / d);
      t.data[i * d + j] = Math.sin(i * f);
      if (j + 1 < d) t.data[i * d + j + 1] = Math.cos(i * f);
    }
  posCache.set(key, t);
  return t;
}
