import {
  T,
  param,
  matmul,
  matmulT,
  addBias,
  add,
  addConst,
  scale,
  mul,
  gelu,
  relu,
  sigmoid,
  tanh,
  softmax,
  layerNorm,
  concatCols,
  concatRows,
  sliceRows,
  embLookup,
  crossEntropy,
  causalMask,
  sinusoidalPos,
  newGraph,
  backward,
  setGradEnabled,
} from './tensor';

export type ArchKind = 'decoder' | 'seq2seq' | 'slot';

export interface ModelCfg {
  arch: ArchKind;
  dModel: number;
  nHeads: number;
  nLayers: number;
  dff: number;
  maxLen: number;
  slots: number;
  activation: 'gelu' | 'relu';
  posMode: 'learned' | 'sinusoid';
}

export const DEFAULT_CFG: ModelCfg = {
  arch: 'decoder',
  dModel: 32,
  nHeads: 4,
  nLayers: 2,
  dff: 96,
  maxLen: 112,
  slots: 4,
  activation: 'gelu',
  posMode: 'learned',
};

/* ------------------------- diagnostics bus ------------------------- */

export interface AttnRec {
  layer: number;
  head: number;
  tag: string;
  rows: number;
  cols: number;
  data: Float32Array;
  qTok: string[];
  kTok: string[];
}

export interface FfnSnap {
  layer: number;
  W1: T;
  b1: T;
  W2: T;
  input: T;
  hidden: T;
}

export const diag = {
  enabled: false,
  ffnLayer: 0,
  attn: [] as AttnRec[],
  ffn: null as FfnSnap | null,
  embOut: null as T | null,
  hidden: null as T | null,
  logits: null as T | null,
  memTok: [] as string[],
};

export function resetDiag() {
  diag.attn = [];
  diag.ffn = null;
  diag.embOut = null;
  diag.hidden = null;
  diag.logits = null;
  diag.memTok = [];
}

function noGrad<T>(fn: () => T): T {
  const prev = true;
  setGradEnabled(false);
  try {
    return fn();
  } finally {
    setGradEnabled(prev);
  }
}

/* ------------------------------ blocks ------------------------------ */

interface AttnBlock {
  wq: T[];
  bq: T[];
  wk: T[];
  bk: T[];
  wv: T[];
  bv: T[];
  wo: T;
  bo: T;
}
interface FFN {
  w1: T;
  b1: T;
  w2: T;
  b2: T;
}
interface LN {
  g: T;
  b: T;
}
interface Layer {
  self: AttnBlock;
  cross?: AttnBlock;
  ffn: FFN;
  ln1: LN;
  ln2: LN;
  ln3?: LN;
}

export class Model {
  cfg: ModelCfg;
  V: number;
  params: T[] = [];
  emb: T;
  posEmb: T;
  layers: Layer[] = [];
  encLayers: Layer[] = [];
  lnfEnc!: LN;
  lnf: LN;
  /* slot-recurrent */
  slotInit!: T;
  sq!: { wq: T; bq: T; wk: T; bk: T; wv: T; bv: T };
  gru!: { wz: T; bz: T; wr: T; br: T; wn: T; bn: T; un: T };
  readQ!: T;
  lnSlot!: LN;

  constructor(cfg: ModelCfg, V: number) {
    this.cfg = cfg;
    this.V = V;
    const D = cfg.dModel;
    const std = Math.min(0.25, 1 / Math.sqrt(D));
    this.emb = param(V, D, 0.4, 'gauss', 'emb');
    this.posEmb = param(cfg.maxLen, D, 0.02, 'gauss', 'pos');
    this.lnf = { g: param(1, D, 0, 'ones', 'lnf.g'), b: param(1, D, 0, 'zeros', 'lnf.b') };

    if (cfg.arch === 'slot') {
      const S = cfg.slots;
      this.slotInit = param(S, D, 0.3, 'gauss', 'slotInit');
      this.sq = {
        wq: param(D, D, std, 'gauss', 'sq.wq'),
        bq: param(1, D, 0, 'zeros', 'sq.bq'),
        wk: param(D, D, std, 'gauss', 'sq.wk'),
        bk: param(1, D, 0, 'zeros', 'sq.bk'),
        wv: param(D, D, std, 'gauss', 'sq.wv'),
        bv: param(1, D, 0, 'zeros', 'sq.bv'),
      };
      this.gru = {
        wz: param(2 * D, D, std, 'gauss', 'gru.wz'),
        bz: param(1, D, 0, 'zeros', 'gru.bz'),
        wr: param(2 * D, D, std, 'gauss', 'gru.wr'),
        br: param(1, D, 0, 'zeros', 'gru.br'),
        wn: param(2 * D, D, std, 'gauss', 'gru.wn'),
        bn: param(1, D, 0, 'zeros', 'gru.bn'),
        un: param(D, D, std, 'gauss', 'gru.un'),
      };
      this.readQ = param(1, D, std, 'gauss', 'readQ');
      this.lnSlot = { g: param(1, D, 0, 'ones', 'lnSlot.g'), b: param(1, D, 0, 'zeros', 'lnSlot.b') };
      this.params.push(
        this.emb,
        this.posEmb,
        this.lnf.g,
        this.lnf.b,
        this.slotInit,
        ...Object.values(this.sq),
        ...Object.values(this.gru),
        this.readQ,
        this.lnSlot.g,
        this.lnSlot.b,
      );
    } else {
      const nLayers = cfg.arch === 'seq2seq' ? cfg.nLayers : cfg.nLayers;
      for (let l = 0; l < nLayers; l++) {
        const L: Layer = {
          self: this.mkAttn(std, `L${l}.self`),
          ffn: this.mkFfn(`L${l}.ffn`),
          ln1: { g: param(1, D, 0, 'ones', `L${l}.ln1.g`), b: param(1, D, 0, 'zeros', `L${l}.ln1.b`) },
          ln2: { g: param(1, D, 0, 'ones', `L${l}.ln2.g`), b: param(1, D, 0, 'zeros', `L${l}.ln2.b`) },
        };
        if (cfg.arch === 'seq2seq') {
          L.cross = this.mkAttn(std, `L${l}.cross`);
          L.ln3 = { g: param(1, D, 0, 'ones', `L${l}.ln3.g`), b: param(1, D, 0, 'zeros', `L${l}.ln3.b`) };
          this.params.push(L.ln3.g, L.ln3.b);
          const E: Layer = {
            self: this.mkAttn(std, `E${l}.self`),
            ffn: this.mkFfn(`E${l}.ffn`),
            ln1: { g: param(1, D, 0, 'ones', `E${l}.ln1.g`), b: param(1, D, 0, 'zeros', `E${l}.ln1.b`) },
            ln2: { g: param(1, D, 0, 'ones', `E${l}.ln2.g`), b: param(1, D, 0, 'zeros', `E${l}.ln2.b`) },
          };
          this.encLayers.push(E);
          this.params.push(
            ...E.self.wq, ...E.self.bq, ...E.self.wk, ...E.self.bk, ...E.self.wv, ...E.self.bv,
            E.self.wo, E.self.bo, E.ffn.w1, E.ffn.b1, E.ffn.w2, E.ffn.b2,
            E.ln1.g, E.ln1.b, E.ln2.g, E.ln2.b,
          );
        }
        this.layers.push(L);
        this.params.push(
          ...L.self.wq,
          ...L.self.bq,
          ...L.self.wk,
          ...L.self.bk,
          ...L.self.wv,
          ...L.self.bv,
          L.self.wo,
          L.self.bo,
          L.ffn.w1,
          L.ffn.b1,
          L.ffn.w2,
          L.ffn.b2,
          L.ln1.g,
          L.ln1.b,
          L.ln2.g,
          L.ln2.b,
        );
        if (L.cross)
          this.params.push(
            ...L.cross.wq,
            ...L.cross.bq,
            ...L.cross.wk,
            ...L.cross.bk,
            ...L.cross.wv,
            ...L.cross.bv,
            L.cross.wo,
            L.cross.bo,
          );
      }
      if (cfg.arch === 'seq2seq') {
        this.lnfEnc = { g: param(1, D, 0, 'ones', 'lnfEnc.g'), b: param(1, D, 0, 'zeros', 'lnfEnc.b') };
        this.params.push(this.lnfEnc.g, this.lnfEnc.b);
      }
      this.params.push(this.emb, this.posEmb, this.lnf.g, this.lnf.b);
    }
  }

  private mkAttn(std: number, tag: string): AttnBlock {
    const D = this.cfg.dModel;
    const H = this.cfg.nHeads;
    const Dh = Math.max(1, Math.floor(D / H));
    const b: AttnBlock = { wq: [], bq: [], wk: [], bk: [], wv: [], bv: [], wo: param(D, H * Dh, std, 'gauss', tag + '.wo'), bo: param(1, H * Dh, 0, 'zeros', tag + '.bo') };
    for (let h = 0; h < H; h++) {
      b.wq.push(param(D, Dh, std, 'gauss', `${tag}.h${h}.wq`));
      b.bq.push(param(1, Dh, 0, 'zeros', `${tag}.h${h}.bq`));
      b.wk.push(param(D, Dh, std, 'gauss', `${tag}.h${h}.wk`));
      b.bk.push(param(1, Dh, 0, 'zeros', `${tag}.h${h}.bk`));
      b.wv.push(param(D, Dh, std, 'gauss', `${tag}.h${h}.wv`));
      b.bv.push(param(1, Dh, 0, 'zeros', `${tag}.h${h}.bv`));
    }
    return b;
  }

  private mkFfn(tag: string): FFN {
    const D = this.cfg.dModel;
    const F = this.cfg.dff;
    return {
      w1: param(D, F, Math.min(0.25, 1 / Math.sqrt(D)), 'gauss', tag + '.w1'),
      b1: param(1, F, 0, 'zeros', tag + '.b1'),
      w2: param(F, D, Math.min(0.25, 1 / Math.sqrt(F)), 'gauss', tag + '.w2'),
      b2: param(1, D, 0, 'zeros', tag + '.b2'),
    };
  }

  get dh() {
    return Math.max(1, Math.floor(this.cfg.dModel / this.cfg.nHeads));
  }

  countParams() {
    let n = 0;
    for (const p of this.params) n += p.n;
    return n;
  }

  private embed(ids: Int32Array | number[]): T {
    const arr = ids instanceof Int32Array ? ids : Int32Array.from(ids);
    const S = arr.length;
    const x = embLookup(this.emb, arr);
    if (this.cfg.posMode === 'learned') return add(x, sliceRows(this.posEmb, 0, Math.min(S, this.cfg.maxLen)));
    return addConst(x, sinusoidalPos(S, this.cfg.dModel));
  }

  private attention(
    q: T,
    kv: T,
    A: AttnBlock,
    mask: T | null,
    layer: number,
    tag: string,
    qTok: string[],
    kTok: string[],
  ): T {
    const H = this.cfg.nHeads;
    const Dh = this.dh;
    const outs: T[] = [];
    for (let h = 0; h < H; h++) {
      const qh = addBias(matmul(q, A.wq[h]), A.bq[h]);
      const kh = addBias(matmul(kv, A.wk[h]), A.bk[h]);
      const vh = addBias(matmul(kv, A.wv[h]), A.bv[h]);
      let s = scale(matmulT(qh, kh), 1 / Math.sqrt(Dh));
      if (mask) s = addConst(s, mask);
      const a = softmax(s);
      if (diag.enabled) {
        diag.attn.push({ layer, head: h, tag, rows: a.r, cols: a.c, data: a.data.slice(), qTok, kTok });
      }
      outs.push(matmul(a, vh));
    }
    return addBias(matmul(concatCols(outs), A.wo), A.bo);
  }

  private ffn(x: T, F: FFN, layer: number): T {
    const h0 = addBias(matmul(x, F.w1), F.b1);
    const h = this.cfg.activation === 'relu' ? relu(h0) : gelu(h0);
    if (diag.enabled && layer === diag.ffnLayer) {
      diag.ffn = { layer, W1: F.w1, b1: F.b1, W2: F.w2, input: x, hidden: h };
    }
    return addBias(matmul(h, F.w2), F.b2);
  }

  private block(x: T, L: Layer, mask: T | null, mem: T | null, layer: number, qTok: string[], kTok: string[]): T {
    const h1 = layerNorm(x, L.ln1.g, L.ln1.b);
    x = add(x, this.attention(h1, h1, L.self, mask, layer, 'self', qTok, kTok));
    if (mem && L.cross && L.ln3) {
      const h2 = layerNorm(x, L.ln3.g, L.ln3.b);
      x = add(x, this.attention(h2, mem, L.cross, null, layer, 'cross', qTok, diag.memTok));
    }
    const h3 = layerNorm(x, L.ln2.g, L.ln2.b);
    x = add(x, this.ffn(h3, L.ffn, layer));
    return x;
  }

  /** run the network, returning logits [S,V] (and hidden state) */
  logits(ids: Int32Array | number[], memIds?: Int32Array | number[], qTok: string[] = [], kTok: string[] = []): T {
    let mem: T | null = null;
    if (this.cfg.arch === 'seq2seq' && memIds) {
      let m = this.embed(memIds);
      const eTok = diag.memTok.length ? diag.memTok : qTok;
      for (let l = 0; l < this.encLayers.length; l++) {
        const L = this.encLayers[l];
        const h1 = layerNorm(m, L.ln1.g, L.ln1.b);
        m = add(m, this.attention(h1, h1, L.self, null, l, 'enc', eTok, eTok));
        const h3 = layerNorm(m, L.ln2.g, L.ln2.b);
        m = add(m, this.ffn(h3, L.ffn, -1));
      }
      mem = layerNorm(m, this.lnfEnc.g, this.lnfEnc.b);
    }

    if (this.cfg.arch === 'slot') return this.slotForward(ids, qTok, kTok);

    let x = this.embed(ids);
    if (diag.enabled) diag.embOut = x;
    const S = x.r;
    const mask = causalMask(S);
    for (let l = 0; l < this.layers.length; l++) x = this.block(x, this.layers[l], mask, mem, l, qTok, kTok);
    x = layerNorm(x, this.lnf.g, this.lnf.b);
    if (diag.enabled) diag.hidden = x;
    const out = matmulT(x, this.emb);
    if (diag.enabled) diag.logits = out;
    return out;
  }

  private slotForward(ids: Int32Array | number[], qTok: string[], _kTok: string[]): T {
    const arr = ids instanceof Int32Array ? ids : Int32Array.from(ids);
    const S = arr.length;
    const D = this.cfg.dModel;
    const NS = this.cfg.slots;
    const x = this.embed(arr);
    if (diag.enabled) diag.embOut = x;
    const K = addBias(matmul(x, this.sq.wk), this.sq.bk);
    const V = addBias(matmul(x, this.sq.wv), this.sq.bv);
    const slotLog = diag.enabled ? new Float32Array(NS * S * S) : null;
    let H: T = this.slotInit;
    const rows: T[] = [];
    const inv = 1 / Math.sqrt(D);
    for (let t = 0; t < S; t++) {
      const xt = sliceRows(x, t, t + 1);
      const xtE = NS === 1 ? xt : concatRows(Array.from({ length: NS }, () => xt));
      const Q = addBias(matmul(H, this.sq.wq), this.sq.bq);
      const Kt = sliceRows(K, 0, t + 1);
      const Vt = sliceRows(V, 0, t + 1);
      const a = softmax(scale(matmulT(Q, Kt), inv));
      if (slotLog) {
        for (let s = 0; s < NS; s++) for (let j = 0; j <= t; j++) slotLog[(s * S + t) * S + j] = a.data[s * (t + 1) + j];
      }
      const ctx = matmul(a, Vt);
      const xz = concatCols([ctx, xtE]);
      if (diag.enabled && t === S - 1) {
        diag.ffn = { layer: 0, W1: this.gru.wn, b1: this.gru.bn, W2: this.gru.un, input: xz, hidden: ctx };
      }
      const z = sigmoid(addBias(matmul(xz, this.gru.wz), this.gru.bz));
      const r = sigmoid(addBias(matmul(xz, this.gru.wr), this.gru.br));
      const cand = tanh(addBias(add(matmul(xz, this.gru.wn), matmul(mul(r, H), this.gru.un)), this.gru.bn));
      H = add(cand, mul(z, add(H, scale(cand, -1))));
      const rw = softmax(matmulT(this.readQ, H));
      const read = layerNorm(matmul(rw, H), this.lnSlot.g, this.lnSlot.b);
      rows.push(matmulT(read, this.emb));
    }
    if (diag.enabled && slotLog) {
      for (let s = 0; s < NS; s++) {
        const sub = new Float32Array(S * S);
        for (let t = 0; t < S; t++) for (let j = 0; j < S; j++) sub[t * S + j] = slotLog[(s * S + t) * S + j];
        diag.attn.push({ layer: 0, head: s, tag: `slot${s}`, rows: S, cols: S, data: sub, qTok, kTok: qTok });
      }
      diag.hidden = H;
    }
    const logits = concatRows(rows);
    if (diag.enabled) diag.logits = logits;
    return logits;
  }

  /** teacher-forced loss over a batch of 1 sample */
  lossFor(ids: Int32Array, targets: Int32Array, weights: Float32Array, memIds?: Int32Array): { loss: T; logits: T } {
    const logits = this.logits(ids, memIds);
    return { loss: crossEntropy(logits, targets, weights), logits };
  }

  /** forward + backward on one sample; returns loss value */
  trainSample(ids: Int32Array, targets: Int32Array, weights: Float32Array, memIds?: Int32Array): { loss: number; correct: number; total: number } {
    newGraph();
    const { loss, logits } = this.lossFor(ids, targets, weights, memIds);
    backward(loss);
    let correct = 0,
      total = 0;
    const V = logits.c;
    for (let i = 0; i < logits.r; i++) {
      if (weights[i] === 0) continue;
      total++;
      let best = 0,
        bv = -Infinity;
      for (let j = 0; j < V; j++) {
        const v = logits.data[i * V + j];
        if (v > bv) {
          bv = v;
          best = j;
        }
      }
      if (best === targets[i]) correct++;
    }
    return { loss: loss.data[0], correct, total };
  }

  /** inference-only logits (no tape) */
  infer(ids: number[], memIds?: number[], qTok: string[] = [], kTok: string[] = []): T {
    return noGrad(() =>
      this.logits(Int32Array.from(ids), memIds ? Int32Array.from(memIds) : undefined, qTok, kTok),
    );
  }
}

export function accuracyFromLogits(logits: T, targets: Int32Array, weights: Float32Array) {
  const V = logits.c;
  let c = 0,
    t = 0;
  for (let i = 0; i < logits.r; i++) {
    if (weights[i] === 0) continue;
    t++;
    let best = 0,
      bv = -Infinity;
    for (let j = 0; j < V; j++) {
      const v = logits.data[i * V + j];
      if (v > bv) {
        bv = v;
        best = j;
      }
    }
    if (best === targets[i]) c++;
  }
  return t ? c / t : 0;
}
