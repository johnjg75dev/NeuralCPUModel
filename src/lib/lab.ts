import { Model, diag, resetDiag } from './model';
import { T, argmaxRow, softmaxRowsRaw } from './tensor';
import { EncodedSample, TokScheme, Vocab, splitTokens } from './tokenizer';
import {
  BOS,
  EOS,
  SEP,
  CpuState,
  Instr,
  fmtInstr,
  fmtState,
  makeCpu,
  maskOf,
  runProgram,
  REGS,
} from './asm';

/* ------------------------------- Adam ------------------------------- */

export class Adam {
  m: Float32Array[] = [];
  v: Float32Array[] = [];
  t = 0;
  lr: number;
  b1 = 0.9;
  b2 = 0.98;
  eps = 1e-8;
  wd: number;
  clip: number;
  constructor(params: T[], lr: number, wd = 0, clip = 1) {
    this.lr = lr;
    this.wd = wd;
    this.clip = clip;
    for (const p of params) {
      this.m.push(new Float32Array(p.n));
      this.v.push(new Float32Array(p.n));
    }
  }
  step(params: T[]) {
    this.t++;
    if (this.clip > 0) {
      let gn = 0;
      for (const p of params) for (let i = 0; i < p.grad.length; i++) gn += p.grad[i] * p.grad[i];
      gn = Math.sqrt(gn);
      if (gn > this.clip) {
        const s = this.clip / (gn || 1);
        for (const p of params) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= s;
      }
    }
    const c1 = 1 - Math.pow(this.b1, this.t);
    const c2 = 1 - Math.pow(this.b2, this.t);
    for (let k = 0; k < params.length; k++) {
      const p = params[k],
        m = this.m[k],
        v = this.v[k],
        d = p.data,
        g = p.grad;
      const a = this.lr / c1;
      for (let i = 0; i < d.length; i++) {
        let gi = g[i];
        if (this.wd) gi += this.wd * d[i];
        m[i] = this.b1 * m[i] + (1 - this.b1) * gi;
        v[i] = this.b2 * v[i] + (1 - this.b2) * gi * gi;
        d[i] -= (a * m[i]) / (Math.sqrt(v[i] / c2) + this.eps);
      }
    }
  }
}

/* ------------------------------ Trainer ------------------------------ */

export interface TrainCfg {
  lr: number;
  batch: number;
  clip: number;
  wd: number;
}

export interface StepStat {
  step: number;
  loss: number;
  acc: number;
  ms: number;
  tokens: number;
}

export class Trainer {
  model: Model;
  data: EncodedSample[];
  opt: Adam;
  cfg: TrainCfg;
  history: StepStat[] = [];
  stepCount = 0;
  cursor = 0;
  seenTokens = 0;
  emaLoss = 0;
  emaAcc = 0;
  lastMs = 0;
  bestLoss = Infinity;

  constructor(model: Model, data: EncodedSample[], cfg: TrainCfg) {
    this.model = model;
    this.data = data;
    this.cfg = cfg;
    this.opt = new Adam(model.params, cfg.lr, cfg.wd, cfg.clip);
  }

  setCfg(cfg: TrainCfg) {
    const rebuild = cfg.lr !== this.cfg.lr || cfg.wd !== this.cfg.wd || cfg.clip !== this.cfg.clip;
    this.cfg = cfg;
    if (rebuild) this.opt = new Adam(this.model.params, cfg.lr, cfg.wd, cfg.clip);
  }

  private sample(): EncodedSample {
    const s = this.data[this.cursor % this.data.length];
    this.cursor++;
    return s;
  }

  doStep(): StepStat {
    const t0 = performance.now();
    const { model } = this;
    let loss = 0,
      correct = 0,
      total = 0,
      toks = 0;
    for (const p of model.params) p.grad.fill(0);
    for (let b = 0; b < this.cfg.batch; b++) {
      const s = this.sample();
      diag.memTok = s.tokens;
      const r =
        model.cfg.arch === 'seq2seq'
          ? model.trainSample(s.decIds, s.decTargets, s.decWeights, s.encIds)
          : model.trainSample(s.ids, s.targets, s.weights);
      loss += r.loss;
      correct += r.correct;
      total += r.total;
      toks += r.total;
    }
    this.opt.step(model.params);
    const ms = performance.now() - t0;
    loss /= this.cfg.batch;
    const acc = total ? correct / total : 0;
    this.stepCount++;
    this.seenTokens += toks;
    this.lastMs = ms;
    const k = Math.min(1, 2 / (this.stepCount + 4));
    this.emaLoss = this.emaLoss ? this.emaLoss * (1 - k) + loss * k : loss;
    this.emaAcc = this.emaAcc ? this.emaAcc * (1 - k) + acc * k : acc;
    if (loss < this.bestLoss) this.bestLoss = loss;
    const st: StepStat = { step: this.stepCount, loss, acc, ms, tokens: toks };
    this.history.push(st);
    if (this.history.length > 4000) this.history.splice(0, 1000);
    return st;
  }

  evaluate(n = 24): { loss: number; acc: number } {
    let loss = 0,
      c = 0,
      t = 0;
    const m = Math.min(n, this.data.length);
    for (let i = 0; i < m; i++) {
      const s = this.data[(i * 7 + 3) % this.data.length];
      const logits = this.model.infer(
        Array.from(this.model.cfg.arch === 'seq2seq' ? s.decIds : s.ids),
        this.model.cfg.arch === 'seq2seq' ? Array.from(s.encIds) : undefined,
      );
      const tg = this.model.cfg.arch === 'seq2seq' ? s.decTargets : s.targets;
      const w = this.model.cfg.arch === 'seq2seq' ? s.decWeights : s.weights;
      const V = logits.c;
      for (let r = 0; r < logits.r; r++) {
        if (w[r] === 0) continue;
        t++;
        let mx = -Infinity,
          sum = 0;
        for (let j = 0; j < V; j++) {
          const e = Math.exp(logits.data[r * V + j] - 8);
          sum += e;
          if (j === tg[r]) mx = e;
        }
        loss += -Math.log(Math.max(1e-9, mx / sum)) * w[r];
        if (argmaxRow(logits.data, r, V) === tg[r]) c++;
      }
    }
    return { loss: t ? loss / t : 0, acc: t ? c / t : 0 };
  }
}

/* ------------------------------ inference ------------------------------ */

export function decodeGreedy(
  model: Model,
  prompt: number[],
  maxNew: number,
  stopIds: number[],
  memIds?: number[],
  temperature = 0,
  rng: () => number = Math.random,
): number[] {
  const ids = prompt.slice();
  const out: number[] = [];
  for (let i = 0; i < maxNew; i++) {
    if (ids.length >= model.cfg.maxLen) break;
    const logits = model.infer(ids, memIds);
    const last = logits.r - 1;
    let next: number;
    if (temperature <= 0) {
      next = argmaxRow(logits.data, last, logits.c);
    } else {
      const row = new Float32Array(logits.c);
      let mx = -Infinity;
      for (let j = 0; j < logits.c; j++) mx = Math.max(mx, logits.data[last * logits.c + j]);
      let s = 0;
      for (let j = 0; j < logits.c; j++) {
        row[j] = Math.exp((logits.data[last * logits.c + j] - mx) / temperature);
        s += row[j];
      }
      let r = rng() * s,
        pick = 0;
      for (let j = 0; j < row.length; j++) {
        r -= row[j];
        if (r <= 0) {
          pick = j;
          break;
        }
      }
      next = pick;
    }
    out.push(next);
    ids.push(next);
    if (stopIds.includes(next)) break;
  }
  return out;
}

/* ------------------- program execution + validation ------------------- */

export interface StepCheck {
  index: number;
  instr: Instr;
  truth: number[];
  truthText: string;
  predText: string;
  predTokens: string[];
  truthTokens: string[];
  tokenOk: boolean[];
  exact: boolean;
  note: string;
  regsBefore: number[];
}

export interface RunResult {
  states: number[][];
  notes: string[];
  checks: StepCheck[];
  exactRate: number;
  tokenRate: number;
}

export function activeRegs(n: number) {
  return Array.from({ length: n }, (_, i) => i);
}

/** run the listing on the emulated core, then interrogate the network at every step */
export function validateProgram(opts: {
  instrs: Instr[];
  cpu: CpuState;
  model: Model;
  vocab: Vocab;
  scheme: TokScheme;
  bits: number;
  nRegs: number;
}): RunResult {
  const { instrs, cpu, model, vocab, scheme, bits, nRegs } = opts;
  const { states, notes } = runProgram(cpu, instrs);
  const act = activeRegs(nRegs);
  const checks: StepCheck[] = [];
  let exactHits = 0,
    tokHits = 0,
    tokTotal = 0;

  for (let i = 0; i < instrs.length; i++) {
    const before = states[i];
    const truth = states[i + 1];
    const truthText = fmtState(truth, bits, act);
    const instrText = fmtInstr(instrs[i], bits);
    const sepId = vocab.id(SEP);
    const eosId = vocab.id(EOS);

    let predText = '';
    let predTokens: string[] = [];
    if (model.cfg.arch === 'seq2seq') {
      const encText = BOS + SEP + instrText + SEP + EOS;
      const decPrompt = splitTokens(BOS + SEP + fmtState(before, bits, act) + SEP, scheme).map((t) => vocab.id(t));
      const encIds = splitTokens(encText, scheme).map((t) => vocab.id(t));
      const gen = decodeGreedy(model, decPrompt, 14, [sepId, eosId], encIds);
      predTokens = gen.filter((g) => g !== sepId && g !== eosId).map((g) => vocab.tok(g));
    } else {
      const promptText = BOS + fmtState(before, bits, act) + SEP + instrText + SEP;
      const prompt = splitTokens(promptText, scheme).map((t) => vocab.id(t));
      const gen = decodeGreedy(model, prompt, 14, [sepId, eosId]);
      predTokens = gen.filter((g) => g !== sepId && g !== eosId).map((g) => vocab.tok(g));
    }
    predText = predTokens.join(scheme === 'char' ? '' : ' ');
    const truthTokens = splitTokens(truthText, scheme);
    const tokenOk = truthTokens.map((t, k) => predTokens[k] === t);
    const norm = (s: string) => s.replace(/\s+/g, '').toUpperCase();
    const exact = norm(predText) === norm(truthText);
    if (exact) exactHits++;
    tokHits += tokenOk.filter(Boolean).length;
    tokTotal += truthTokens.length;
    checks.push({
      index: i,
      instr: instrs[i],
      truth,
      truthText,
      predText,
      predTokens,
      truthTokens,
      tokenOk,
      exact,
      note: notes[i],
      regsBefore: before,
    });
  }
  return {
    states,
    notes,
    checks,
    exactRate: instrs.length ? exactHits / instrs.length : 0,
    tokenRate: tokTotal ? tokHits / tokTotal : 0,
  };
}

/** free-run the network from a seed state and let it invent a trace */
export function neuralFreeRun(
  model: Model,
  vocab: Vocab,
  scheme: TokScheme,
  regs: number[],
  bits: number,
  nRegs: number,
  maxTokens = 64,
  temperature = 0.7,
  encText?: string,
): { text: string; tokens: string[]; ids: number[] } {
  const act = activeRegs(nRegs);
  const promptText =
    model.cfg.arch === 'seq2seq' ? BOS + SEP + fmtState(regs, bits, act) + SEP : BOS + fmtState(regs, bits, act) + SEP;
  const prompt = splitTokens(promptText, scheme).map((t) => vocab.id(t));
  const memIds =
    model.cfg.arch === 'seq2seq' && encText ? splitTokens(encText, scheme).map((t) => vocab.id(t)) : undefined;
  const gen = decodeGreedy(model, prompt, maxTokens, [vocab.id(EOS)], memIds, temperature);
  const ids = [...prompt, ...gen];
  const tokens = ids.map((i) => vocab.tok(i));
  return { text: tokens.join(scheme === 'char' ? '' : ' '), tokens, ids };
}

/** attention + embedding capture for a concrete sequence */
export function captureFor(model: Model, ids: number[], qTok: string[], kTok: string[], memIds?: number[], memTok?: string[]) {
  resetDiag();
  diag.enabled = true;
  if (memTok) diag.memTok = memTok;
  const logits = model.infer(ids, memIds, qTok, kTok);
  const probs = softmaxRowsRaw(logits.data, logits.r, logits.c);
  diag.enabled = false;
  return { attn: diag.attn.slice(), ffn: diag.ffn, emb: diag.embOut, hidden: diag.hidden, logits, probs };
}

export function initialCpu(bits: number, seed?: number[]): CpuState {
  const c = makeCpu(bits);
  if (seed) for (let i = 0; i < Math.min(seed.length, REGS.length); i++) c.regs[i] = seed[i] & maskOf(bits);
  return c;
}
