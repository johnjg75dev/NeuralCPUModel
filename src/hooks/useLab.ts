import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  DEFAULT_CORPUS,
  BOS,
  EOS,
  SEP,
  OpName,
  parseProgram,
  REGS,
  Sample,
  genCorpus,
  CorpusCfg,
  maskOf,
  fmtInstr,
  fmtState,
  runProgram,
} from '@/lib/asm';
import { DEFAULT_CFG, Model, ModelCfg, AttnRec, FfnSnap, diag } from '@/lib/model';
import { EncodedSample, TokScheme, Vocab, encodeSample, splitTokens } from '@/lib/tokenizer';
import { RunResult, StepStat, Trainer, captureFor, initialCpu, validateProgram, neuralFreeRun } from '@/lib/lab';

export interface LabConfig extends ModelCfg, CorpusCfg {
  scheme: TokScheme;
  stateWeight: number;
  nSamples: number;
  lr: number;
  batch: number;
  clip: number;
  wd: number;
}

export const DEFAULT_LAB: LabConfig = {
  ...DEFAULT_CFG,
  ...DEFAULT_CORPUS,
  scheme: 'word',
  stateWeight: 4,
  nSamples: 256,
  lr: 0.008,
  batch: 4,
  clip: 1,
  wd: 0,
};

export interface Snap {
  attn: AttnRec[];
  ffn: FfnSnap | null;
  ctxEmb: Float32Array | null;
  probs: Float32Array | null;
  emb: Float32Array | null;
  D: number;
  V: number;
  S: number;
  tokens: string[];
  ids: number[];
  vocabList: string[];
  version: number;
}

export interface ThetaRow {
  name: string;
  gn: number;
  wn: number;
  n: number;
}

export interface Stats {
  step: number;
  loss: number;
  emaLoss: number;
  acc: number;
  emaAcc: number;
  valLoss: number;
  valAcc: number;
  ms: number;
  tps: number;
  params: number;
  vocab: number;
  avgLen: number;
  training: boolean;
  snapVersion: number;
  seenTokens: number;
  bestLoss: number;
  elapsed: number;
}

const EMPTY_SNAP: Snap = {
  attn: [],
  ffn: null,
  ctxEmb: null,
  probs: null,
  emb: null,
  D: 0,
  V: 0,
  S: 0,
  tokens: [],
  ids: [],
  vocabList: [],
  version: 0,
};

/* ------------------------- serialisation helpers ------------------------- */

export function inspectionText(program: string, c: LabConfig, seedRegs: number[]): string {
  const { instrs } = parseProgram(program);
  const cpu = initialCpu(c.bits, seedRegs);
  const act = Array.from({ length: c.nRegs }, (_, i) => i);
  const list = instrs.slice(0, 5);
  const { states } = runProgram(cpu, list);
  const parts = [BOS + fmtState(states[0], c.bits, act)];
  for (let i = 0; i < list.length; i++) {
    parts.push(SEP + fmtInstr(list[i], c.bits));
    parts.push(SEP + fmtState(states[i + 1], c.bits, act));
  }
  parts.push(SEP + EOS);
  return parts.join('');
}

export function inspectionParts(program: string, c: LabConfig, seedRegs: number[]) {
  const { instrs } = parseProgram(program);
  const cpu = initialCpu(c.bits, seedRegs);
  const act = Array.from({ length: c.nRegs }, (_, i) => i);
  const list = instrs.slice(0, 5);
  const { states } = runProgram(cpu, list);
  const enc = BOS + list.map((s) => SEP + fmtInstr(s, c.bits)).join('') + SEP + EOS;
  const dec = BOS + states.map((s) => SEP + fmtState(s, c.bits, act)).join('') + SEP + EOS;
  return { enc, dec };
}

/* --------------------------------- hook --------------------------------- */

export function useLab() {
  const [cfg, setCfg] = useState<LabConfig>(DEFAULT_LAB);
  const cfgRef = useRef(cfg);
  cfgRef.current = cfg;

  const modelRef = useRef<Model | null>(null);
  const trainerRef = useRef<Trainer | null>(null);
  const dataRef = useRef<EncodedSample[]>([]);
  const samplesRef = useRef<Sample[]>([]);
  const vocabRef = useRef<Vocab | null>(null);
  const snapRef = useRef<Snap>(EMPTY_SNAP);
  const thetaRef = useRef<ThetaRow[]>([]);
  const histRef = useRef<StepStat[]>([]);

  const [building, setBuilding] = useState(true);
  const [training, setTraining] = useState(false);
  const trainingRef = useRef(false);
  const [meta, setMeta] = useState({ params: 0, vocab: 0, avgLen: 0 });
  const metaRef = useRef(meta);
  metaRef.current = meta;

  const [program, setProgram] = useState(`; NEU-86 listing — edit me
MOV AX,3
MOV BX,5
ADD AX,BX
XOR CX,CX
ADD CX,AX
XOR AX,AX
SUB BX,2`);
  const programRef = useRef(program);
  programRef.current = program;

  const [seedRegs, setSeedRegs] = useState<number[]>([0, 0, 0, 0]);
  const seedRef = useRef(seedRegs);
  seedRef.current = seedRegs;

  const [runResult, setRunResult] = useState<RunResult | null>(null);
  const [focusStep, setFocusStep] = useState(0);
  const [freeText, setFreeText] = useState('');
  const [ffnLayer, setFfnLayerState] = useState(0);
  const [stats, setStats] = useState<Stats>({
    step: 0,
    loss: 0,
    emaLoss: 0,
    acc: 0,
    emaAcc: 0,
    valLoss: 0,
    valAcc: 0,
    ms: 0,
    tps: 0,
    params: 0,
    vocab: 0,
    avgLen: 0,
    training: false,
    snapVersion: 0,
    seenTokens: 0,
    bestLoss: Infinity,
    elapsed: 0,
  });

  const parsed = useMemo(() => parseProgram(program), [program]);

  /* ------------------------- diagnostic snapshot ------------------------- */
  const refreshSnapshot = useCallback((force = false) => {
    const model = modelRef.current;
    const vocab = vocabRef.current;
    const c = cfgRef.current;
    if (!model || !vocab) return;
    if (!force && snapRef.current.version === 0) return;
    let text = inspectionText(programRef.current, c, seedRef.current);
    let encText: string | undefined;
    let memTok: string[] | undefined;
    if (c.arch === 'seq2seq') {
      const s = inspectionParts(programRef.current, c, seedRef.current);
      encText = s.enc;
      text = s.dec;
      memTok = splitTokens(encText, c.scheme);
    }
    const tokens = splitTokens(text, c.scheme);
    let ids = tokens.map((t) => vocab.id(t));
    if (ids.length > c.maxLen) {
      ids = ids.slice(0, c.maxLen);
      tokens.length = c.maxLen;
    }
    if (ids.length < 2) return;
    const memIds = encText ? splitTokens(encText, c.scheme).map((t) => vocab.id(t)) : undefined;
    diag.ffnLayer = c.arch === 'slot' ? 0 : ffnLayerRef.current;
    const cap = captureFor(model, ids, tokens, memTok ?? tokens, memIds, memTok);
    snapRef.current = {
      attn: cap.attn,
      ffn: cap.ffn,
      ctxEmb: cap.emb ? (cap.emb.data as Float32Array) : null,
      probs: cap.probs,
      emb: model.emb.data as Float32Array,
      D: model.cfg.dModel,
      V: vocab.size,
      S: ids.length,
      tokens,
      ids,
      vocabList: vocab.i2t,
      version: snapRef.current.version + 1,
    };
  }, []);

  const ffnLayerRef = useRef(0);
  const setFfnLayer = useCallback((n: number) => {
    ffnLayerRef.current = n;
    diag.ffnLayer = n;
    setFfnLayerState(n);
    refreshSnapshot(true);
  }, [refreshSnapshot]);

  /* ------------------------------ build ------------------------------ */
  const build = useCallback((c: LabConfig) => {
    const samples = genCorpus(c as CorpusCfg, c.nSamples);
    const vocab = Vocab.build(
      c.arch === 'seq2seq' ? samples.flatMap((s) => [s.encText, s.decText]) : samples.map((s) => s.text),
      c.scheme,
    );
    const data = samples.map((s) => encodeSample(s.text, s.encText, s.decText, vocab, c.scheme, c.stateWeight));
    const mcfg: ModelCfg = {
      arch: c.arch,
      dModel: c.dModel,
      nHeads: c.nHeads,
      nLayers: c.nLayers,
      dff: c.dff,
      maxLen: c.maxLen,
      slots: c.slots,
      activation: c.activation,
      posMode: c.posMode,
    };
    const model = new Model(mcfg, vocab.size);
    const trainer = new Trainer(model, data, { lr: c.lr, batch: c.batch, clip: c.clip, wd: c.wd });
    modelRef.current = model;
    trainerRef.current = trainer;
    dataRef.current = data;
    samplesRef.current = samples;
    vocabRef.current = vocab;
    histRef.current = trainer.history;
    const avgLen = data.reduce((a, s) => a + s.ids.length, 0) / Math.max(1, data.length);
    thetaRef.current = model.params.map((p, i) => ({ name: p.name ?? `θ[${i}]`, gn: 0, wn: 0, n: p.n }));
    setMeta({ params: model.countParams(), vocab: vocab.size, avgLen });
    setRunResult(null);
    setFreeText('');
    snapRef.current = { ...EMPTY_SNAP, D: c.dModel, V: vocab.size, vocabList: vocab.i2t };
    setStats({
      step: 0,
      loss: 0,
      emaLoss: 0,
      acc: 0,
      emaAcc: 0,
      valLoss: 0,
      valAcc: 0,
      ms: 0,
      tps: 0,
      params: model.countParams(),
      vocab: vocab.size,
      avgLen,
      training: trainingRef.current,
      snapVersion: 0,
      seenTokens: 0,
      bestLoss: Infinity,
      elapsed: 0,
    });
    setBuilding(false);
    refreshSnapshot(true);
  }, [refreshSnapshot]);

  useEffect(() => {
    build(cfgRef.current);
  }, [build]);

  /* ------------- rebuild on structural change ------------- */
  const structural = [
    cfg.arch,
    cfg.dModel,
    cfg.nHeads,
    cfg.nLayers,
    cfg.dff,
    cfg.maxLen,
    cfg.slots,
    cfg.activation,
    cfg.posMode,
    cfg.scheme,
    cfg.bits,
    cfg.nRegs,
    cfg.ops.join(','),
    cfg.progMin,
    cfg.progMax,
    cfg.immBias,
    cfg.nSamples,
    cfg.seed,
    cfg.stateWeight,
  ].join('|');
  const lastStructural = useRef(structural);
  const firstRun = useRef(true);
  useEffect(() => {
    if (firstRun.current) {
      firstRun.current = false;
      lastStructural.current = structural;
      return;
    }
    if (lastStructural.current !== structural) {
      lastStructural.current = structural;
      setBuilding(true);
      const id = setTimeout(() => build(cfgRef.current), 40);
      return () => clearTimeout(id);
    }
  }, [structural, build]);

  useEffect(() => {
    trainerRef.current?.setCfg({ lr: cfg.lr, batch: cfg.batch, clip: cfg.clip, wd: cfg.wd });
  }, [cfg.lr, cfg.batch, cfg.clip, cfg.wd]);

  /* ------------------------------ publish ------------------------------ */
  const startedAt = useRef(0);
  const lastPub = useRef(0);
  const lastSnap = useRef(0);

  const lastEval = useRef({ loss: 0, acc: 0 });
  const publish = useCallback(() => {
    const tr = trainerRef.current;
    const model = modelRef.current;
    if (!tr || !model) return;
    const v = tr.stepCount % 2 === 0 ? tr.evaluate(6) : lastEval.current;
    thetaRef.current = model.params
      .map((p, i) => {
        let g = 0,
          w = 0;
        for (let k = 0; k < p.data.length; k++) {
          g += p.grad[k] * p.grad[k];
          w += p.data[k] * p.data[k];
        }
        return { name: p.name ?? `θ${i}`, gn: Math.sqrt(g), wn: Math.sqrt(w / Math.max(1, p.n)), n: p.n };
      })
      .filter((r) => r.n > 1)
      .sort((a, b) => b.gn - a.gn)
      .slice(0, 14);
    lastEval.current = v;
    const el = startedAt.current ? (performance.now() - startedAt.current) / 1000 : 0;
    setStats({
      step: tr.stepCount,
      loss: tr.history.length ? tr.history[tr.history.length - 1].loss : 0,
      emaLoss: tr.emaLoss,
      acc: tr.history.length ? tr.history[tr.history.length - 1].acc : 0,
      emaAcc: tr.emaAcc,
      valLoss: v.loss,
      valAcc: v.acc,
      ms: tr.lastMs,
      tps: el > 0.05 ? tr.seenTokens / el : 0,
      params: metaRef.current.params,
      vocab: metaRef.current.vocab,
      avgLen: metaRef.current.avgLen,
      training: trainingRef.current,
      snapVersion: snapRef.current.version,
      seenTokens: tr.seenTokens,
      bestLoss: tr.bestLoss,
      elapsed: el,
    });
  }, []);

  /* ------------------------------ loop ------------------------------ */
  const rafRef = useRef(0);
  const loop = useCallback(() => {
    const tr = trainerRef.current;
    if (!tr || !trainingRef.current) return;
    const frameStart = performance.now();
    let guard = 0;
    while (performance.now() - frameStart < 13 && guard < 200) {
      tr.doStep();
      guard++;
    }
    const now = performance.now();
    if (now - lastSnap.current > 480) {
      lastSnap.current = now;
      try {
        refreshSnapshot();
      } catch {
        /* keep the loop alive */
      }
    }
    if (now - lastPub.current > 150) {
      lastPub.current = now;
      publish();
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [publish, refreshSnapshot]);

  const setRunning = useCallback(
    (v: boolean) => {
      trainingRef.current = v;
      setTraining(v);
      if (v) {
        if (!startedAt.current) startedAt.current = performance.now();
        cancelAnimationFrame(rafRef.current);
        rafRef.current = requestAnimationFrame(loop);
      } else {
        cancelAnimationFrame(rafRef.current);
        publish();
        refreshSnapshot(true);
      }
    },
    [loop, publish, refreshSnapshot],
  );

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  /* keep the diagnostic snapshot locked to the listing being edited */
  useEffect(() => {
    const id = setTimeout(() => {
      try {
        refreshSnapshot(true);
        setStats((s) => ({ ...s, snapVersion: snapRef.current.version }));
      } catch {
        /* listing mid-edit */
      }
    }, 220);
    return () => clearTimeout(id);
  }, [program, seedRegs, refreshSnapshot]);

  /* the lab boots straight into descent — halt any time */
  const autoStarted = useRef(false);
  useEffect(() => {
    if (autoStarted.current) return;
    autoStarted.current = true;
    const id = setTimeout(() => setRunning(true), 1200);
    return () => clearTimeout(id);
  }, [setRunning]);

  const stepOnce = useCallback(() => {
    if (!startedAt.current) startedAt.current = performance.now();
    trainerRef.current?.doStep();
    refreshSnapshot(true);
    publish();
  }, [publish, refreshSnapshot]);

  const resetWeights = useCallback(() => {
    setRunning(false);
    startedAt.current = 0;
    build(cfgRef.current);
  }, [build, setRunning]);

  /* ------------------------------ execution ------------------------------ */
  const doRun = useCallback(() => {
    const model = modelRef.current;
    const vocab = vocabRef.current;
    if (!model || !vocab) return;
    const c = cfgRef.current;
    const { instrs } = parseProgram(programRef.current);
    const cpu = initialCpu(c.bits, seedRef.current);
    const res = validateProgram({ instrs, cpu, model, vocab, scheme: c.scheme, bits: c.bits, nRegs: c.nRegs });
    setRunResult(res);
    setFocusStep(Math.max(0, res.checks.length - 1));
    const free = neuralFreeRun(
      model,
      vocab,
      c.scheme,
      seedRef.current.map((v) => v & maskOf(c.bits)),
      c.bits,
      c.nRegs,
      48,
      0.75,
      inspectionParts(programRef.current, c, seedRef.current).enc,
    );
    setFreeText(free.text);
    refreshSnapshot(true);
    publish();
  }, [publish, refreshSnapshot]);

  return {
    cfg,
    setCfg,
    patch: (p: Partial<LabConfig>) => setCfg((c) => ({ ...c, ...p })),
    building,
    training,
    setRunning,
    stepOnce,
    resetWeights,
    stats,
    histRef,
    snapRef,
    thetaRef,
    modelRef,
    vocabRef,
    dataRef,
    samplesRef,
    meta,
    program,
    setProgram,
    parsed,
    seedRegs,
    setSeedRegs,
    runResult,
    doRun,
    focusStep,
    setFocusStep,
    freeText,
    refreshSnapshot,
    ffnLayer,
    setFfnLayer,
    registerNames: REGS,
  };
}

export type { OpName, Sample, Trainer };
