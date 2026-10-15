/* ------------------------------------------------------------------
   NEU-86 : a tiny x86 register-machine emulator + ground-truth
   corpus generator for the transformer lab.
------------------------------------------------------------------ */

export const REGS = ['AX', 'BX', 'CX', 'DX'] as const;
export type RegName = (typeof REGS)[number];
export const OPS = ['MOV', 'ADD', 'XOR', 'SUB'] as const;
export type OpName = (typeof OPS)[number];

export interface Operand {
  kind: 'reg' | 'imm';
  reg?: RegName;
  val: number;
}

export interface Instr {
  op: OpName;
  dst: RegName;
  src: Operand;
  text: string;
  line: number;
}

export interface Issue {
  line: number;
  msg: string;
  level: 'error' | 'warn';
}

const REG_ALIAS: Record<string, RegName> = {
  EAX: 'AX',
  EBX: 'BX',
  ECX: 'CX',
  EDX: 'DX',
  RAX: 'AX',
  RBX: 'BX',
  RCX: 'CX',
  RDX: 'DX',
  AL: 'AX',
  BL: 'BX',
  CL: 'CX',
  DL: 'DX',
  AH: 'AX',
  AX: 'AX',
  BX: 'BX',
  CX: 'CX',
  DX: 'DX',
};

function parseOperand(tok: string): Operand | null {
  const up = tok.trim().toUpperCase();
  if (!up) return null;
  if (REG_ALIAS[up]) return { kind: 'reg', reg: REG_ALIAS[up], val: 0 };
  let v: number;
  if (/^0X[0-9A-F]+$/i.test(up)) v = parseInt(up.slice(2), 16);
  else if (/^[0-9A-F]+H$/i.test(up)) v = parseInt(up.slice(0, -1), 16);
  else if (/^-?\d+$/.test(up)) v = parseInt(up, 10);
  else if (/^[0-9A-F]+$/i.test(up) && /[A-F]/i.test(up)) v = parseInt(up, 16);
  else return null;
  return { kind: 'imm', val: v };
}

/** parse an assembly listing into instructions + diagnostics */
export function parseProgram(text: string): { instrs: Instr[]; issues: Issue[] } {
  const instrs: Instr[] = [];
  const issues: Issue[] = [];
  const lines = text.split('\n');
  lines.forEach((raw, li) => {
    const line = raw.replace(/;.*$/, '').trim();
    if (!line) return;
    const m = line.match(/^([A-Za-z]{2,4})\s*(.*)$/);
    if (!m) {
      issues.push({ line: li + 1, msg: `unrecognised statement "${line}"`, level: 'error' });
      return;
    }
    const op = m[1].toUpperCase() as OpName;
    if (!(OPS as readonly string[]).includes(op)) {
      issues.push({ line: li + 1, msg: `opcode ${op} not in ISA subset {${OPS.join(', ')}}`, level: 'error' });
      return;
    }
    const parts = m[2].split(',').map((s) => s.trim()).filter(Boolean);
    if (parts.length !== 2) {
      issues.push({ line: li + 1, msg: `${op} needs two operands, got ${parts.length}`, level: 'error' });
      return;
    }
    const dst = REG_ALIAS[parts[0].toUpperCase()];
    if (!dst) {
      issues.push({ line: li + 1, msg: `unknown destination register "${parts[0]}"`, level: 'error' });
      return;
    }
    const src = parseOperand(parts[1]);
    if (!src) {
      issues.push({ line: li + 1, msg: `bad source operand "${parts[1]}"`, level: 'error' });
      return;
    }
    if (op === 'MOV' && src.kind === 'reg' && src.reg === dst)
      issues.push({ line: li + 1, msg: `MOV ${dst},${dst} is a no-op`, level: 'warn' });
    instrs.push({ op, dst, src, text: `${op} ${dst},${src.kind === 'reg' ? src.reg : fmtImm(src.val)}`, line: li + 1 });
  });
  return { instrs, issues };
}

export function fmtImm(v: number): string {
  return v < 0 ? `-${Math.abs(v)}` : String(v);
}

export interface CpuState {
  regs: number[]; // indexed by REGS order
  bits: number;
  zf: boolean;
  sf: boolean;
  cf: boolean;
  of: boolean;
}

export function makeCpu(bits: number): CpuState {
  return { regs: REGS.map(() => 0), bits, zf: true, sf: false, cf: false, of: false };
}

export function maskOf(bits: number) {
  return (1 << bits) - 1;
}

/** execute one instruction, returning a fresh state + human note */
export function execInstr(cpu: CpuState, ins: Instr): { cpu: CpuState; note: string } {
  const mask = maskOf(cpu.bits);
  const regs = cpu.regs.slice();
  const di = REGS.indexOf(ins.dst);
  const sv = ins.src.kind === 'reg' ? regs[REGS.indexOf(ins.src.reg!)] : ins.src.val & mask;
  const old = regs[di];
  let note = '';
  switch (ins.op) {
    case 'MOV':
      regs[di] = sv & mask;
      note = `${ins.dst} ← ${sv & mask}`;
      break;
    case 'ADD': {
      const full = old + sv;
      regs[di] = full & mask;
      cpu.cf = full > mask;
      note = `${old} + ${sv} = ${full}${full > mask ? ' (carry)' : ''}`;
      break;
    }
    case 'SUB': {
      const full = old - sv;
      regs[di] = full & mask;
      cpu.cf = full < 0;
      note = `${old} − ${sv} = ${full}${full < 0 ? ' (borrow)' : ''}`;
      break;
    }
    case 'XOR':
      regs[di] = (old ^ sv) & mask;
      note = `${old} ⊕ ${sv} = ${regs[di]}`;
      break;
  }
  cpu.zf = regs[di] === 0;
  cpu.sf = (regs[di] & (1 << (cpu.bits - 1))) !== 0;
  return { cpu: { ...cpu, regs }, note };
}

export function runProgram(cpu: CpuState, instrs: Instr[]): { states: number[][]; notes: string[] } {
  const states: number[][] = [cpu.regs.slice()];
  const notes: string[] = [];
  let cur = cpu;
  for (const ins of instrs) {
    const r = execInstr(cur, ins);
    cur = r.cpu;
    states.push(cur.regs.slice());
    notes.push(r.note);
  }
  return { states, notes };
}

/* ---------------------------- serialisation ---------------------------- */

export const BOS = '@';
export const EOS = '#';
export const SEP = '|';

export function hexWidth(bits: number) {
  return Math.max(1, bits / 4);
}

export function fmtVal(v: number, bits: number): string {
  return v.toString(16).toUpperCase().padStart(hexWidth(bits), '0');
}

export function fmtState(regs: number[], bits: number, active: number[]): string {
  return active.map((i) => fmtVal(regs[i], bits)).join('.');
}

export function fmtInstr(ins: Instr, bits: number): string {
  return `${ins.op} ${ins.dst},${ins.src.kind === 'reg' ? ins.src.reg : fmtVal(ins.src.val & maskOf(bits), bits)}`;
}

export interface Sample {
  text: string;
  instrs: Instr[];
  states: number[][];
  encText: string; // instruction stream (seq2seq encoder)
  decText: string; // state stream (seq2seq decoder)
}

export interface CorpusCfg {
  bits: number;
  nRegs: number; // 2..4 registers observed in the state vector
  ops: OpName[];
  progMin: number;
  progMax: number;
  immBias: number; // 0..1 — chance the source operand is an immediate
  seed: number;
}

export const DEFAULT_CORPUS: CorpusCfg = {
  bits: 4,
  nRegs: 3,
  ops: ['MOV', 'ADD', 'XOR'],
  progMin: 3,
  progMax: 4,
  immBias: 0.45,
  seed: 1337,
};

export function mulberry32(a: number) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(rnd: () => number, arr: readonly T[]): T {
  return arr[Math.floor(rnd() * arr.length)];
}

export function genSample(cfg: CorpusCfg, rnd: () => number): Sample {
  const active = Array.from({ length: cfg.nRegs }, (_, i) => i);
  const mask = maskOf(cfg.bits);
  const maxImm = Math.min(mask, cfg.bits >= 8 ? 15 : mask);
  const n = cfg.progMin + Math.floor(rnd() * (cfg.progMax - cfg.progMin + 1));
  const instrs: Instr[] = [];
  for (let i = 0; i < n; i++) {
    const op = pick(rnd, cfg.ops);
    const dst = pick(rnd, active.map((k) => REGS[k]));
    let src: Operand;
    if (rnd() < cfg.immBias) src = { kind: 'imm', val: Math.floor(rnd() * (maxImm + 1)) };
    else src = { kind: 'reg', reg: pick(rnd, active.map((k) => REGS[k])), val: 0 };
    instrs.push({ op, dst, src, text: '', line: i });
  }
  const cpu = makeCpu(cfg.bits);
  // random non-zero start state on ~half the samples
  if (rnd() < 0.5) for (const k of active) cpu.regs[k] = Math.floor(rnd() * (maxImm + 1));
  const { states } = runProgram(cpu, instrs);

  const parts: string[] = [];
  parts.push(BOS + fmtState(states[0], cfg.bits, active));
  for (let i = 0; i < instrs.length; i++) {
    parts.push(SEP + fmtInstr(instrs[i], cfg.bits));
    parts.push(SEP + fmtState(states[i + 1], cfg.bits, active));
  }
  parts.push(SEP + EOS);
  const text = parts.join('');

  const encText = BOS + instrs.map((s) => SEP + fmtInstr(s, cfg.bits)).join('') + SEP + EOS;
  const decText = BOS + states.map((s) => SEP + fmtState(s, cfg.bits, active)).join('') + SEP + EOS;

  return { text, instrs, states, encText, decText };
}

export function genCorpus(cfg: CorpusCfg, n: number): Sample[] {
  const rnd = mulberry32(cfg.seed);
  return Array.from({ length: n }, () => genSample(cfg, rnd));
}

export const DEFAULT_PROGRAM = `; NEU-86 demo listing
MOV AX,3
MOV BX,5
ADD AX,BX
XOR CX,CX
ADD CX,AX
XOR AX,AX
SUB BX,2
MOV CX,AX`;
