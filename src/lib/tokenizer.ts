import { BOS, EOS } from './asm';

export type TokScheme = 'char' | 'word';

const CHAR_RE = /\S/g;
const WORD_RE = /[A-Za-z]{2,}|[0-9A-Fa-f]+|\S/g;

export function splitTokens(text: string, scheme: TokScheme): string[] {
  const re = scheme === 'char' ? CHAR_RE : WORD_RE;
  return text.match(re) ?? [];
}

export class Vocab {
  i2t: string[] = [];
  t2i = new Map<string, number>();
  unk = 0;
  pad = 1;
  bos = 2;
  eos = 3;

  static build(texts: string[], scheme: TokScheme): Vocab {
    const freq = new Map<string, number>();
    for (const t of texts) for (const tok of splitTokens(t, scheme)) freq.set(tok, (freq.get(tok) ?? 0) + 1);
    const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    const v = new Vocab();
    const specials = ['<unk>', '<pad>', BOS, EOS];
    for (const s of specials) v.push(s);
    for (const [tok] of sorted) if (!v.t2i.has(tok)) v.push(tok);
    return v;
  }

  private push(tok: string) {
    this.t2i.set(tok, this.i2t.length);
    this.i2t.push(tok);
  }

  get size() {
    return this.i2t.length;
  }

  encode(text: string, scheme: TokScheme): number[] {
    return splitTokens(text, scheme).map((t) => this.t2i.get(t) ?? this.unk);
  }

  decode(ids: number[]): string {
    return ids.map((i) => this.i2t[i] ?? '?').join('');
  }

  tok(i: number) {
    return this.i2t[i] ?? '?';
  }

  id(t: string) {
    return this.t2i.get(t) ?? this.unk;
  }
}

export interface EncodedSample {
  text: string;
  tokens: string[];
  ids: Int32Array;
  encIds: Int32Array;
  decIds: Int32Array;
  decTokens: string[];
  targets: Int32Array;
  weights: Float32Array;
  decTargets: Int32Array;
  decWeights: Float32Array;
  /** index within `ids` of every state-value token (the supervised payload) */
  valueIdx: number[];
}

const VALUE_RE = /^[0-9A-Fa-f]+$/;

/** a token counts as register payload if it is hex and not a register name */
export function isValueToken(tok: string, next?: string): boolean {
  if (!VALUE_RE.test(tok)) return false;
  if (tok.length === 1 && next && next.toUpperCase() === 'X') return false; // the "A" of "AX"
  return !/^[A-Z]{1,2}X$/i.test(tok);
}

/** teacher-forcing targets + per-position loss weights for a token list */
export function makeTargets(tokens: string[], ids: Int32Array | number[], stateWeight: number) {
  const n = ids.length;
  const targets = new Int32Array(n);
  const weights = new Float32Array(n);
  const valueIdx: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    targets[i] = ids[i + 1] as number;
    const v = isValueToken(tokens[i + 1]);
    weights[i] = v ? stateWeight : 1;
    if (v) valueIdx.push(i);
  }
  targets[n - 1] = ids[n - 1] as number;
  weights[n - 1] = 0;
  return { targets, weights, valueIdx };
}

export function encodeSample(
  text: string,
  encText: string,
  decText: string,
  vocab: Vocab,
  scheme: TokScheme,
  stateWeight: number,
): EncodedSample {
  const tokens = splitTokens(text, scheme);
  const ids = Int32Array.from(tokens.map((t) => vocab.id(t)));
  const main = makeTargets(tokens, ids, stateWeight);
  const decTokens = splitTokens(decText, scheme);
  const decIds = Int32Array.from(decTokens.map((t) => vocab.id(t)));
  const dec = makeTargets(decTokens, decIds, stateWeight);
  return {
    text,
    tokens,
    ids,
    encIds: Int32Array.from(vocab.encode(encText, scheme)),
    decIds,
    decTokens,
    targets: main.targets,
    weights: main.weights,
    decTargets: dec.targets,
    decWeights: dec.weights,
    valueIdx: main.valueIdx,
  };
}
