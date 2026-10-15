import { useEffect, useMemo, useRef, useState } from 'react';
import { CorpusCfg, Issue, REGS, fmtInstr, fmtVal, genSample, hexWidth, maskOf } from '@/lib/asm';
import { LabConfig } from '@/hooks/useLab';
import { RunResult } from '@/lib/lab';
import { Btn, Led, Panel, Stat } from './ui';
import { cn } from '@/utils/cn';

/* ------------------------- asm highlighter ------------------------- */

const HL = /(;[^\n]*)|(\b(?:MOV|ADD|XOR|SUB)\b)|(\b(?:E|R)?[ABCD]X\b|\b[ABCD][LH]\b)|(\b0[xX][0-9A-Fa-f]+\b|\b[0-9]+[Hh]?\b|\b[0-9A-Fa-f]+\b)|([,\[\]])/g;

function highlightLine(line: string) {
  const out: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  HL.lastIndex = 0;
  let k = 0;
  while ((m = HL.exec(line))) {
    if (m.index > last) out.push(<span key={k++}>{line.slice(last, m.index)}</span>);
    const [full, com, op, reg, num, punc] = m;
    const cls = com
      ? 'text-dim italic'
      : op
        ? 'text-amber font-semibold'
        : reg
          ? 'text-teal'
          : num
            ? 'text-lime'
            : punc
              ? 'text-rose'
              : 'text-chalk';
    out.push(
      <span key={k++} className={cls}>
        {full}
      </span>,
    );
    last = m.index + full.length;
  }
  if (last < line.length) out.push(<span key={k++}>{line.slice(last)}</span>);
  return out;
}

/* -------------------------- register bank -------------------------- */

function RegCell({ name, value, prev, width, active, hot }: { name: string; value: number; prev: number | null; width: number; active: boolean; hot: boolean }) {
  const changed = prev !== null && prev !== value;
  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-[3px] border px-2 py-1.5 transition-all duration-300',
        active ? 'border-hull-500 bg-hull-950/80' : 'border-hull-700/60 bg-hull-950/40 opacity-45',
        hot && 'border-amber/70 shadow-[0_0_18px_-6px_rgba(255,182,56,0.8)]',
      )}
    >
      <div className="flex items-baseline justify-between">
        <span className={cn('panel-title text-[9.5px]', hot ? 'text-amber' : 'text-dim')}>{name}</span>
        {changed && <span className="font-mono text-[8.5px] text-lime">Δ</span>}
      </div>
      <div
        className={cn(
          'font-mono text-[19px] leading-none tracking-[0.14em] transition-colors duration-300',
          changed ? 'txt-glow-amber text-amber' : active ? 'text-chalk' : 'text-dim',
        )}
      >
        {fmtVal(value, width)}
      </div>
      <div className="mt-1 flex gap-[2px]">
        {Array.from({ length: width * 4 }).map((_, i) => {
          const bit = (value >> (width * 4 - 1 - i)) & 1;
          return <span key={i} className={cn('h-[3px] flex-1 rounded-[1px]', bit ? 'bg-teal/80' : 'bg-hull-700')} />;
        })}
      </div>
    </div>
  );
}

function pretty(t: string, scheme: 'char' | 'word') {
  let s = scheme === 'char' ? t : t.replace(/\s*\.\s*/g, '.').replace(/\s*,\s*/g, ',').replace(/\s+\|\s+/g, '|');
  s = s.replace(/([A-Z]{3})([A-Z]{2},)/g, '$1 $2');
  return s.replace(/\|/g, '  │  ');
}

const PRESETS: { label: string; code: string }[] = [
  {
    label: 'xor-swap',
    code: `; classic register swap\nMOV AX,5\nMOV BX,9\nXOR AX,BX\nXOR BX,AX\nXOR AX,BX`,
  },
  {
    label: 'accumulate',
    code: `; doubling chain\nMOV AX,1\nADD AX,AX\nADD AX,AX\nADD AX,AX\nMOV BX,AX\nSUB BX,3`,
  },
  {
    label: 'clear/mask',
    code: `; zero idioms + wrap\nMOV CX,F\nXOR CX,CX\nMOV AX,A\nADD AX,9\nSUB AX,4`,
  },
];

/* ------------------------------ deck ------------------------------ */

export default function ProgramDeck({
  cfg,
  program,
  setProgram,
  issues,
  seedRegs,
  setSeedRegs,
  run,
  doRun,
  focusStep,
  setFocusStep,
  freeText,
  training,
  nInstr,
}: {
  cfg: LabConfig;
  program: string;
  setProgram: (s: string) => void;
  issues: Issue[];
  seedRegs: number[];
  setSeedRegs: (v: number[]) => void;
  run: RunResult | null;
  doRun: () => void;
  focusStep: number;
  setFocusStep: React.Dispatch<React.SetStateAction<number>>;
  freeText: string;
  training: boolean;
  nInstr: number;
}) {
  const lines = program.split('\n');
  const width = hexWidth(cfg.bits);
  const mask = maskOf(cfg.bits);
  const act = useMemo(() => Array.from({ length: cfg.nRegs }, (_, i) => i), [cfg.nRegs]);
  const [walking, setWalking] = useState(false);
  const preRef = useRef<HTMLPreElement | null>(null);
  const gutterRef = useRef<HTMLDivElement | null>(null);
  const nChecks = run?.checks.length ?? 0;
  useEffect(() => {
    if (!walking || nChecks === 0) return;
    const id = setInterval(() => setFocusStep((f) => (f + 1) % nChecks), 950);
    return () => clearInterval(id);
  }, [walking, nChecks, setFocusStep]);

  const before = run ? run.states[Math.min(focusStep, run.states.length - 1)] : seedRegs;
  const after = run ? run.states[Math.min(focusStep + 1, run.states.length - 1)] : null;
  const check = run?.checks[Math.min(focusStep, Math.max(0, run.checks.length - 1))];

  return (
    <Panel
      title="program deck"
      sub="x86 subset · MOV / ADD / XOR / SUB"
      accent="teal"
      className="min-w-0"
      right={
        <div className="flex items-center gap-2">
          <Led on={!issues.some((i) => i.level === 'error')} color={issues.some((i) => i.level === 'error') ? 'rose' : 'lime'} />
          <span className="font-mono text-[10px] text-dim">
            {nInstr} instr · {issues.length ? `${issues.length} diag` : 'clean'}
          </span>
        </div>
      }
      bodyClass="p-0"
    >
      {/* preset bar */}
      <div className="flex items-center gap-1 border-b border-hull-600/60 bg-hull-900/50 px-2 py-1.5">
        <span className="panel-title mr-1 text-[8.5px] text-dim">listing presets</span>
        {PRESETS.map((p) => (
          <button
            key={p.label}
            onClick={() => setProgram(p.code)}
            className="btn rounded-[2px] border border-hull-600 px-1.5 py-[2px] text-[9px] text-mute hover:border-teal hover:text-teal"
          >
            {p.label}
          </button>
        ))}
        <button
          onClick={() =>
            setProgram(
              '; random draw from the corpus distribution\n' +
                genSample(cfg as CorpusCfg, Math.random)
                  .instrs.map((i) => fmtInstr(i, cfg.bits))
                  .join('\n'),
            )
          }
          className="btn rounded-[2px] border border-hull-600 px-1.5 py-[2px] text-[9px] text-mute hover:border-amber hover:text-amber"
        >
          random
        </button>
        <span className="ml-auto font-mono text-[9px] text-dim">isa {cfg.ops.join('/')}</span>
      </div>

      {/* editor */}
      <div className="relative border-b border-hull-600/60 bg-hull-950/70">
        <div className="flex max-h-[188px] min-h-[188px] overflow-hidden">
          <div
            ref={gutterRef}
            className="w-9 shrink-0 select-none overflow-hidden border-r border-hull-700/60 bg-hull-900/60 py-2 text-right font-mono text-[11.5px] leading-[20px] text-dim"
          >
            {lines.map((_, i) => (
              <div key={i} className="pr-2">
                {String(i + 1).padStart(2, '0')}
              </div>
            ))}
          </div>
          <div className="relative flex-1">
            <pre
              ref={preRef}
              aria-hidden
              className="pointer-events-none absolute inset-0 overflow-hidden whitespace-pre-wrap break-words px-3 py-2 font-mono text-[11.5px] leading-[20px]"
            >
              {lines.map((l, i) => (
                <div key={i}>{highlightLine(l) as React.ReactNode}</div>
              ))}
            </pre>
            <textarea
              spellCheck={false}
              value={program}
              onScroll={(e) => {
                if (preRef.current) preRef.current.scrollTop = e.currentTarget.scrollTop;
                if (gutterRef.current) gutterRef.current.scrollTop = e.currentTarget.scrollTop;
              }}
              onChange={(e) => setProgram(e.target.value)}
              className="relative h-[188px] w-full resize-none overflow-auto whitespace-pre-wrap break-words bg-transparent px-3 py-2 font-mono text-[11.5px] leading-[20px] text-transparent caret-amber outline-none selection:bg-amber/30 selection:text-chalk"
            />
          </div>
        </div>
      </div>

      {/* issues */}
      {issues.length > 0 && (
        <div className="max-h-[62px] overflow-auto border-b border-hull-600/60 bg-rose/[0.06] px-3 py-1.5 font-mono text-[10px]">
          {issues.map((is, i) => (
            <div key={i} className={is.level === 'error' ? 'text-rose' : 'text-amber/80'}>
              <span className="text-dim">L{String(is.line).padStart(2, '0')}</span> {is.msg}
            </div>
          ))}
        </div>
      )}

      {/* seed + registers */}
      <div className="border-b border-hull-600/60 px-3 py-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="panel-title text-[9.5px] text-mute">
            register file · {cfg.bits}-bit <span className="text-dim">/ seed vector</span>
          </span>
          <div className="flex gap-1">
            {REGS.map((r, i) => (
              <input
                key={r}
                value={seedRegs[i].toString(16).toUpperCase()}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 16);
                  const next = seedRegs.slice();
                  next[i] = isNaN(v) ? 0 : v & mask;
                  setSeedRegs(next);
                }}
                className={cn(
                  'w-9 rounded-[2px] border bg-hull-950 px-1 py-[2px] text-center font-mono text-[10px] outline-none transition-colors',
                  i < cfg.nRegs ? 'border-hull-500 text-teal focus:border-teal' : 'border-hull-700 text-dim',
                )}
              />
            ))}
          </div>
        </div>
        <div className="grid grid-cols-4 gap-2">
          {REGS.map((r, i) => (
            <RegCell
              key={r}
              name={r}
              value={after ? after[i] : before[i]}
              prev={after ? before[i] : null}
              width={width}
              active={act.includes(i)}
              hot={!!after && after[i] !== before[i]}
            />
          ))}
        </div>
        {check && (
          <div className="mt-2 flex items-center gap-2 font-mono text-[10px] text-dim">
            <span className="text-mute">ALU</span>
            <span className="h-px flex-1 rail" />
            <span className="text-ice">{check.note}</span>
            <span className={cn('rounded-[2px] px-1.5 py-[1px]', check.exact ? 'bg-lime/15 text-lime' : 'bg-rose/15 text-rose')}>
              {check.exact ? 'NET ✓ EXACT' : 'NET ✗ DRIFT'}
            </span>
          </div>
        )}
      </div>

      {/* step strip */}
      <div className="px-3 py-3">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <Btn tone="teal" onClick={doRun} className="relative overflow-hidden">
            execute + interrogate net
          </Btn>
          <div className="flex overflow-hidden rounded-[3px] border border-hull-600">
            <button
              onClick={() => setFocusStep((f) => Math.max(0, f - 1))}
              className="btn px-2 py-1.5 text-[10.5px] text-mute hover:bg-hull-700 hover:text-chalk"
            >
              ←
            </button>
            <button
              onClick={() => setWalking(!walking)}
              disabled={!nChecks}
              className={cn(
                'btn px-2.5 py-1.5 text-[10.5px] disabled:opacity-30',
                walking ? 'bg-lime text-hull-950' : 'text-mute hover:bg-hull-700 hover:text-chalk',
              )}
            >
              {walking ? '❚❚ walk' : '▶ walk'}
            </button>
            <button
              onClick={() => setFocusStep((f) => Math.min(nChecks - 1, f + 1))}
              className="btn px-2 py-1.5 text-[10.5px] text-mute hover:bg-hull-700 hover:text-chalk"
            >
              →
            </button>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <Stat label="state exact" value={run ? `${(run.exactRate * 100).toFixed(0)}%` : '—'} tone={run && run.exactRate > 0.6 ? 'lime' : 'rose'} />
            <Stat label="token acc" value={run ? `${(run.tokenRate * 100).toFixed(0)}%` : '—'} tone="amber" />
            {training && (
              <span className="flex items-center gap-1 font-mono text-[9.5px] text-lime">
                <Led on live color="lime" /> live
              </span>
            )}
          </div>
        </div>

        <div className="space-y-1">
          {!run && (
            <div className="rounded-[3px] border border-dashed border-hull-600 px-3 py-6 text-center font-mono text-[10.5px] text-dim">
              press <span className="text-teal">EXECUTE</span> — the emulator produces ground truth, then every
              register-state token the network predicts is diffed against it.
            </div>
          )}
          {run?.checks.map((c, i) => (
            <button
              key={i}
              onClick={() => setFocusStep(i)}
              className={cn(
                'group w-full rounded-[3px] border px-2.5 py-1.5 text-left transition-all duration-200',
                i === focusStep
                  ? 'border-amber/70 bg-amber/[0.07] shadow-[0_0_22px_-10px_rgba(255,182,56,0.9)]'
                  : 'border-hull-700/70 bg-hull-900/40 hover:border-hull-500 hover:bg-hull-800/60',
              )}
            >
              <div className="flex items-center gap-2">
                <span className="font-mono text-[9px] text-dim">{String(i).padStart(2, '0')}</span>
                <span className="w-[92px] shrink-0 font-mono text-[11px] text-teal">{c.instr.text.replace(',', ', ')}</span>
                <span className="font-mono text-[11px] text-mute">{c.truthText}</span>
                <span className="ml-auto flex items-center gap-[3px]">
                  {c.tokenOk.map((ok, k) => (
                    <span
                      key={k}
                      className={cn('h-[10px] w-[5px] rounded-[1px] transition-colors', ok ? 'bg-lime' : 'bg-rose')}
                    />
                  ))}
                  <span className={cn('ml-1 font-mono text-[9.5px]', c.exact ? 'text-lime' : 'text-rose')}>
                    {c.exact ? '✓' : '✗'}
                  </span>
                </span>
              </div>
              {i === focusStep && (
                <div className="pop mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-hull-700/70 pt-1.5 font-mono text-[10px]">
                  <span className="text-dim">net →</span>
                  <span className="flex gap-[3px]">
                    {c.truthTokens.map((t, k) => (
                      <span
                        key={k}
                        className={cn(
                          'rounded-[2px] border px-1 py-[1px]',
                          c.tokenOk[k] ? 'border-lime/40 bg-lime/10 text-lime' : 'border-rose/50 bg-rose/10 text-rose',
                        )}
                      >
                        {c.predTokens[k] ?? '·'}
                        <span className="text-dim">/{t}</span>
                      </span>
                    ))}
                  </span>
                </div>
              )}
            </button>
          ))}
        </div>

        {freeText && (
          <div className="mt-3 rounded-[3px] border border-hull-600/70 bg-hull-950/60 p-2.5">
            <div className="mb-1 flex items-center gap-2">
              <span className="panel-title text-[9px] text-rose">neural free-run</span>
              <span className="h-px flex-1 rail" />
              <span className="font-mono text-[9px] text-dim">T=0.75 · unprompted continuation</span>
            </div>
            <div className="break-all font-mono text-[11px] leading-relaxed text-ice/90">{pretty(freeText, cfg.scheme)}</div>
          </div>
        )}
      </div>
    </Panel>
  );
}
