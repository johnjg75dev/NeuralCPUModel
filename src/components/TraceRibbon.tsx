import { useMemo, useState } from 'react';
import { Snap } from '@/hooks/useLab';
import { cn } from '@/utils/cn';

export default function TraceRibbon({ snap, live, scheme }: { snap: Snap; live: boolean; scheme: 'char' | 'word' }) {
  const [hover, setHover] = useState<number | null>(null);
  const { tokens, ids, probs, V } = snap;

  const rows = useMemo(() => {
    if (!probs || !tokens.length) return [];
    return tokens.map((t, i) => {
      if (i === 0 || !probs) return { tok: t, ok: null as boolean | null, p: 0, top: [] as { t: string; p: number }[] };
      const base = (i - 1) * V;
      let best = 0,
        bv = -Infinity;
      const cand: { j: number; p: number }[] = [];
      for (let j = 0; j < V; j++) {
        const p = probs[base + j];
        if (p > bv) {
          bv = p;
          best = j;
        }
        cand.push({ j, p });
      }
      cand.sort((a, b) => b.p - a.p);
      return {
        tok: t,
        ok: best === ids[i],
        p: probs[base + ids[i]] ?? 0,
        top: cand.slice(0, 5).map((c) => ({ t: snap.vocabList[c.j] ?? '?', p: c.p })),
      };
    });
  }, [probs, tokens, ids, V, snap.vocabList]);

  const score = useMemo(() => {
    const r = rows.filter((x) => x.ok !== null);
    const okc = r.filter((x) => x.ok).length;
    const meanP = r.reduce((a, b) => a + b.p, 0) / Math.max(1, r.length);
    return { n: r.length, okc, rate: r.length ? okc / r.length : 0, meanP };
  }, [rows]);

  return (
    <div className="panel relative z-20 px-4 py-3">
      <div className={cn('pointer-events-none absolute inset-x-0 top-0 h-px', live ? 'bg-amber/70' : 'bg-hull-600')} />
      <div className="flex flex-col gap-3 xl:flex-row xl:items-center">
        <div className="flex shrink-0 items-end gap-5 xl:w-[290px]">
          <div>
            <div className="panel-title text-[9.5px] text-dim">teacher-forced decode</div>
            <div className="font-display text-[34px] leading-[0.95] font-bold tracking-tight text-chalk">
              {(score.rate * 100).toFixed(0)}
              <span className="text-[16px] text-mute">%</span>
            </div>
            <div className="font-mono text-[9.5px] text-dim">
              {score.okc}/{score.n} next-token · p̄ {score.meanP.toFixed(3)}
            </div>
          </div>
          <div className="mb-1 flex items-center gap-1.5">
            <span className={cn('h-2 w-2 rounded-full', live ? 'led led-live bg-lime text-lime' : 'bg-hull-600')} />
            <span className="panel-title text-[9px] text-mute">{live ? 'streaming' : 'static'}</span>
          </div>
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-end gap-[3px]">
            {rows.map((r, i) => (
              <div
                key={i}
                className="relative"
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              >
                <div
                  className={cn(
                    'flex min-w-[22px] flex-col items-center rounded-[2px] border px-[5px] pt-[3px] transition-all duration-150',
                    r.ok === null && 'border-hull-700 bg-hull-900/70 text-dim',
                    r.ok === true && 'border-lime/50 bg-lime/[0.09] text-lime',
                    r.ok === false && 'border-rose/50 bg-rose/[0.08] text-rose',
                    hover === i && 'z-20 -translate-y-[2px] border-amber bg-amber/15 text-amber shadow-[0_0_18px_-6px_rgba(255,182,56,0.9)]',
                  )}
                >
                  <span className="font-mono text-[11.5px] leading-none">{r.tok === ' ' ? '␣' : r.tok}</span>
                  <span className="mt-[3px] h-[3px] w-full overflow-hidden rounded-[1px] bg-hull-800">
                    <span
                      className={cn('block h-full', r.ok ? 'bg-lime' : r.ok === null ? 'bg-hull-600' : 'bg-rose')}
                      style={{ width: `${Math.max(2, Math.min(100, r.p * 100))}%` }}
                    />
                  </span>
                </div>
                {hover === i && r.ok !== null && (
                  <div className="pop absolute top-[calc(100%+7px)] left-1/2 z-30 w-[136px] -translate-x-1/2 rounded-[3px] border border-hull-500 bg-hull-950/97 p-1.5 shadow-2xl">
                    <div className="mb-1 flex items-baseline justify-between">
                      <span className="font-mono text-[9px] text-dim">p(·|t&lt;{i})</span>
                      <span className="font-mono text-[9px] text-amber">true “{r.tok}”</span>
                    </div>
                    {r.top.map((c, k) => (
                      <div key={k} className="flex items-center gap-1 font-mono text-[9.5px]">
                        <span className={cn('w-6 shrink-0', c.t === r.tok ? 'text-lime' : 'text-mute')}>{c.t}</span>
                        <span className="h-[4px] flex-1 overflow-hidden rounded-[1px] bg-hull-800">
                          <span className={cn('block h-full', c.t === r.tok ? 'bg-lime' : 'bg-ice')} style={{ width: `${Math.max(1, c.p * 100)}%` }} />
                        </span>
                        <span className="w-8 shrink-0 text-right text-dim">{c.p.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="mt-2 flex items-center gap-2 font-mono text-[9px] text-dim">
            <span>tokenisation</span>
            <span className="text-ice">{scheme === 'char' ? 'char / byte' : 'word / byte-run'}</span>
            <span className="h-px flex-1 rail" />
            <span>
              ground truth from the <span className="text-teal">NEU-86</span> core · hover a cell for the full posterior
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
