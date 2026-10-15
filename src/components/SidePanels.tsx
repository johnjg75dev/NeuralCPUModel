import { useEffect, useMemo, useState } from 'react';
import { EncodedSample } from '@/lib/tokenizer';
import { ThetaRow } from '@/hooks/useLab';
import { Panel } from './ui';
import { cn } from '@/utils/cn';

/* --------------------------- corpus feed --------------------------- */

export function CorpusFeed({
  data,
  scheme,
  vocab,
}: {
  data: EncodedSample[];
  scheme: 'char' | 'word';
  vocab: string[];
}) {
  const [seed, setSeed] = useState(0);
  const picks = useMemo(() => {
    if (!data.length) return [];
    const out: EncodedSample[] = [];
    for (let k = 0; k < 3; k++) out.push(data[(seed * 7 + k * 13) % data.length]);
    return out;
  }, [data, seed]);
  const join = scheme === 'char' ? '' : ' ';

  return (
    <Panel
      title="corpus feed"
      sub={`${data.length} auto-generated ground-truth traces`}
      accent="ice"
      right={
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="btn rounded-[2px] border border-hull-600 px-2 py-[3px] text-[9px] text-mute hover:border-ice hover:text-ice"
        >
          shuffle
        </button>
      }
      bodyClass="flex p-3 flex-col"
    >
      <div className="space-y-2.5">
        {picks.map((s, k) => {
          const valueSet = new Set(s.valueIdx.map((i) => i + 1));
          return (
            <div key={k} className="rounded-[3px] border border-hull-700/70 bg-hull-950/60 p-2.5">
              <div className="mb-1.5 flex items-center gap-2 font-mono text-[9px] text-dim">
                <span className="text-ice">trace #{((seed * 7 + k * 13) % data.length).toString().padStart(3, '0')}</span>
                <span className="h-px flex-1 rail" />
                <span>{s.ids.length} tok</span>
              </div>
              <div className="flex flex-wrap gap-[2px]">
                {s.tokens.map((t, i) => (
                  <span
                    key={i}
                    className={cn(
                      'rounded-[1px] px-[2px] py-[1px] font-mono text-[10px] leading-none',
                      t === '@' || t === '#'
                        ? 'bg-hull-700 text-mute'
                        : t === '|'
                          ? 'text-hull-500'
                          : valueSet.has(i)
                            ? 'bg-amber/15 text-amber'
                            : /^[A-Z]{2,}$/.test(t)
                              ? 'text-teal'
                              : 'text-mute',
                    )}
                  >
                    {t}
                  </span>
                ))}
              </div>
              <div className="mt-1.5 break-all font-mono text-[9.5px] text-dim">{s.tokens.join(join)}</div>
            </div>
          );
        })}
      </div>
      <div className="mt-3 border-t border-hull-600/50 pt-2 font-mono text-[9px] text-dim">
        <div className="mb-1.5 flex items-center gap-2">
          <span className="panel-title text-[8.5px] text-mute">learned vocabulary</span>
          <span className="h-px flex-1 rail" />
          <span>{vocab.length}</span>
        </div>
        <div className="flex flex-wrap gap-[3px]">
          {vocab.slice(0, 64).map((t) => (
            <span
              key={t}
              className={cn(
                'rounded-[1px] border px-1 py-[1px] font-mono text-[9px] leading-none',
                /^[A-Z]{2,}$/.test(t)
                  ? 'border-teal/40 bg-teal/10 text-teal'
                  : /^[0-9A-F]+$/.test(t)
                    ? 'border-amber/40 bg-amber/10 text-amber'
                    : 'border-hull-600 bg-hull-950 text-dim',
              )}
            >
              {t}
            </span>
          ))}
        </div>
      </div>
      <div className="mt-auto flex items-center gap-3 border-t border-hull-600/50 pt-2 font-mono text-[9px] text-dim">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-[1px] bg-amber/40" /> payload ×w
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-[1px] bg-teal/60" /> mnemonic
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-[1px] bg-hull-600" /> control
        </span>
      </div>
    </Panel>
  );
}

/* --------------------------- theta analytics --------------------------- */

export function ThetaAnalytics({ theta, version }: { theta: React.RefObject<ThetaRow[]>; version: number }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => tick((t) => t + 1), 400);
    return () => clearInterval(id);
  }, []);
  const rows = theta.current ?? [];
  const maxG = Math.max(1e-6, ...rows.map((r) => r.gn));
  void version;
  return (
    <Panel title="θ analytics" sub="per-tensor gradient ‖g‖₂ and weight RMS after the last backward pass" accent="lime" bodyClass="p-3">
      <div className="space-y-[5px]">
        {rows.map((r, i) => (
          <div key={`${r.name}#${i}`} className="group flex items-center gap-2 font-mono text-[9.5px]">
            <span className="w-[104px] shrink-0 truncate text-mute transition-colors group-hover:text-chalk" title={r.name}>
              {r.name}
            </span>
            <span className="relative h-[7px] flex-1 overflow-hidden rounded-[1px] bg-hull-900">
              <span
                className="absolute inset-y-0 left-0 rounded-[1px] bg-gradient-to-r from-rose/80 to-amber transition-[width] duration-300"
                style={{ width: `${Math.max(1, (r.gn / maxG) * 100)}%` }}
              />
            </span>
            <span className="w-[52px] shrink-0 text-right text-amber">{r.gn.toExponential(1)}</span>
            <span className="w-[42px] shrink-0 text-right text-dim">{r.wn.toFixed(3)}</span>
          </div>
        ))}
        {!rows.length && <div className="py-6 text-center font-mono text-[10px] text-dim">run a step to populate</div>}
      </div>
      <div className="mt-2 flex justify-between border-t border-hull-600/50 pt-2 font-mono text-[9px] text-dim">
        <span>‖g‖₂ (log-scaled to max)</span>
        <span>rms(w)</span>
      </div>
    </Panel>
  );
}
