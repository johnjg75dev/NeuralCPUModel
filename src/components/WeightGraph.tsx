import { useEffect, useMemo, useRef, useState } from 'react';
import { FfnSnap } from '@/lib/model';
import { Panel } from './ui';
import { cn } from '@/utils/cn';

interface NodePos {
  x: number;
  y: number;
  i: number;
  act: number;
  w?: number;
}

function pick(n: number, want: number): number[] {
  if (n <= want) return Array.from({ length: n }, (_, i) => i);
  const out: number[] = [];
  for (let k = 0; k < want; k++) out.push(Math.round((k * (n - 1)) / (want - 1)));
  return out;
}

export default function WeightGraph({
  ffn,
  tokens,
  arch,
  layer,
  setLayer,
  nLayers,
}: {
  ffn: FfnSnap | null;
  tokens: string[];
  arch: string;
  layer: number;
  setLayer: (n: number) => void;
  nLayers: number;
}) {
  const [row, setRow] = useState(0);
  const [hover, setHover] = useState<{ col: number; idx: number } | null>(null);
  const [pulse, setPulse] = useState(true);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const W = 720;
  const H = 340;
  const IN = 11;
  const HID = 15;
  const OUT = 9;

  const model = useMemo(() => {
    if (!ffn) return null;
    const W1 = ffn.W1;
    const W2 = ffn.W2;
    const r = Math.min(row, ffn.input.r - 1);
    const inIdx = pick(W1.r, IN);
    const hidIdx = pick(W1.c, HID);
    const outIdx = pick(W2.c, OUT);

    const inAct = inIdx.map((i) => ffn.input.data[r * ffn.input.c + i] ?? 0);
    const hidAct = hidIdx.map((h) => ffn.hidden.data[Math.min(r, ffn.hidden.r - 1) * ffn.hidden.c + h] ?? 0);

    let maxW = 1e-6;
    for (let i = 0; i < W1.data.length; i++) maxW = Math.max(maxW, Math.abs(W1.data[i]));

    const inN: NodePos[] = inIdx.map((i, k) => ({ x: 62, y: 30 + (k * (H - 70)) / (inIdx.length - 1), i, act: inAct[k] }));
    const hidN: NodePos[] = hidIdx.map((h, k) => ({ x: W / 2, y: 26 + (k * (H - 60)) / (hidIdx.length - 1), i: h, act: hidAct[k] }));
    const outN: NodePos[] = outIdx.map((o, k) => ({ x: W - 62, y: 30 + (k * (H - 70)) / (outIdx.length - 1), i: o, act: 0 }));

    const edges: { x1: number; y1: number; x2: number; y2: number; w: number; a: number; b: number; strong: boolean }[] = [];
    inN.forEach((n, ai) => {
      hidN.forEach((m, bi) => {
        const w = W1.data[n.i * W1.c + m.i];
        const mag = Math.abs(w) / maxW;
        edges.push({ x1: n.x, y1: n.y, x2: m.x, y2: m.y, w, a: ai, b: bi, strong: mag > 0.62 });
      });
    });
    let maxW2 = 1e-6;
    for (let i = 0; i < W2.data.length; i++) maxW2 = Math.max(maxW2, Math.abs(W2.data[i]));
    const edges2: typeof edges = [];
    hidN.forEach((n, ai) => {
      outN.forEach((m, bi) => {
        const w = W2.data[n.i * W2.c + m.i];
        const mag = Math.abs(w) / maxW2;
        edges2.push({ x1: n.x, y1: n.y, x2: m.x, y2: m.y, w, a: ai, b: bi, strong: mag > 0.62 });
      });
    });

    // weight histogram
    const bins = new Array(41).fill(0);
    const stride = Math.max(1, Math.floor(W1.data.length / 4000));
    let cnt = 0,
      mean = 0,
      sd = 0;
    for (let i = 0; i < W1.data.length; i += stride) {
      const v = W1.data[i];
      cnt++;
      mean += v;
      bins[Math.max(0, Math.min(40, Math.round(((v / maxW) * 20) + 20)))]++;
    }
    mean /= cnt || 1;
    for (let i = 0; i < W1.data.length; i += stride) sd += (W1.data[i] - mean) ** 2;
    sd = Math.sqrt(sd / (cnt || 1));

    return { inN, hidN, outN, edges, edges2, maxW, maxW2, bins, sd, mean, r, W1, W2, b1: ffn.b1 };
  }, [ffn, row]);

  useEffect(() => {
    if (ffn) setRow((r) => Math.min(r, ffn.input.r - 1));
  }, [ffn]);

  if (!ffn || !model) {
    return (
      <Panel title="synapse graph" sub="feed-forward weights" accent="rose">
        <div className="py-16 text-center font-mono text-[11px] text-dim">waiting for a forward pass…</div>
      </Panel>
    );
  }

  const stroke = (w: number, max: number, dim: boolean) => {
    const mag = Math.min(1, Math.abs(w) / max);
    const col = w >= 0 ? '47,224,205' : '255,95,126';
    return {
      stroke: `rgba(${col},${dim ? 0.05 : 0.1 + mag * 0.75})`,
      width: 0.35 + mag * 2.1,
    };
  };

  const isHot = (col: number, idx: number) => (hover ? hover.col === col && hover.idx === idx : false);
  const touches = (e: { a: number; b: number }, stage: 0 | 1) => {
    if (!hover) return false;
    if (stage === 0) return (hover.col === 0 && hover.idx === e.a) || (hover.col === 1 && hover.idx === e.b);
    return (hover.col === 1 && hover.idx === e.a) || (hover.col === 2 && hover.idx === e.b);
  };

  return (
    <Panel
      title={arch === 'slot' ? 'recurrent synapse graph' : 'synapse graph'}
      sub={arch === 'slot' ? 'GRU candidate weights · live tensor values' : `feed-forward W₁ ${ffn.W1.r}×${ffn.W1.c} · W₂ ${ffn.W2.r}×${ffn.W2.c}`}
      accent="rose"
      right={
        <div className="flex items-center gap-1">
          {arch !== 'slot' && nLayers > 1 && (
            <div className="mr-1 flex overflow-hidden rounded-[2px] border border-hull-600">
              {Array.from({ length: nLayers }).map((_, i) => (
                <button
                  key={i}
                  onClick={() => setLayer(i)}
                  className={cn('px-1.5 py-[2px] font-mono text-[9px]', i === layer ? 'bg-rose text-hull-950' : 'text-dim hover:text-chalk')}
                >
                  L{i}
                </button>
              ))}
            </div>
          )}
          <button
            onClick={() => setPulse(!pulse)}
            className="btn rounded-[2px] border border-hull-600 px-2 py-[3px] text-[9px] text-mute hover:border-rose hover:text-rose"
          >
            flux {pulse ? 'on' : 'off'}
          </button>
        </div>
      }
      bodyClass="p-3"
    >
      <div className="mb-2 flex items-center gap-3">
        <span className="panel-title shrink-0 text-[9px] text-mute">activation row</span>
        <input
          type="range"
          className="w-full"
          min={0}
          max={Math.max(0, ffn.input.r - 1)}
          value={model.r}
          onChange={(e) => setRow(+e.target.value)}
        />
        <span className="shrink-0 rounded-[2px] border border-hull-600 bg-hull-950 px-2 py-[2px] font-mono text-[10px] text-amber">
          {tokens[model.r] ?? '—'} <span className="text-dim">#{model.r}</span>
        </span>
      </div>

      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full rounded-[3px] border border-hull-700/60 bg-hull-950/70">
        <defs>
          <radialGradient id="nodeGlow">
            <stop offset="0%" stopColor="rgba(255,182,56,0.55)" />
            <stop offset="100%" stopColor="rgba(255,182,56,0)" />
          </radialGradient>
        </defs>
        {[0, 1, 2].map((c) => (
          <text
            key={c}
            x={c === 0 ? 62 : c === 1 ? W / 2 : W - 62}
            y={14}
            textAnchor="middle"
            className="fill-[#55627d] font-mono"
            fontSize="9"
            letterSpacing="1.6"
          >
            {c === 0 ? (arch === 'slot' ? 'CTX ⊕ x_t' : 'x · LN') : c === 1 ? (arch === 'slot' ? 'CANDIDATE' : 'GELU·h') : arch === 'slot' ? 'U_n·h' : 'W₂·h'}
          </text>
        ))}

        {/* stage 1 edges */}
        {model.edges.map((e, k) => {
          const s = stroke(e.w, model.maxW, hover ? !touches(e, 0) : false);
          return (
            <line
              key={`a${k}`}
              x1={e.x1}
              y1={e.y1}
              x2={e.x2}
              y2={e.y2}
              stroke={s.stroke}
              strokeWidth={s.width}
              className={pulse && e.strong ? 'edge-live' : undefined}
              strokeDasharray={pulse && e.strong ? '3 9' : undefined}
            />
          );
        })}
        {/* stage 2 edges */}
        {model.edges2.map((e, k) => {
          const s = stroke(e.w, model.maxW2, hover ? !touches(e, 1) : false);
          return (
            <line
              key={`b${k}`}
              x1={e.x1}
              y1={e.y1}
              x2={e.x2}
              y2={e.y2}
              stroke={s.stroke}
              strokeWidth={s.width}
            />
          );
        })}

        {/* input nodes */}
        {model.inN.map((n, k) => {
          const a = Math.min(1, Math.abs(n.act) / 2.2);
          const hot = isHot(0, k);
          return (
            <g key={k} onMouseEnter={() => setHover({ col: 0, idx: k })} onMouseLeave={() => setHover(null)} className="cursor-crosshair">
              {a > 0.15 && <circle cx={n.x} cy={n.y} r={14 + a * 10} fill="url(#nodeGlow)" opacity={a * 0.6} />}
              <circle cx={n.x} cy={n.y} r={4 + a * 4.4} fill={n.act >= 0 ? '#2fe0cd' : '#ff5f7e'} opacity={0.25 + a * 0.75} stroke={hot ? '#ffb638' : 'rgba(233,239,252,0.25)'} strokeWidth={hot ? 1.6 : 0.7} />
              <text x={n.x - 15} y={n.y + 3} textAnchor="end" fontSize="8" className="fill-[#55627d] font-mono">
                d{n.i}
              </text>
            </g>
          );
        })}
        {/* hidden nodes */}
        {model.hidN.map((n, k) => {
          const a = Math.min(1, Math.abs(n.act) / 1.8);
          const hot = isHot(1, k);
          return (
            <g key={k} onMouseEnter={() => setHover({ col: 1, idx: k })} onMouseLeave={() => setHover(null)} className="cursor-crosshair">
              {a > 0.1 && <circle cx={n.x} cy={n.y} r={13 + a * 12} fill="url(#nodeGlow)" opacity={a * 0.55} />}
              <rect
                x={n.x - (3.6 + a * 4.4)}
                y={n.y - (3.6 + a * 4.4)}
                width={2 * (3.6 + a * 4.4)}
                height={2 * (3.6 + a * 4.4)}
                rx="1.5"
                fill={n.act >= 0 ? '#ffb638' : '#63a9ff'}
                opacity={0.22 + a * 0.78}
                stroke={hot ? '#e9effc' : 'rgba(233,239,252,0.22)'}
                strokeWidth={hot ? 1.6 : 0.7}
              />
              <text x={n.x} y={n.y - 12} textAnchor="middle" fontSize="7.5" className="fill-[#55627d] font-mono">
                h{n.i}
              </text>
            </g>
          );
        })}
        {/* output nodes */}
        {model.outN.map((n, k) => {
          const hot = isHot(2, k);
          return (
            <g key={k} onMouseEnter={() => setHover({ col: 2, idx: k })} onMouseLeave={() => setHover(null)} className="cursor-crosshair">
              <circle cx={n.x} cy={n.y} r={4.4} fill="#b8f04a" opacity={hot ? 1 : 0.5} stroke={hot ? '#e9effc' : 'rgba(233,239,252,0.2)'} strokeWidth={hot ? 1.5 : 0.7} />
              <text x={n.x + 15} y={n.y + 3} fontSize="8" className="fill-[#55627d] font-mono">
                d{n.i}
              </text>
            </g>
          );
        })}
      </svg>

      {/* readout strip */}
      <div className="mt-2 grid grid-cols-[minmax(0,1fr)_190px] gap-3">
        <div className="rounded-[3px] border border-hull-700/70 bg-hull-950/60 p-2 font-mono text-[9.5px] leading-relaxed">
          {hover ? (
            <div className="text-chalk">
              {hover.col === 0 && (
                <>
                  <span className="text-teal">input d{model.inN[hover.idx].i}</span> · activation{' '}
                  <span className="text-amber">{model.inN[hover.idx].act.toFixed(4)}</span>
                  <div className="mt-1 text-dim">
                    fan-out {model.W1.c} units · strongest edge{' '}
                    <span className="text-teal">
                      {(() => {
                        let best = 0,
                          bv = -1;
                        for (let j = 0; j < model.W1.c; j++) {
                          const v = Math.abs(model.W1.data[model.inN[hover.idx].i * model.W1.c + j]);
                          if (v > bv) {
                            bv = v;
                            best = j;
                          }
                        }
                        return `h${best} (${model.W1.data[model.inN[hover.idx].i * model.W1.c + best].toFixed(3)})`;
                      })()}
                    </span>
                  </div>
                </>
              )}
              {hover.col === 1 && (
                <>
                  <span className="text-amber">hidden h{model.hidN[hover.idx].i}</span> · post-activation{' '}
                  <span className="text-amber">{model.hidN[hover.idx].act.toFixed(4)}</span> · bias{' '}
                  <span className="text-ice">{model.b1.data[model.hidN[hover.idx].i].toFixed(3)}</span>
                </>
              )}
              {hover.col === 2 && (
                <>
                  <span className="text-lime">output d{model.outN[hover.idx].i}</span> · fan-in from {model.W2.r} hidden units
                </>
              )}
            </div>
          ) : (
            <div className="text-dim">
              <span className="text-mute">hover any node</span> — edge thickness = |w|, hue = sign.{' '}
              <span className="text-teal">teal</span> excitatory / <span className="text-rose">rose</span> inhibitory. Node
              size tracks the live activation for token <span className="text-amber">“{tokens[model.r]}”</span>.
            </div>
          )}
        </div>
        <div className="rounded-[3px] border border-hull-700/70 bg-hull-950/60 p-2">
          <div className="mb-1 flex items-baseline justify-between">
            <span className="panel-title text-[8.5px] text-mute">W₁ histogram</span>
            <span className="font-mono text-[8.5px] text-dim">σ {model.sd.toFixed(3)}</span>
          </div>
          <svg viewBox="0 0 170 40" className="h-10 w-full">
            {model.bins.map((b, i) => {
              const mx = Math.max(...model.bins) || 1;
              const hh = (b / mx) * 34;
              const neg = i < 20;
              return <rect key={i} x={(i * 170) / 41} y={38 - hh} width={170 / 41 - 0.6} height={hh} fill={neg ? 'rgba(255,95,126,0.75)' : 'rgba(47,224,205,0.75)'} />;
            })}
            <line x1={85} y1={2} x2={85} y2={38} stroke="rgba(233,239,252,0.25)" strokeWidth="0.6" />
          </svg>
        </div>
      </div>
    </Panel>
  );
}
