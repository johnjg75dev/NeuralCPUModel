import { useEffect, useMemo, useRef, useState } from 'react';
import { AttnRec } from '@/lib/model';
import { Panel, useElementWidth } from './ui';
import { cn } from '@/utils/cn';

/* perceptual heat ramp: hull → teal → amber → white */
const STOPS: [number, number[]][] = [
  [0.0, [8, 12, 22]],
  [0.18, [22, 60, 82]],
  [0.42, [30, 170, 165]],
  [0.66, [140, 224, 120]],
  [0.84, [255, 182, 56]],
  [1.0, [255, 246, 224]],
];

export function heat(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [a, ca] = STOPS[i - 1];
      const [b, cb] = STOPS[i];
      const k = (t - a) / (b - a || 1);
      const c = ca.map((x, j) => Math.round(x + (cb[j] - x) * k));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return 'rgb(255,246,224)';
}

function drawTile(cv: HTMLCanvasElement, rec: AttnRec, size: number, rowNorm: boolean, hi?: { r: number; c: number } | null) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = size * dpr;
  cv.height = size * dpr;
  cv.style.width = size + 'px';
  cv.style.height = size + 'px';
  const ctx = cv.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const { rows, cols, data } = rec;
  const img = ctx.createImageData(size * dpr, size * dpr);
  const cellW = (size * dpr) / cols;
  const cellH = (size * dpr) / rows;
  // per-row max for normalisation
  const rmax = new Float32Array(rows);
  let gmax = 0;
  for (let i = 0; i < rows; i++) {
    let m = 0;
    for (let j = 0; j < cols; j++) {
      const v = data[i * cols + j];
      if (v > m) m = v;
      if (v > gmax) gmax = v;
    }
    rmax[i] = m || 1;
  }
  gmax = gmax || 1;
  for (let y = 0; y < size * dpr; y++) {
    const i = Math.min(rows - 1, Math.floor(y / cellH));
    for (let x = 0; x < size * dpr; x++) {
      const j = Math.min(cols - 1, Math.floor(x / cellW));
      const raw = data[i * cols + j];
      const v = rowNorm ? raw / rmax[i] : raw / gmax;
      const t = Math.pow(Math.max(0, Math.min(1, v)), 0.55);
      let c: number[] = STOPS[STOPS.length - 1][1];
      for (let s = 1; s < STOPS.length; s++) {
        if (t <= STOPS[s][0]) {
          const [a, ca] = STOPS[s - 1];
          const [b, cb] = STOPS[s];
          const k = (t - a) / (b - a || 1);
          c = ca.map((x, idx) => x + (cb[idx] - x) * k);
          break;
        }
      }
      const o = (y * size * dpr + x) * 4;
      img.data[o] = c[0];
      img.data[o + 1] = c[1];
      img.data[o + 2] = c[2];
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  if (hi) {
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 1;
    ctx.strokeRect((hi.c * size) / cols + 0.5, (hi.r * size) / rows + 0.5, size / cols, size / rows);
    ctx.strokeStyle = 'rgba(255,182,56,0.5)';
    ctx.beginPath();
    ctx.moveTo(0, ((hi.r + 1) * size) / rows);
    ctx.lineTo(size, ((hi.r + 1) * size) / rows);
    ctx.stroke();
  }
}

function Tile({
  rec,
  size,
  rowNorm,
  selected,
  onSelect,
  hi,
}: {
  rec: AttnRec;
  size: number;
  rowNorm: boolean;
  selected: boolean;
  onSelect: () => void;
  hi?: { r: number; c: number } | null;
}) {
  const cv = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    if (cv.current) drawTile(cv.current, rec, size, rowNorm, hi);
  }, [rec, size, rowNorm, hi]);
  return (
    <button
      onClick={onSelect}
      className={cn(
        'group relative block rounded-[2px] border p-[3px] transition-all duration-200',
        selected ? 'border-amber bg-amber/10 shadow-[0_0_18px_-6px_rgba(255,182,56,0.9)]' : 'border-hull-600/70 hover:border-teal',
      )}
    >
      <canvas ref={cv} className="block" />
      <span
        className={cn(
          'absolute -top-[7px] left-1 rounded-[2px] px-1 font-mono text-[8px] transition-colors',
          selected ? 'bg-amber text-hull-950' : 'bg-hull-800 text-dim group-hover:text-teal',
        )}
      >
        L{rec.layer}·{rec.tag === 'cross' ? 'X' : rec.tag === 'enc' ? 'E' : 'H'}
        {rec.head}
      </span>
    </button>
  );
}

export default function AttentionGrid({ attn, tokens, scheme }: { attn: AttnRec[]; tokens: string[]; scheme: 'char' | 'word' }) {
  const [sel, setSel] = useState(0);
  const [rowNorm, setRowNorm] = useState(true);
  const [hi, setHi] = useState<{ r: number; c: number } | null>(null);
  const [probeRef, probeW] = useElementWidth<HTMLDivElement>();
  const rec = attn[Math.min(sel, Math.max(0, attn.length - 1))];
  const probeSize = Math.max(120, Math.min(300, Math.floor(probeW || 236)));

  const groups = useMemo(() => {
    const m = new Map<string, AttnRec[]>();
    for (const a of attn) {
      const k = a.tag === 'cross' ? 'cross' : a.tag === 'enc' ? 'encoder' : a.tag.startsWith('slot') ? 'slots' : 'self';
      if (!m.has(k)) m.set(k, []);
      m.get(k)!.push(a);
    }
    return [...m.entries()];
  }, [attn]);

  const topKeys = useMemo(() => {
    if (!rec || hi === null) return [];
    const row = Array.from({ length: rec.cols }, (_, j) => ({ j, v: rec.data[hi.r * rec.cols + j] }));
    return row.sort((a, b) => b.v - a.v).slice(0, 5);
  }, [rec, hi]);

  const tileSize = rec ? Math.max(52, Math.min(84, Math.round(760 / Math.max(4, Math.ceil(Math.sqrt(attn.length || 1)) * 2)))) : 64;

  return (
    <Panel
      title="attention field"
      sub={`${attn.length} head-matrices · ${rec ? `${rec.rows}×${rec.cols}` : '—'} · row = query, col = key`}
      accent="ice"
      right={
        <button
          onClick={() => setRowNorm(!rowNorm)}
          className="btn rounded-[2px] border border-hull-600 px-2 py-[3px] text-[9px] text-mute hover:border-ice hover:text-ice"
        >
          norm: {rowNorm ? 'row-max' : 'global'}
        </button>
      }
    >
      {!attn.length ? (
        <div className="py-10 text-center font-mono text-[11px] text-dim">no attention captured yet — execute a program</div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_252px]">
          <div className="space-y-3">
            {groups.map(([tag, list]) => (
              <div key={tag}>
                <div className="mb-1.5 flex items-center gap-2">
                  <span className="panel-title text-[9px] text-ice">{tag}</span>
                  <span className="h-px flex-1 rail" />
                  <span className="font-mono text-[9px] text-dim">{list.length}</span>
                </div>
                <div className="flex flex-wrap gap-2 pt-1">
                  {list.map((r) => (
                    <Tile
                      key={`${r.tag}-${r.layer}-${r.head}`}
                      rec={r}
                      size={attn.indexOf(r) === sel ? tileSize + 8 : tileSize}
                      rowNorm={rowNorm}
                      selected={attn.indexOf(r) === sel}
                      onSelect={() => {
                        setSel(attn.indexOf(r));
                        setHi(null);
                      }}
                      hi={attn.indexOf(r) === sel ? hi : null}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* probe */}
          <div className="min-w-0">
            <div className="mb-2 panel-title text-[9px] text-mute">probe · hover the selected tile</div>
            {rec && (
              <div
                ref={probeRef}
                className="relative rounded-[3px] border border-hull-600 bg-hull-950 p-[3px]"
                onMouseLeave={() => setHi(null)}
              >
                <Tile rec={rec} size={probeSize - 8} rowNorm={rowNorm} selected onSelect={() => {}} hi={hi} />
                <div
                  className="pointer-events-none absolute inset-0"
                  onMouseMove={(e) => {
                    const r = e.currentTarget.getBoundingClientRect();
                    setHi({
                      r: Math.floor(((e.clientY - r.top) / r.height) * rec.rows),
                      c: Math.floor(((e.clientX - r.left) / r.width) * rec.cols),
                    });
                  }}
                />
              </div>
            )}
            <div className="mt-2 min-h-[92px] rounded-[3px] border border-hull-700/70 bg-hull-950/70 p-2 font-mono text-[10px]">
              {hi && rec ? (
                <>
                  <div className="mb-1.5 flex items-center gap-1.5">
                    <span className="rounded-[2px] bg-amber px-1.5 py-[1px] text-hull-950">{rec.qTok[hi.r] ?? '?'}</span>
                    <span className="text-dim">attends to →</span>
                  </div>
                  {topKeys.map((k) => (
                    <div key={k.j} className="mb-[3px] flex items-center gap-1.5">
                      <span className="w-8 shrink-0 text-chalk">{rec.kTok[k.j] ?? '?'}</span>
                      <span className="h-[5px] rounded-[1px]" style={{ width: `${Math.max(2, k.v * 150)}px`, background: heat(k.v) }} />
                      <span className="ml-auto text-dim">{k.v.toFixed(3)}</span>
                    </div>
                  ))}
                </>
              ) : (
                <div className="text-dim">
                  <div className="mb-1 text-mute">sequence under test</div>
                  <div className="break-all text-[9.5px] leading-relaxed text-ice/70">{tokens.join(scheme === 'char' ? '' : ' ')}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
