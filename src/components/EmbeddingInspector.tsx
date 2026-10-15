import { useEffect, useMemo, useRef, useState } from 'react';
import { Panel } from './ui';
import { cn } from '@/utils/cn';

/** power-iteration PCA (top-2) over the V×D embedding table */
function pca2(emb: Float32Array, V: number, D: number): [Float32Array, Float32Array] {
  const mean = new Float32Array(D);
  for (let i = 0; i < V; i++) for (let j = 0; j < D; j++) mean[j] += emb[i * D + j];
  for (let j = 0; j < D; j++) mean[j] /= V;
  const cov = new Float32Array(D * D);
  for (let i = 0; i < V; i++) {
    for (let a = 0; a < D; a++) {
      const va = emb[i * D + a] - mean[a];
      for (let b = a; b < D; b++) cov[a * D + b] += va * (emb[i * D + b] - mean[b]);
    }
  }
  for (let a = 0; a < D; a++) for (let b = a; b < D; b++) {
    cov[a * D + b] /= V;
    cov[b * D + a] = cov[a * D + b];
  }
  const iter = (deflate: boolean, p1?: Float32Array, l1?: number) => {
    let v = new Float32Array(D);
    for (let j = 0; j < D; j++) v[j] = Math.sin(j + 1);
    let lam = 0;
    for (let it = 0; it < 60; it++) {
      const nv = new Float32Array(D);
      for (let a = 0; a < D; a++) {
        let s = 0;
        for (let b = 0; b < D; b++) s += cov[a * D + b] * v[b];
        nv[a] = s;
      }
      let n = 0;
      for (let a = 0; a < D; a++) n += nv[a] * nv[a];
      n = Math.sqrt(n) || 1;
      for (let a = 0; a < D; a++) nv[a] /= n;
      v = nv;
      lam = n;
    }
    if (deflate && p1 && l1) {
      for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) cov[a * D + b] -= l1! * p1[a] * p1[b];
    }
    return { v, lam };
  };
  const a = iter(false);
  const b = iter(true, a.v, a.lam);
  return [a.v, b.v];
}

export default function EmbeddingInspector({
  tokens,
  ids,
  emb,
  ctxEmb,
  V,
  D,
  version,
  vocabList,
  scheme,
}: {
  tokens: string[];
  ids: number[];
  emb: Float32Array | null;
  ctxEmb: Float32Array | null;
  V: number;
  D: number;
  version: number;
  vocabList: string[];
  scheme: 'char' | 'word';
}) {
  const [sel, setSel] = useState(0);
  const [mode, setMode] = useState<'ctx' | 'static'>('ctx');
  const [hoverBar, setHoverBar] = useState<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => setSel((s) => Math.min(s, Math.max(0, tokens.length - 1))), [tokens.length]);

  const basis = useMemo(() => (emb && V > 1 && D > 1 ? pca2(emb, V, D) : null), [emb, V, D, version]);

  const proj = useMemo(() => {
    if (!emb || !basis) return null;
    const [p1, p2] = basis;
    const out = new Float32Array(V * 2);
    for (let i = 0; i < V; i++) {
      let a = 0,
        b = 0;
      for (let j = 0; j < D; j++) {
        const v = emb[i * D + j];
        a += v * p1[j];
        b += v * p2[j];
      }
      out[i * 2] = a;
      out[i * 2 + 1] = b;
    }
    return out;
  }, [emb, basis, version, V, D]);

  const pos = Math.min(sel, Math.max(0, tokens.length - 1));
  const id = ids[pos] ?? 0;
  const vec: Float32Array | null = useMemo(() => {
    if (mode === 'static' && emb) return emb.subarray(id * D, id * D + D) as Float32Array;
    if (ctxEmb && ctxEmb.length >= (pos + 1) * D) return ctxEmb.subarray(pos * D, pos * D + D) as Float32Array;
    if (emb) return emb.subarray(id * D, id * D + D) as Float32Array;
    return null;
  }, [mode, emb, ctxEmb, id, pos, D]);

  const neighbours = useMemo(() => {
    if (!emb || !vec) return [];
    const dot = (a: ArrayLike<number>, b: ArrayLike<number>, o: number) => {
      let s = 0,
        na = 0,
        nb = 0;
      for (let j = 0; j < D; j++) {
        s += a[o + j] * b[j];
        na += a[o + j] * a[o + j];
        nb += b[j] * b[j];
      }
      return s / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
    };
    const list: { i: number; s: number }[] = [];
    for (let i = 0; i < V; i++) {
      if (i === id) continue;
      list.push({ i, s: dot(emb, vec, i * D) });
    }
    return list.sort((a, b) => b.s - a.s).slice(0, 9);
  }, [emb, vec, V, D, id, version]);

  const maxAbs = vec ? Math.max(0.001, ...Array.from(vec).map(Math.abs)) : 1;

  /* ---- latent scatter ---- */
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !proj) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = cv.clientWidth,
      h = cv.clientHeight;
    cv.width = w * dpr;
    cv.height = h * dpr;
    const ctx = cv.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    let mx = 1e-6;
    for (let i = 0; i < proj.length; i++) mx = Math.max(mx, Math.abs(proj[i]));
    const X = (v: number) => w / 2 + (v / mx) * (w / 2 - 12);
    const Y = (v: number) => h / 2 - (v / mx) * (h / 2 - 12);
    ctx.strokeStyle = 'rgba(56,71,106,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2, 4);
    ctx.lineTo(w / 2, h - 4);
    ctx.moveTo(4, h / 2);
    ctx.lineTo(w - 4, h / 2);
    ctx.stroke();
    for (let i = 0; i < V; i++) {
      const isSel = i === id;
      ctx.beginPath();
      ctx.arc(X(proj[i * 2]), Y(proj[i * 2 + 1]), isSel ? 4.5 : 2.2, 0, Math.PI * 2);
      ctx.fillStyle = isSel ? '#ffb638' : 'rgba(99,169,255,0.5)';
      ctx.fill();
      if (isSel) {
        ctx.strokeStyle = 'rgba(255,182,56,0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(X(proj[i * 2]), Y(proj[i * 2 + 1]), 9, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.font = '8px "IBM Plex Mono", monospace';
    ctx.fillStyle = 'rgba(126,141,169,0.9)';
    for (let i = 0; i < V; i++) {
      const x = X(proj[i * 2]),
        y = Y(proj[i * 2 + 1]);
      if (i === id || neighbours.slice(0, 4).some((n) => n.i === i)) {
        ctx.fillText(vocabList[i] ?? '?', x + 6, y + 3);
      }
    }
  }, [proj, id, V, neighbours, vocabList]);

  return (
    <Panel
      title="embedding inspector"
      sub={`latent table ${V}×${D} · ${mode === 'ctx' ? 'contextual (emb ⊕ pos)' : 'static lookup row'}`}
      accent="lime"
      right={
        <div className="flex overflow-hidden rounded-[2px] border border-hull-600">
          {(['ctx', 'static'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={cn('px-2 py-[3px] font-mono text-[9px]', m === mode ? 'bg-lime text-hull-950' : 'text-dim hover:text-chalk')}
            >
              {m}
            </button>
          ))}
        </div>
      }
      bodyClass="p-3"
    >
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_248px]">
        <div className="min-w-0">
          {/* sequence chips */}
          <div className="mb-2 flex flex-wrap gap-[3px]">
            {tokens.map((t, i) => (
              <button
                key={i}
                onClick={() => setSel(i)}
                className={cn(
                  'rounded-[2px] border px-1.5 py-[2px] font-mono text-[10px] transition-all duration-150',
                  i === pos
                    ? 'border-amber bg-amber text-hull-950 shadow-[0_0_14px_-3px_rgba(255,182,56,0.9)]'
                    : 'border-hull-600/70 bg-hull-950/60 text-mute hover:border-lime hover:text-lime',
                )}
              >
                {t === ' ' ? '␣' : t}
              </button>
            ))}
          </div>

          {/* bipolar latent strip */}
          <div className="relative mb-1 flex h-[74px] items-center gap-[2px] rounded-[3px] border border-hull-700/60 bg-hull-950/70 px-2 py-1">
            <div className="pointer-events-none absolute inset-x-2 top-1/2 h-px bg-hull-600/70" />
            {vec &&
              Array.from(vec).map((v, i) => (
                <div
                  key={i}
                  onMouseEnter={() => setHoverBar(i)}
                  onMouseLeave={() => setHoverBar(null)}
                  className="group relative flex h-full flex-1 cursor-crosshair items-center"
                >
                  <div
                    className={cn('w-full rounded-[1px] transition-all duration-200', v >= 0 ? 'bg-teal' : 'bg-rose')}
                    style={{
                      height: `${Math.max(2, (Math.abs(v) / maxAbs) * 32)}px`,
                      opacity: hoverBar === null || hoverBar === i ? 0.9 : 0.35,
                      marginTop: v >= 0 ? undefined : 0,
                      transform: v >= 0 ? 'translateY(-50%)' : 'translateY(50%)',
                      position: 'absolute',
                      top: '50%',
                      boxShadow: hoverBar === i ? '0 0 10px currentColor' : undefined,
                    }}
                  />
                </div>
              ))}
            <div className="pointer-events-none absolute right-2 top-1 font-mono text-[9px] text-dim">
              {hoverBar !== null && vec ? `d${hoverBar} = ${vec[hoverBar] >= 0 ? '+' : ''}${vec[hoverBar].toFixed(4)}` : `‖v‖∞ = ${maxAbs.toFixed(3)}`}
            </div>
          </div>

          {/* numeric dump */}
          <div className="max-h-[132px] overflow-auto rounded-[3px] border border-hull-700/60 bg-hull-950/50 p-2">
            <div className="grid grid-cols-4 gap-x-3 gap-y-[2px] font-mono text-[9.5px] sm:grid-cols-6 lg:grid-cols-8">
              {vec &&
                Array.from(vec).map((v, i) => (
                  <div
                    key={i}
                    className={cn('flex justify-between gap-1 rounded-[1px] px-1', hoverBar === i && 'bg-amber/15')}
                    onMouseEnter={() => setHoverBar(i)}
                    onMouseLeave={() => setHoverBar(null)}
                  >
                    <span className="text-dim">{String(i).padStart(2, '0')}</span>
                    <span className={v >= 0 ? 'text-teal' : 'text-rose'}>
                      {v >= 0 ? '+' : ''}
                      {v.toFixed(3)}
                    </span>
                  </div>
                ))}
            </div>
          </div>

          {/* neighbours */}
          <div className="mt-2">
            <div className="mb-1 flex items-center gap-2">
              <span className="panel-title text-[9px] text-mute">cosine neighbours of “{tokens[pos]}”</span>
              <span className="h-px flex-1 rail" />
            </div>
            <div className="flex flex-wrap gap-1">
              {neighbours.map((n) => (
                <span
                  key={n.i}
                  className="flex items-center gap-1.5 rounded-[2px] border border-hull-600/70 bg-hull-950/60 px-1.5 py-[2px] font-mono text-[9.5px]"
                >
                  <span className="text-chalk">{vocabList[n.i]}</span>
                  <span className="h-[4px] rounded-[1px] bg-lime" style={{ width: `${Math.max(1, n.s * 26)}px` }} />
                  <span className={n.s >= 0 ? 'text-lime' : 'text-rose'}>{n.s.toFixed(2)}</span>
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* latent projection */}
        <div className="min-w-0">
          <div className="mb-1 flex items-center gap-2">
            <span className="panel-title text-[9px] text-mute">PC1 × PC2</span>
            <span className="h-px flex-1 rail" />
            <span className="font-mono text-[8.5px] text-dim">{scheme}</span>
          </div>
          <canvas ref={canvasRef} className="h-[210px] w-full rounded-[3px] border border-hull-700/60 bg-hull-950/70" />
          <div className="mt-2 rounded-[3px] border border-hull-700/60 bg-hull-950/50 p-2 font-mono text-[9.5px] leading-relaxed text-dim">
            <div className="mb-1 text-mute">selected vector</div>
            <div>
              tok <span className="text-amber">“{tokens[pos] ?? '—'}”</span> id{' '}
              <span className="text-ice">{id}</span>
            </div>
            <div>
              pos <span className="text-teal">{pos}</span> / {Math.max(0, tokens.length - 1)} · dim{' '}
              <span className="text-teal">{D}</span>
            </div>
            <div className="mt-1 break-all text-[9px]">
              μ {(vec ? Array.from(vec).reduce((a, b) => a + b, 0) / D : 0).toFixed(4)} · σ{' '}
              {(vec
                ? Math.sqrt(Array.from(vec).reduce((a, b) => a + b * b, 0) / D - (Array.from(vec).reduce((a, b) => a + b, 0) / D) ** 2)
                : 0
              ).toFixed(4)}
            </div>
          </div>
        </div>
      </div>
    </Panel>
  );
}
