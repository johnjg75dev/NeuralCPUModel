import { useEffect, useRef, useState } from 'react';
import { StepStat } from '@/lib/lab';
import { Led, Panel, Stat } from './ui';

function fmt(n: number, d = 3) {
  if (!isFinite(n)) return '—';
  return n.toFixed(d);
}

export function LossChart({ histRef, height = 226 }: { histRef: React.RefObject<StepStat[]>; height?: number }) {
  const cv = useRef<HTMLCanvasElement | null>(null);
  const wrap = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<{ x: number; s: StepStat } | null>(null);
  const hoverRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = cv.current!;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;
    let w = 0,
      h = 0;
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    const resize = () => {
      const r = wrap.current!.getBoundingClientRect();
      w = r.width;
      h = height;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap.current!);

    const draw = () => {
      const hist = histRef.current ?? [];
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // frame + grid
      ctx.strokeStyle = 'rgba(56,71,106,0.35)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = Math.round((h - 18) * (i / 4)) + 6;
        ctx.beginPath();
        ctx.moveTo(30, y + 0.5);
        ctx.lineTo(w - 6, y + 0.5);
        ctx.stroke();
      }

      if (hist.length > 1) {
        const maxL = Math.max(0.4, ...hist.map((s) => Math.min(s.loss, 8)));
        const n = hist.length;
        const stride = Math.max(1, Math.floor(n / 900));
        const idx: number[] = [];
        for (let i = 0; i < n; i += stride) idx.push(i);
        if (idx[idx.length - 1] !== n - 1) idx.push(n - 1);
        const m = idx.length;
        const px = (k: number) => 30 + (idx[k] / (n - 1)) * (w - 38);
        const py = (v: number) => 6 + (1 - Math.min(v, maxL) / maxL) * (h - 24);

        // accuracy area
        ctx.beginPath();
        ctx.moveTo(px(0), h - 18);
        for (let k = 0; k < m; k++) ctx.lineTo(px(k), 6 + (1 - hist[idx[k]].acc) * (h - 24));
        ctx.lineTo(px(m - 1), h - 18);
        ctx.closePath();
        const g = ctx.createLinearGradient(0, 0, 0, h);
        g.addColorStop(0, 'rgba(47,224,205,0.22)');
        g.addColorStop(1, 'rgba(47,224,205,0)');
        ctx.fillStyle = g;
        ctx.fill();
        ctx.beginPath();
        for (let k = 0; k < m; k++) {
          const y = 6 + (1 - hist[idx[k]].acc) * (h - 24);
          k ? ctx.lineTo(px(k), y) : ctx.moveTo(px(k), y);
        }
        ctx.strokeStyle = 'rgba(47,224,205,0.85)';
        ctx.lineWidth = 1.2;
        ctx.stroke();

        // raw loss scatter-line
        ctx.beginPath();
        for (let k = 0; k < m; k++) {
          const y = py(hist[idx[k]].loss);
          k ? ctx.lineTo(px(k), y) : ctx.moveTo(px(k), y);
        }
        ctx.strokeStyle = 'rgba(255,182,56,0.32)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // ema loss
        let e = hist[0].loss;
        ctx.beginPath();
        for (let k = 0; k < m; k++) {
          const upto = idx[k];
          for (let i = (k ? idx[k - 1] + 1 : 0); i <= upto; i++) e = e * 0.94 + hist[i].loss * 0.06;
          const y = py(e);
          k ? ctx.lineTo(px(k), y) : ctx.moveTo(px(k), y);
        }
        ctx.strokeStyle = '#ffb638';
        ctx.lineWidth = 1.9;
        ctx.shadowColor = 'rgba(255,182,56,0.55)';
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // head marker
        const lx = px(m - 1);
        const ly = py(hist[n - 1].loss);
        ctx.fillStyle = '#ffb638';
        ctx.beginPath();
        ctx.arc(lx, ly, 2.6, 0, Math.PI * 2);
        ctx.fill();

        // y labels
        ctx.fillStyle = 'rgba(126,141,169,0.75)';
        ctx.font = '9px "IBM Plex Mono", monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
          const v = maxL * (1 - i / 4);
          ctx.fillText(v.toFixed(1), 26, 6 + (h - 24) * (i / 4) + 3);
        }
        ctx.textAlign = 'left';
        ctx.fillText(`s${hist[0].step}`, 32, h - 6);
        ctx.textAlign = 'right';
        ctx.fillText(`s${hist[n - 1].step}`, w - 6, h - 6);

        // hover crosshair
        if (hoverRef.current !== null) {
          const mx = hoverRef.current;
          const i = Math.round(((mx - 30) / (w - 38)) * (n - 1));
          if (i >= 0 && i < n) {
            const x = px(i);
            ctx.strokeStyle = 'rgba(233,239,252,0.35)';
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(x, 4);
            ctx.lineTo(x, h - 16);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#2fe0cd';
            ctx.beginPath();
            ctx.arc(x, 6 + (1 - hist[i].acc) * (h - 24), 2.4, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#ffb638';
            ctx.beginPath();
            ctx.arc(x, py(hist[i].loss), 2.4, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      } else {
        ctx.fillStyle = 'rgba(85,98,125,0.8)';
        ctx.font = '11px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('awaiting gradient descent …', w / 2, h / 2);
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [histRef, height]);

  return (
    <div
      ref={wrap}
      className="relative"
      onMouseMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - r.left;
        hoverRef.current = x;
        const hist = histRef.current ?? [];
        if (hist.length > 1) {
          const i = Math.round(((x - 30) / Math.max(1, r.width - 38)) * (hist.length - 1));
          if (i >= 0 && i < hist.length) setHover({ x, s: hist[i] });
          else setHover(null);
        }
      }}
      onMouseLeave={() => {
        hoverRef.current = null;
        setHover(null);
      }}
    >
      <canvas ref={cv} className="block w-full" />
      {hover && (
        <div
          className="pointer-events-none absolute top-1 z-10 -translate-x-1/2 rounded-[3px] border border-hull-500 bg-hull-950/95 px-2 py-1 font-mono text-[9.5px] whitespace-nowrap text-chalk shadow-lg"
          style={{ left: Math.max(56, Math.min(hover.x, (wrap.current?.clientWidth ?? 200) - 56)) }}
        >
          <span className="text-dim">step</span> {hover.s.step} <span className="text-amber">L {fmt(hover.s.loss)}</span>{' '}
          <span className="text-teal">a {(hover.s.acc * 100).toFixed(0)}%</span>{' '}
          <span className="text-dim">{hover.s.ms.toFixed(1)}ms</span>
        </div>
      )}
    </div>
  );
}

export default function TrainingDeck({
  stats,
  histRef,
  training,
  setRunning,
  stepOnce,
  resetWeights,
  building,
  cfg,
}: {
  stats: {
    step: number;
    emaLoss: number;
    loss: number;
    emaAcc: number;
    valLoss: number;
    valAcc: number;
    ms: number;
    tps: number;
    params: number;
    vocab: number;
    avgLen: number;
    seenTokens: number;
    elapsed: number;
    bestLoss: number;
  };
  histRef: React.RefObject<StepStat[]>;
  training: boolean;
  setRunning: (v: boolean) => void;
  stepOnce: () => void;
  resetWeights: () => void;
  building: boolean;
  cfg: { arch: string; dModel: number; nLayers: number; nHeads: number; slots: number; scheme: string };
}) {
  return (
    <Panel
      title="training telemetry"
      sub={`backprop on ${cfg.arch} · d=${cfg.dModel} · L=${cfg.nLayers}`}
      accent="amber"
      className="min-w-0"
      right={
        <span className="flex items-center gap-1.5 font-mono text-[9.5px] text-dim">
          <Led on={training} color={training ? 'lime' : 'amber'} live={training} />
          {building ? 'compiling…' : training ? 'descending' : 'idle'}
        </span>
      }
      bodyClass="flex p-3 flex-col"
    >
      <div className="mb-3 flex gap-2">
        <button
          onClick={() => setRunning(!training)}
          className={`btn relative flex-1 overflow-hidden rounded-[3px] py-2 text-[11px] font-semibold ${
            training ? 'bg-rose text-hull-950' : 'bg-amber text-hull-950 sweep'
          }`}
        >
          {training ? '■ halt' : '▶ run backprop'}
        </button>
        <button onClick={stepOnce} className="btn rounded-[3px] border border-hull-600 bg-hull-800/70 px-3 text-[11px] hover:border-teal hover:text-teal">
          step
        </button>
        <button onClick={resetWeights} className="btn rounded-[3px] border border-hull-600 bg-hull-800/70 px-3 text-[11px] hover:border-rose hover:text-rose" title="re-initialise weights & corpus">
          reset
        </button>
      </div>

      <div className="min-h-[226px] flex-1">
        <LossChart histRef={histRef} />
      </div>

      <div className="mt-auto grid grid-cols-4 gap-x-3 gap-y-2.5 border-t border-hull-600/50 pt-3">
        <Stat label="ema loss" value={fmt(stats.emaLoss)} tone="amber" big />
        <Stat label="token acc" value={`${(stats.emaAcc * 100).toFixed(1)}`} unit="%" tone="teal" big />
        <Stat label="val loss" value={fmt(stats.valLoss)} tone="rose" />
        <Stat label="val acc" value={`${(stats.valAcc * 100).toFixed(0)}`} unit="%" tone="lime" />
        <Stat label="step" value={stats.step.toLocaleString()} />
        <Stat label="ms / step" value={fmt(stats.ms, 1)} />
        <Stat label="tok / s" value={Math.round(stats.tps).toLocaleString()} />
        <Stat label="elapsed" value={`${stats.elapsed.toFixed(0)}s`} />
        <Stat label="θ params" value={stats.params.toLocaleString()} />
        <Stat label="vocab" value={stats.vocab} />
        <Stat label="avg seq" value={stats.avgLen.toFixed(1)} />
        <Stat label="tokens seen" value={(stats.seenTokens / 1000).toFixed(1)} unit="k" />
      </div>
    </Panel>
  );
}
