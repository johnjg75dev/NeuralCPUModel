import { useLab } from '@/hooks/useLab';
import ConfigBay from '@/components/ConfigBay';
import ProgramDeck from '@/components/ProgramDeck';
import TrainingDeck from '@/components/TrainingDeck';
import AttentionGrid from '@/components/AttentionGrid';
import WeightGraph from '@/components/WeightGraph';
import EmbeddingInspector from '@/components/EmbeddingInspector';
import TraceRibbon from '@/components/TraceRibbon';
import { CorpusFeed, ThetaAnalytics } from '@/components/SidePanels';
import { Sparkline, useReveal } from '@/components/ui';
import { cn } from '@/utils/cn';

const CHAIN = ['asm listing', 'tokenise', 'embed ⊕ pos', 'N × attn', 'N × ffn', 'logits', 'cross-entropy', '∇ tape', 'adam'];

function Chip() {
  return (
    <svg viewBox="0 0 44 44" className="h-11 w-11 shrink-0">
      <rect x="9" y="9" width="26" height="26" rx="2" fill="none" stroke="#ffb638" strokeWidth="1.4" />
      <rect x="15.5" y="15.5" width="13" height="13" rx="1" fill="rgba(255,182,56,0.16)" stroke="#2fe0cd" strokeWidth="1" />
      {[14, 19, 24, 29].map((p) => (
        <g key={p} stroke="#55627d" strokeWidth="1.2">
          <line x1={p} y1="3" x2={p} y2="9" />
          <line x1={p} y1="35" x2={p} y2="41" />
          <line x1="3" y1={p} x2="9" y2={p} />
          <line x1="35" y1={p} x2="41" y2={p} />
        </g>
      ))}
      <circle cx="22" cy="22" r="2.4" fill="#ff5f7e" />
    </svg>
  );
}

function HeadStat({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="min-w-[74px]">
      <div className="panel-title text-[8.5px] text-dim">{label}</div>
      <div className={cn('font-mono text-[15px] leading-tight font-medium', tone)}>{value}</div>
    </div>
  );
}

export default function App() {
  useReveal();
  const lab = useLab();
  const snap = lab.snapRef.current;
  const { stats, cfg } = lab;

  const lossSeries = (lab.histRef.current ?? []).slice(-260).map((s) => s.loss);

  return (
    <div className="relative min-h-screen">
      <div className="hull-bg" />
      <div className="hull-grid" />
      <div className="hull-scan" />
      <div className="hull-vignette" />

      {/* ---------------- header ---------------- */}
      <header className="sticky top-0 z-40 border-b border-hull-600/70 bg-hull-950/88 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1840px] flex-wrap items-center gap-x-6 gap-y-3 px-4 py-2.5 lg:px-6">
          <div className="flex items-center gap-3">
            <Chip />
            <div>
              <h1 className="font-display text-[27px] leading-none font-bold tracking-[0.01em] text-chalk">
                NEU<span className="text-amber">·</span>86
              </h1>
              <p className="mt-1 font-mono text-[9px] tracking-[0.2em] text-dim uppercase">
                neural x86 emulator × transformer laboratory
              </p>
            </div>
          </div>

          <div className="hidden items-center gap-1.5 xl:flex">
            {cfg.ops.map((o) => (
              <span
                key={o}
                className="group flex items-baseline gap-1.5 rounded-[2px] border border-hull-600/70 bg-hull-900/70 px-2 py-1 transition-colors hover:border-teal/70"
              >
                <span className="font-display text-[11px] font-semibold tracking-wider text-teal">{o}</span>
                <span className="font-mono text-[9px] text-dim transition-colors group-hover:text-mute">
                  {o === 'MOV' ? 'd←s' : o === 'ADD' ? 'd←d+s' : o === 'XOR' ? 'd←d⊕s' : 'd←d−s'}
                </span>
              </span>
            ))}
            <span className="ml-1 rounded-[2px] border border-amber/40 bg-amber/10 px-2 py-1 font-mono text-[9.5px] text-amber">
              {cfg.arch === 'decoder' ? 'causal-decoder' : cfg.arch === 'seq2seq' ? 'seq2seq enc/dec' : `slot-recurrent ×${cfg.slots}`}
            </span>
          </div>

          <div className="ml-auto flex items-center gap-4">
            <div className="hidden w-[132px] md:block">
              <div className="panel-title mb-[2px] text-[8.5px] text-dim">loss trajectory</div>
              <Sparkline data={lossSeries} color="#ffb638" height={26} />
            </div>
            <HeadStat label="step" value={stats.step.toLocaleString()} tone="text-chalk" />
            <HeadStat label="ema loss" value={stats.emaLoss ? stats.emaLoss.toFixed(3) : '—'} tone="text-amber" />
            <HeadStat label="tok acc" value={`${(stats.emaAcc * 100).toFixed(0)}%`} tone="text-teal" />
            <div className="flex items-center gap-2 rounded-[3px] border border-hull-600/70 bg-hull-900/70 px-2.5 py-1.5">
              <span className={cn('h-2 w-2 rounded-full', lab.training ? 'led led-live bg-lime text-lime' : 'bg-hull-600')} />
              <span className="panel-title text-[9px] text-mute">
                {lab.building ? 'compiling' : lab.training ? 'backprop live' : 'halted'}
              </span>
            </div>
          </div>
        </div>

        {/* signal chain */}
        <div className="relative overflow-hidden border-t border-hull-700/60 bg-hull-900/60">
          <div className="mx-auto flex max-w-[1840px] items-center gap-1 overflow-x-auto px-4 py-[5px] lg:px-6">
            {CHAIN.map((c, i) => (
              <span key={c} className="flex shrink-0 items-center gap-1">
                <span
                  className={cn(
                    'font-mono text-[9px] tracking-wide whitespace-nowrap',
                    i === 7 && lab.training ? 'text-rose' : i > 4 ? 'text-amber/80' : 'text-dim',
                  )}
                >
                  {c}
                </span>
                {i < CHAIN.length - 1 && (
                  <svg width="26" height="6" className="shrink-0 overflow-visible">
                    <line
                      x1="0"
                      y1="3"
                      x2="22"
                      y2="3"
                      stroke={lab.training ? 'rgba(255,182,56,0.75)' : 'rgba(56,71,106,0.8)'}
                      strokeWidth="1.2"
                      strokeDasharray="3 5"
                      className={lab.training ? 'edge-live' : undefined}
                    />
                  </svg>
                )}
              </span>
            ))}
            <span className="ml-auto hidden shrink-0 font-mono text-[9px] text-dim lg:block">
              {stats.ms.toFixed(1)} ms/step · {Math.round(stats.tps).toLocaleString()} tok/s · {stats.params.toLocaleString()} θ
            </span>
          </div>
        </div>
      </header>

      {/* ---------------- body ---------------- */}
      <main className="relative z-10 mx-auto max-w-[1840px] space-y-3 px-4 py-4 lg:px-6">
        <TraceRibbon snap={snap} live={lab.training} scheme={cfg.scheme} />

        <ConfigBay cfg={cfg} patch={lab.patch} onRebuild={lab.resetWeights} params={stats.params} vocab={stats.vocab} />

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
          <div className="xl:col-span-5">
            <ProgramDeck
              cfg={cfg}
              program={lab.program}
              setProgram={lab.setProgram}
              issues={lab.parsed.issues}
              seedRegs={lab.seedRegs}
              setSeedRegs={lab.setSeedRegs}
              run={lab.runResult}
              doRun={lab.doRun}
              focusStep={lab.focusStep}
              setFocusStep={lab.setFocusStep}
              freeText={lab.freeText}
              training={lab.training}
              nInstr={lab.parsed.instrs.length}
            />
          </div>
          <div className="xl:col-span-4">
            <TrainingDeck
              stats={stats}
              histRef={lab.histRef}
              training={lab.training}
              setRunning={lab.setRunning}
              stepOnce={lab.stepOnce}
              resetWeights={lab.resetWeights}
              building={lab.building}
              cfg={cfg}
            />
          </div>
          <div className="xl:col-span-3">
            <CorpusFeed data={lab.dataRef.current} scheme={cfg.scheme} vocab={lab.vocabRef.current?.i2t ?? []} />
          </div>
        </div>

        <AttentionGrid attn={snap.attn} tokens={snap.tokens} scheme={cfg.scheme} />

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
          <div className="xl:col-span-7">
            <WeightGraph
              ffn={snap.ffn}
              tokens={snap.tokens}
              arch={cfg.arch}
              layer={lab.ffnLayer}
              setLayer={lab.setFfnLayer}
              nLayers={cfg.nLayers}
            />
          </div>
          <div className="xl:col-span-5">
            <ThetaAnalytics theta={lab.thetaRef} version={stats.step} />
          </div>
        </div>

        <EmbeddingInspector
          tokens={snap.tokens}
          ids={snap.ids}
          emb={snap.emb}
          ctxEmb={snap.ctxEmb}
          V={snap.V}
          D={snap.D}
          version={snap.version}
          vocabList={snap.vocabList}
          scheme={cfg.scheme}
        />

        <footer className="reveal flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-hull-700/60 px-1 py-5 font-mono text-[9.5px] text-dim">
          <span className="text-mute">NEU·86 — everything on this page is computed in your browser.</span>
          <span>
            reverse-mode autodiff tape · {`matmul · matmulT · softmax · layernorm · gelu · gru`} — no ML framework, no
            network calls
          </span>
          <span className="ml-auto">
            arch <span className="text-amber">{cfg.arch}</span> · tok{' '}
            <span className="text-rose">{cfg.scheme}</span> · regs <span className="text-teal">{cfg.nRegs}×{cfg.bits}b</span>{' '}
            · isa <span className="text-teal">{cfg.ops.join('/')}</span>
          </span>
        </footer>
      </main>
    </div>
  );
}
