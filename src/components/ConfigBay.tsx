import { LabConfig } from '@/hooks/useLab';
import { OPS, OpName } from '@/lib/asm';
import { Btn, Field, Seg, Slider } from './ui';
import { cn } from '@/utils/cn';

function Group({
  tag,
  title,
  children,
  className,
  accent,
}: {
  tag: string;
  title: string;
  children: React.ReactNode;
  className?: string;
  accent: string;
}) {
  return (
    <div className={cn('panel panel-notch relative shrink-0 p-3.5', className)}>
      <div className="mb-3 flex items-center gap-2">
        <span className="font-mono text-[9.5px]" style={{ color: accent }}>
          {tag}
        </span>
        <span className="h-px flex-1 rail" />
        <span className="panel-title text-[10px] text-mute">{title}</span>
      </div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

export default function ConfigBay({
  cfg,
  patch,
  onRebuild,
  params,
  vocab,
}: {
  cfg: LabConfig;
  patch: (p: Partial<LabConfig>) => void;
  onRebuild: () => void;
  params: number;
  vocab: number;
}) {
  const toggleOp = (o: OpName) => {
    const has = cfg.ops.includes(o);
    if (has && cfg.ops.length === 1) return;
    patch({ ops: has ? cfg.ops.filter((x) => x !== o) : [...cfg.ops, o] });
  };

  return (
    <div className="flex gap-3 overflow-x-auto pb-2">
      {/* ---- architecture ---- */}
      <Group tag="01" title="architecture" accent="#ffb638" className="w-[330px]">
        <Seg
          value={cfg.arch}
          onChange={(v) => patch({ arch: v })}
          options={[
            { v: 'decoder', label: 'Causal Dec', hint: 'GPT-style stack, next-token over the whole trace' },
            { v: 'seq2seq', label: 'Seq2Seq', hint: 'Bidirectional encoder over ops, cross-attending decoder over states' },
            { v: 'slot', label: 'Slot Recur', hint: 'Persistent slot bank + attention-augmented GRU recurrence' },
          ]}
        />
        <div className="grid grid-cols-2 gap-3">
          <Field label="d_model" hint={String(cfg.dModel)}>
            <Slider value={cfg.dModel} min={16} max={96} step={8} onChange={(v) => patch({ dModel: v })} fmt={() => ''} />
          </Field>
          <Field label="layers" hint={String(cfg.nLayers)}>
            <Slider value={cfg.nLayers} min={1} max={4} onChange={(v) => patch({ nLayers: v })} fmt={() => ''} />
          </Field>
          {cfg.arch === 'slot' ? (
            <Field label="slots" hint={String(cfg.slots)}>
              <Slider value={cfg.slots} min={1} max={8} onChange={(v) => patch({ slots: v })} fmt={() => ''} />
            </Field>
          ) : (
            <Field label="heads" hint={String(cfg.nHeads)}>
              <Slider value={cfg.nHeads} min={1} max={8} onChange={(v) => patch({ nHeads: v })} fmt={() => ''} />
            </Field>
          )}
          <Field label="d_ff" hint={String(cfg.dff)}>
            <Slider value={cfg.dff} min={32} max={256} step={16} onChange={(v) => patch({ dff: v })} fmt={() => ''} />
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Field label="activation">
            <Seg
              size="sm"
              value={cfg.activation}
              onChange={(v) => patch({ activation: v })}
              options={[
                { v: 'gelu', label: 'gelu' },
                { v: 'relu', label: 'relu' },
              ]}
            />
          </Field>
          <Field label="position">
            <Seg
              size="sm"
              value={cfg.posMode}
              onChange={(v) => patch({ posMode: v })}
              options={[
                { v: 'learned', label: 'learn' },
                { v: 'sinusoid', label: 'sine' },
              ]}
            />
          </Field>
        </div>
      </Group>

      {/* ---- substrate ---- */}
      <Group tag="02" title="isa / substrate" accent="#2fe0cd" className="w-[300px]">
        <Field label="opcode set">
          <div className="flex gap-1">
            {OPS.map((o) => (
              <button
                key={o}
                onClick={() => toggleOp(o)}
                className={cn(
                  'btn flex-1 rounded-[2px] border py-1 text-[10px]',
                  cfg.ops.includes(o)
                    ? 'border-teal/70 bg-teal/15 text-teal'
                    : 'border-hull-600 bg-hull-900/60 text-dim hover:text-mute',
                )}
              >
                {o}
              </button>
            ))}
          </div>
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="reg width" hint={`${cfg.bits}-bit`}>
            <Seg
              size="sm"
              value={cfg.bits}
              onChange={(v) => patch({ bits: v })}
              options={[
                { v: 4, label: '4' },
                { v: 8, label: '8' },
                { v: 16, label: '16' },
              ]}
            />
          </Field>
          <Field label="observed regs" hint={String(cfg.nRegs)}>
            <Slider value={cfg.nRegs} min={2} max={4} onChange={(v) => patch({ nRegs: v })} fmt={() => ''} />
          </Field>
          <Field label="prog len" hint={`${cfg.progMin}–${cfg.progMax}`}>
            <div className="flex items-center gap-1">
              <input
                type="number"
                min={1}
                max={8}
                value={cfg.progMin}
                onChange={(e) => patch({ progMin: Math.max(1, Math.min(8, +e.target.value)) })}
                className="w-full rounded-[2px] border border-hull-600 bg-hull-950 px-1.5 py-1 text-[11px] text-chalk outline-none focus:border-teal"
              />
              <span className="text-dim">–</span>
              <input
                type="number"
                min={1}
                max={8}
                value={cfg.progMax}
                onChange={(e) => patch({ progMax: Math.max(cfg.progMin, Math.min(8, +e.target.value)) })}
                className="w-full rounded-[2px] border border-hull-600 bg-hull-950 px-1.5 py-1 text-[11px] text-chalk outline-none focus:border-teal"
              />
            </div>
          </Field>
          <Field label="corpus" hint={String(cfg.nSamples)}>
            <Slider value={cfg.nSamples} min={32} max={512} step={32} onChange={(v) => patch({ nSamples: v })} fmt={() => ''} />
          </Field>
        </div>
        <Field label="immediate bias" hint={cfg.immBias.toFixed(2)}>
          <Slider value={cfg.immBias} min={0} max={1} step={0.05} onChange={(v) => patch({ immBias: v })} fmt={() => ''} />
        </Field>
      </Group>

      {/* ---- tokenizer ---- */}
      <Group tag="03" title="tokenisation" accent="#ff5f7e" className="w-[262px]">
        <Field label="scheme">
          <Seg
            value={cfg.scheme}
            onChange={(v) => patch({ scheme: v })}
            options={[
              { v: 'char', label: 'char / byte', hint: 'every non-space character is a token' },
              { v: 'word', label: 'word / byte-run', hint: 'mnemonics + register names collapse into single tokens' },
            ]}
          />
        </Field>
        <Field label="state-token loss weight" hint={`×${cfg.stateWeight.toFixed(1)}`}>
          <Slider value={cfg.stateWeight} min={1} max={6} step={0.25} onChange={(v) => patch({ stateWeight: v })} fmt={() => ''} />
        </Field>
        <div className="rounded-[3px] border border-hull-600/60 bg-hull-950/70 p-2.5 font-mono text-[10px] leading-relaxed text-dim">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-mute">vocab</span>
            <span className="text-rose">{vocab} tok</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-mute">trace</span>
            <span className="text-rose">
              {cfg.scheme === 'char' ? '@0.0.0|MOV AX,7|7.0.0|#' : '@ 0 . 0 . 0 | MOV AX , 7 | …'}
            </span>
          </div>
        </div>
      </Group>

      {/* ---- optimiser ---- */}
      <Group tag="04" title="optimiser" accent="#b8f04a" className="w-[262px]">
        <Field label="adam lr" hint={cfg.lr.toFixed(4)}>
          <Slider value={cfg.lr} min={0.0002} max={0.02} step={0.0002} onChange={(v) => patch({ lr: v })} fmt={() => ''} />
        </Field>
        <Field label="batch" hint={`${cfg.batch} traces/step`}>
          <Slider value={cfg.batch} min={1} max={12} onChange={(v) => patch({ batch: v })} fmt={() => ''} />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="grad clip" hint={cfg.clip.toFixed(1)}>
            <Slider value={cfg.clip} min={0} max={5} step={0.5} onChange={(v) => patch({ clip: v })} fmt={() => ''} />
          </Field>
          <Field label="wd" hint={cfg.wd.toFixed(3)}>
            <Slider value={cfg.wd} min={0} max={0.05} step={0.001} onChange={(v) => patch({ wd: v })} fmt={() => ''} />
          </Field>
        </div>
        <div className="flex items-center gap-2 pt-1">
          <Btn tone="teal" onClick={onRebuild} className="flex-1">
            re-init corpus + weights
          </Btn>
          <span className="font-mono text-[10px] text-dim">{params.toLocaleString()} θ</span>
        </div>
      </Group>
    </div>
  );
}
