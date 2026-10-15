import { ReactNode, useEffect, useRef, useState } from 'react';
import { cn } from '@/utils/cn';

/* ------------------------------ panel ------------------------------ */

export function Panel({
  title,
  sub,
  right,
  children,
  className,
  accent = 'amber',
  bodyClass,
}: {
  title: string;
  sub?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  accent?: 'amber' | 'teal' | 'rose' | 'lime' | 'ice';
  bodyClass?: string;
}) {
  const bar: Record<string, string> = {
    amber: 'bg-amber',
    teal: 'bg-teal',
    rose: 'bg-rose',
    lime: 'bg-lime',
    ice: 'bg-ice',
  };
  return (
    <section className={cn('panel panel-notch reveal flex flex-col', className)}>
      <header className="flex items-center gap-3 border-b border-hull-600/60 px-4 py-2.5">
        <span className={cn('h-3.5 w-[3px] shrink-0', bar[accent])} />
        <div className="min-w-0 flex-1">
          <h2 className="panel-title truncate text-[12.5px] font-semibold text-chalk">{title}</h2>
          {sub && <p className="truncate font-mono text-[10.5px] text-dim">{sub}</p>}
        </div>
        {right}
      </header>
      <div className={cn('flex-1 p-4', bodyClass)}>{children}</div>
    </section>
  );
}

/* ------------------------------ inputs ------------------------------ */

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="panel-title text-[9.5px] text-mute">{label}</span>
        {hint && <span className="font-mono text-[10px] text-amber">{hint}</span>}
      </div>
      {children}
    </label>
  );
}

export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  fmt,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}) {
  const label = fmt ? fmt(value) : String(value);
  return (
    <div className="flex items-center gap-2">
      <input
        type="range"
        className="w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      {label !== '' && <span className="w-12 shrink-0 text-right font-mono text-[11px] text-chalk">{label}</span>}
    </div>
  );
}

export function Seg<T extends string | number>({
  options,
  value,
  onChange,
  size = 'md',
}: {
  options: { v: T; label: string; hint?: string }[];
  value: T;
  onChange: (v: T) => void;
  size?: 'sm' | 'md';
}) {
  return (
    <div className="flex flex-wrap gap-1 rounded-[3px] border border-hull-600/70 bg-hull-950/60 p-[3px]">
      {options.map((o) => (
        <button
          key={String(o.v)}
          title={o.hint}
          onClick={() => onChange(o.v)}
          className={cn(
            'btn flex-1 whitespace-nowrap rounded-[2px] px-2 text-center',
            size === 'sm' ? 'py-[3px] text-[9.5px]' : 'py-1 text-[10.5px]',
            value === o.v
              ? 'bg-amber text-hull-950 shadow-[0_0_14px_-2px_rgba(255,182,56,0.7)]'
              : 'text-mute hover:bg-hull-700/70 hover:text-chalk',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Btn({
  children,
  onClick,
  tone = 'ghost',
  className,
  disabled,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  tone?: 'amber' | 'teal' | 'rose' | 'ghost';
  className?: string;
  disabled?: boolean;
  title?: string;
}) {
  const tones: Record<string, string> = {
    amber: 'bg-amber text-hull-950 hover:bg-[#ffc65c] shadow-[0_0_20px_-6px_rgba(255,182,56,0.9)]',
    teal: 'bg-teal text-hull-950 hover:bg-[#5ceada] shadow-[0_0_20px_-6px_rgba(47,224,205,0.9)]',
    rose: 'bg-rose text-hull-950 hover:bg-[#ff839b] shadow-[0_0_20px_-6px_rgba(255,95,126,0.9)]',
    ghost: 'border border-hull-600 bg-hull-800/70 text-chalk hover:border-amber/70 hover:bg-hull-700',
  };
  return (
    <button
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        'btn rounded-[3px] px-3 py-1.5 text-[10.5px] font-semibold disabled:cursor-not-allowed disabled:opacity-40',
        tones[tone],
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Led({ on, color = 'lime', live }: { on: boolean; color?: 'lime' | 'amber' | 'rose' | 'teal'; live?: boolean }) {
  const map: Record<string, string> = { lime: 'text-lime', amber: 'text-amber', rose: 'text-rose', teal: 'text-teal' };
  return (
    <span
      className={cn('inline-block h-[7px] w-[7px] rounded-full', map[color], on ? 'led bg-current' : 'bg-hull-600 text-transparent')}
      style={on && live ? { animation: 'pulseLed 1.1s ease-in-out infinite' } : undefined}
    />
  );
}

export function Stat({
  label,
  value,
  unit,
  tone = 'chalk',
  big,
}: {
  label: string;
  value: string | number;
  unit?: string;
  tone?: 'chalk' | 'amber' | 'teal' | 'rose' | 'lime';
  big?: boolean;
}) {
  const c: Record<string, string> = {
    chalk: 'text-chalk',
    amber: 'text-amber',
    teal: 'text-teal',
    rose: 'text-rose',
    lime: 'text-lime',
  };
  return (
    <div className="min-w-0">
      <div className="panel-title text-[9px] text-dim">{label}</div>
      <div className={cn('font-mono leading-tight', big ? 'text-[26px] font-semibold' : 'text-[15px] font-medium', c[tone])}>
        {value}
        {unit && <span className="ml-0.5 text-[10px] text-dim">{unit}</span>}
      </div>
    </div>
  );
}

/* --------------------------- scroll reveal --------------------------- */

export function useReveal() {
  useEffect(() => {
    const seen = new WeakSet<Element>();
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('in');
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.05, rootMargin: '0px 0px -30px 0px' },
    );
    const scan = () => {
      document.querySelectorAll('.reveal').forEach((el) => {
        if (seen.has(el)) return;
        seen.add(el);
        io.observe(el);
      });
    };
    scan();
    const mo = new MutationObserver(scan);
    mo.observe(document.body, { childList: true, subtree: true });
    return () => {
      io.disconnect();
      mo.disconnect();
    };
  }, []);
}

/* ------------------------------ misc ------------------------------ */

export function useElementWidth<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((e) => setW(e[0].contentRect.width));
    ro.observe(ref.current);
    setW(ref.current.clientWidth);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export function Sparkline({ data, color = '#2fe0cd', height = 34 }: { data: number[]; color?: string; height?: number }) {
  if (!data.length) return <div className="h-[34px]" />;
  const w = 100;
  const max = Math.max(...data);
  const min = Math.min(...data);
  const rng = max - min || 1;
  const pts = data
    .map((d, i) => `${(i / Math.max(1, data.length - 1)) * w},${height - 2 - ((d - min) / rng) * (height - 6)}`)
    .join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" className="h-[34px] w-full">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
