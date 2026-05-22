import { useEffect, useState } from 'react';

export default function CapacityGauge({ value, maxValue = 100, label = 'Capacidad', gaugeColor = '#83a598' }) {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    const duration = 1500;
    const start = 0;
    const end = Math.min(value, maxValue);
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(start + (end - start) * eased);
      setAnimatedValue(current);
      if (progress < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [value, maxValue]);

  const radius = 90;
  const circumference = 2 * Math.PI * radius;
  const normalizedScore = animatedValue / maxValue;
  const dashOffset = circumference * (1 - normalizedScore * 0.75);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-56 h-56">
        <svg viewBox="0 0 200 200" className="w-full h-full -rotate-[135deg]">
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke="rgba(168, 153, 132, 0.1)"
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * 0.25}
          />
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke={gaugeColor}
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            className="gauge-fill transition-all duration-1000"
            style={{ filter: `drop-shadow(0 0 8px ${gaugeColor}66)` }}
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-5xl font-black text-white animate-count" style={{ color: gaugeColor }}>
            {animatedValue}
          </span>
          <span className="text-sm font-medium text-surface-200/60 mt-1">de {maxValue}</span>
        </div>
      </div>

      <div
        className="mt-4 px-5 py-2 rounded-full text-sm font-semibold"
        style={{
          backgroundColor: `${gaugeColor}15`,
          color: gaugeColor,
          border: `1px solid ${gaugeColor}30`,
        }}
      >
        {label}
      </div>

      <div className="flex justify-between w-full max-w-[220px] mt-3 text-xs text-surface-200/40">
        <span>0</span>
        <span>{Math.round(maxValue / 2)}</span>
        <span>{maxValue}</span>
      </div>
    </div>
  );
}
