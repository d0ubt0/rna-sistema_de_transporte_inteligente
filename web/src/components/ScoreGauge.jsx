import { useEffect, useState } from 'react';
import { getScoreCategory } from '../model/transportModel';

export default function ScoreGauge({ score, riskColor }) {
  const [animatedScore, setAnimatedScore] = useState(300);
  const category = getScoreCategory(score);

  useEffect(() => {
    const duration = 1500;
    const start = 300;
    const end = score;
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(start + (end - start) * eased);
      setAnimatedScore(current);
      if (progress < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [score]);

  const radius = 90;
  const circumference = 2 * Math.PI * radius;
  const normalizedScore = (score - 300) / 550;
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
            stroke={riskColor}
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            className="gauge-fill transition-all duration-1000"
            style={{ filter: `drop-shadow(0 0 8px ${riskColor}66)` }}
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-5xl font-black text-white animate-count" style={{ color: riskColor }}>
            {animatedScore}
          </span>
          <span className="text-sm font-medium text-surface-200/60 mt-1">de 850</span>
        </div>
      </div>

      <div
        className="mt-4 px-5 py-2 rounded-full text-sm font-semibold"
        style={{
          backgroundColor: `${category.color}15`,
          color: category.color,
          border: `1px solid ${category.color}30`,
        }}
      >
        {category.label}
      </div>

      <div className="flex justify-between w-full max-w-[220px] mt-3 text-xs text-surface-200/40">
        <span>300</span>
        <span>575</span>
        <span>850</span>
      </div>
    </div>
  );
}
