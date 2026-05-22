import { Layers } from 'lucide-react';
import { FEATURE_IMPORTANCE } from '../model/transportModel';

export default function FeatureImportance() {
  const maxImportance = Math.max(...FEATURE_IMPORTANCE.map((f) => f.importance));

  return (
    <div className="glass rounded-3xl p-8">
      <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
        <Layers className="w-5 h-5 text-brand-400" />
        Importancia de Variables del Sistema
      </h3>
      <p className="text-sm text-surface-200/50 mb-6">
        ¿Qué tanto influye cada variable en la predicción de demanda?
      </p>

      <div className="space-y-4">
        {FEATURE_IMPORTANCE.map((feature, index) => {
          const widthPercent = (feature.importance / maxImportance) * 100;
          const hue = 250 - (feature.importance / maxImportance) * 200;
          const color = `hsl(${hue}, 70%, 55%)`;

          return (
            <div key={feature.key} className="group">
              <div className="flex items-center justify-between text-sm mb-1.5">
                <span className="text-surface-200/80 group-hover:text-white transition-colors">
                  {feature.name}
                </span>
                <span className="font-mono font-semibold text-white/70 text-xs">
                  {(feature.importance * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-3 rounded-full bg-surface-900/60 overflow-hidden">
                <div
                  className="h-full rounded-full grow-bar group-hover:brightness-125 transition-all"
                  style={{
                    width: `${widthPercent}%`,
                    background: `linear-gradient(90deg, ${color}88, ${color})`,
                    animationDelay: `${0.3 + index * 0.12}s`,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <p className="mt-6 text-xs text-surface-200/40 leading-relaxed">
        * Importancia calculada por permutation importance sobre los datos históricos de transporte.
      </p>
    </div>
  );
}
