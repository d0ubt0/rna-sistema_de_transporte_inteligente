/**
 * ============================================
 * AboutModel - Sección descriptiva del proyecto
 * ============================================
 * Explica los tres módulos del sistema de transporte inteligente.
 */
import { BookOpen, Brain, Cpu, Database } from 'lucide-react';

const features = [
  {
    icon: Brain,
    title: 'Predicción de Demanda',
    description:
      'Modelo de series de tiempo basado en deep learning para predecir la demanda de transporte en rutas específicas durante los próximos 30 días.',
  },
  {
    icon: Database,
    title: 'Clasificación de Conductores',
    description:
      'Red neuronal convolucional entrenada para clasificar imágenes de conductores e identificar comportamientos distractores como uso de celular o somnolencia.',
  },

  {
    icon: Cpu,
    title: 'Recomendación de Destinos',
    description:
      'Sistema de filtrado colaborativo e híbrido que sugiere destinos personalizados basados en el historial y comportamiento en la plataforma de reservas.',
  },
  {
    icon: BookOpen,
    title: 'Métricas y Evaluación',
    description:
      'Cada módulo incluye métricas de rendimiento: RMSE/MAE para demanda, accuracy/F1-score para clasificación, Precision/Recall para recomendaciones.',
  }
];

export default function AboutModel() {
  return (
    <section id="acerca" className="py-24 px-4 sm:px-6">
      <div className="max-w-6xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass-light mb-4">
            <Brain className="w-4 h-4 text-brand-400" />
            <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
              Módulos del Sistema
            </span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-black text-white mb-4">
            Arquitectura del Proyecto
          </h2>
          <p className="text-surface-200/60 max-w-2xl mx-auto leading-relaxed">
            Sistema integral de aprendizaje profundo compuesto por tres módulos especializados
            que abordan predicción de demanda, seguridad vial y personalización de rutas.
          </p>
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-6">
          {features.map(({ icon: Icon, title, description }) => (
            <div
              key={title}
              className="glass rounded-2xl p-6 hover:bg-white/5 transition-all duration-300 group
                hover:border-brand-500/30 hover:translate-y-[-2px] hover:shadow-lg hover:shadow-brand-500/5"
            >
              <div className="w-12 h-12 rounded-xl bg-brand-600/15 flex items-center justify-center mb-4
                group-hover:bg-brand-600/25 group-hover:scale-110 transition-all duration-300">
                <Icon className="w-6 h-6 text-brand-400" />
              </div>
              <h3 className="font-bold text-white mb-2">{title}</h3>
              <p className="text-sm text-surface-200/60 leading-relaxed">{description}</p>
            </div>
          ))}
        </div>

        {/* Architecture diagram placeholder
        <div className="mt-16 glass rounded-3xl p-8 text-center">
          <h3 className="text-xl font-bold text-white mb-4">Flujo del Sistema</h3>
          <div className="flex items-center justify-center gap-2 sm:gap-4 flex-wrap py-8">
            {[
              { label: 'Datos de Entrada', sub: 'Historial + Imágenes', color: '#83a598' },
              { label: 'Módulo Demanda', sub: 'LSTM + Prophet', color: '#8ec07c' },
              { label: 'Módulo Conducción', sub: 'CNN + ResNet', color: '#fabd2f' },
              { label: 'Módulo Recomendación', sub: 'Filtrado Colaborativo', color: '#8ec07c' },
              { label: 'Resultados', sub: 'Predicciones + Alertas', color: '#83a598' },
            ].map((layer, i, arr) => (
              <div key={layer.label} className="flex items-center gap-2 sm:gap-4">
                <div
                  className="rounded-xl px-4 py-3 text-center min-w-[90px]"
                  style={{ backgroundColor: `${layer.color}20`, border: `1px solid ${layer.color}40` }}
                >
                  <p className="text-xs font-bold text-white">{layer.label}</p>
                  <p className="text-[10px] text-surface-200/50 mt-0.5">{layer.sub}</p>
                </div>
                {i < arr.length - 1 && (
                  <div className="text-brand-400/50 hidden sm:block">→</div>
                )}
              </div>
            ))}
          </div>
          <p className="text-xs text-surface-200/40 mt-4">
            RMSE, MAE (Demanda) | Accuracy, F1-Score (Conducción) | Precision, Recall (Recomendación)
          </p>
        </div> */}
      </div>
    </section>
  );
}
