/**
 * ============================================
 * Footer - Pie de página
 * ============================================
 */
import { Brain, Heart } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="border-t border-surface-700/30 py-10 px-4 sm:px-6">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-600 to-brand-800 flex items-center justify-center">
            <Brain className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-white">TransportAI</span>
        </div>

        {/* Middle text */}
        <p className="text-sm text-surface-200/40 text-center">
          Proyecto académico · Red Neuronal para Predicción, Clasificación y Recomendación en Transporte
        </p>

        {/* Right */}
        <p className="text-sm text-surface-200/40 flex items-center gap-1">
          Desarrollado con <Heart className="w-3.5 h-3.5 text-red-400" /> en 2026
        </p>
      </div>
    </footer>
  );
}
