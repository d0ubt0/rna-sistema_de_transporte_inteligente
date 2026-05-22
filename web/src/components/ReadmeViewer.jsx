import { Book } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
// Vite feature to import raw string
import readmeContent from '../../public/README.md?raw';

export default function ReadmeViewer() {
  return (
    <section id="documentacion" className="py-24 px-4 sm:px-6">
      <div className="max-w-5xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass-light mb-4">
            <Book className="w-4 h-4 text-brand-400" />
            <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
              Documentación
            </span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-black text-white mb-4">
            Reporte Técnico Completo
          </h2>
          <p className="text-surface-200/60 max-w-xl mx-auto">
            Explora el reporte técnico del proyecto con los 11 capítulos: portada, resumen,
            metodología, desarrollo técnico por módulo y resultados del sistema de transporte.
          </p>
        </div>

        {/* Markdown Content */}
        <div className="glass rounded-3xl p-6 sm:p-10 glow-brand overflow-hidden">
          <article className="prose prose-invert max-w-none 
                              prose-headings:text-white prose-p:text-surface-200/80 
                              prose-a:text-brand-400 hover:prose-a:text-brand-300
                              prose-strong:text-white prose-code:text-brand-300
                              prose-pre:bg-surface-900 prose-pre:border prose-pre:border-brand-500/20
                              prose-img:rounded-xl prose-img:border prose-img:border-brand-500/20">
            <ReactMarkdown 
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {readmeContent}
            </ReactMarkdown>
          </article>
        </div>
      </div>
    </section>
  );
}
