import { Sparkles } from 'lucide-react';
import { useCallback, useRef, useState } from 'react';
import AboutModel from './components/AboutModel';
import CreditForm from './components/CreditForm';
import FeatureImportance from './components/FeatureImportance';
import Footer from './components/Footer';
import Hero from './components/Hero';
import Navbar from './components/Navbar';
import Resources from './components/Resources';
import RiskResult from './components/RiskResult';
import ReadmeViewer from './components/ReadmeViewer';
import { predictDemand, classifyDriver, recommendDestinations, checkServerHealth } from './model/transportModel';

export default function App() {
  const [demandResult, setDemandResult] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [recommendationResult, setRecommendationResult] = useState(null);
  const [moduleTab, setModuleTab] = useState('demand');
  const [isProcessing, setIsProcessing] = useState(false);
  const resultRef = useRef(null);
  const [showDocs, setShowDocs] = useState(false);
  const [serverStatus, setServerStatus] = useState('checking');

  const handlePredictDemand = useCallback(async (routeId, tab) => {
    setModuleTab(tab || 'demand');
    if (!routeId) return;
    setIsProcessing(true);
    await new Promise((r) => setTimeout(r, 1200));
    const result = predictDemand(routeId);
    setDemandResult(result);
    setIsProcessing(false);
    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }, []);

  const handleClassifyDriver = useCallback(async (fileName, tab) => {
    setModuleTab(tab || 'classification');
    setIsProcessing(true);
    const result = classifyDriver();
    await new Promise((r) => setTimeout(r, result.latency));
    setClassificationResult(result);
    setIsProcessing(false);
    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }, []);

  const handleRecommendDestinations = useCallback(async (clientId, tab) => {
    setModuleTab(tab || 'recommendation');
    setIsProcessing(true);
    await new Promise((r) => setTimeout(r, 1000));
    const destinations = recommendDestinations(clientId);
    setRecommendationResult({ destinations, clientId });
    setIsProcessing(false);
    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }, []);

  const handleToggleDocs = useCallback(() => {
    setShowDocs((prev) => {
      const newState = !prev;
      if (newState) {
        setTimeout(() => {
          document.getElementById('documentacion')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
      }
      return newState;
    });
  }, []);

  const handleShowDocs = useCallback(() => {
    setShowDocs(true);
    setTimeout(() => {
      document.getElementById('documentacion')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }, []);

  return (
    <div className="bg-animated min-h-screen">
      <Navbar onShowDocs={handleShowDocs} />

      <Hero serverStatus={serverStatus} />

      <section id="evaluacion" className="py-24 px-4 sm:px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass-light mb-4">
              <Sparkles className="w-4 h-4 text-brand-400" />
              <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
                Herramientas
              </span>
            </div>
            <h2 className="text-3xl sm:text-4xl font-black text-white mb-4">
              Funcionalidades del Sistema
            </h2>
            <p className="text-surface-200/60 max-w-xl mx-auto">
              Explora los tres módulos principales del sistema: predicción de demanda,
              clasificación de conductores y recomendación de destinos personalizados.
            </p>
          </div>

          <div className="glass rounded-3xl p-6 sm:p-8 glow-brand mb-8">
            <CreditForm
              onPredictDemand={handlePredictDemand}
              onClassifyDriver={handleClassifyDriver}
              onRecommendDestinations={handleRecommendDestinations}
              isProcessing={isProcessing}
              activeModule={moduleTab}
            />
          </div>

          <RiskResult
            ref={resultRef}
            demandResult={demandResult}
            classificationResult={classificationResult}
            recommendationResult={recommendationResult}
            moduleTab={moduleTab}
          />

          {demandResult && (
            <div className="mt-8">
              <FeatureImportance />
            </div>
          )}
        </div>
      </section>

      <AboutModel />
      <Resources onToggleDocs={handleToggleDocs} showDocs={showDocs} />
      {showDocs && <ReadmeViewer />}
      <Footer />
    </div>
  );
}
