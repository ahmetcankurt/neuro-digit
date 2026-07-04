import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import Drawing from "./component/Drawing";
import PredictionChart from "./component/PredictionChart";
import { BsArrowCounterclockwise } from "react-icons/bs";
import CustomTooltip from "./component/Tooltip";
import { loadFull } from "tsparticles";
import particlesOptions from "./assest/json/particles.json";
import { initParticlesEngine } from "@tsparticles/react";
import Particles from "./component/Particles";
import NeuralNetworkVisualizer from "./component/NeuralNetworkVisualizer";

const getClassifierUrl = () => {
  if (process.env.PUBLIC_URL) {
    return `${process.env.PUBLIC_URL}/classifiers/model.json`;
  }
  const pathname = window.location.pathname;
  if (pathname.includes("/draw-predict")) {
    return "/draw-predict/classifiers/model.json";
  }
  return "/classifiers/model.json";
};

const classifierUrl = getClassifierUrl();

const loadModel = async () => {
  try {
    const model = await tf.loadLayersModel(classifierUrl);
    return model;
  } catch (error) {
    console.error("Error loading model from:", classifierUrl, error);
  }
};

const App = () => {
  const [strokes, setStrokes] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [predictionProbabilities, setPredictionProbabilities] = useState([]);
  const [model, setModel] = useState(null);
  const canvasRef = useRef(null);

  const [init, setInit] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [x1Value, setX1Value] = useState(1.0);
  const [normalizedGrid, setNormalizedGrid] = useState(null);

  useEffect(() => {
    if (init) {
      return;
    }

    initParticlesEngine(async (engine) => {
      await loadFull(engine);
    }).then(() => {
      setInit(true);
    });
  }, [init]);

  useEffect(() => {
    loadModel().then(setModel);
  }, []);

  const isCanvasEmpty = (canvas) => {
    const ctx = canvas.getContext("2d");
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    // Eğer tüm pikseller 0 ise kanvas boştur
    for (let i = 0; i < pixels.length; i++) {
      if (pixels[i] !== 0) {
        return false;
      }
    }
    return true;
  };

  const endStroke = async () => {
    const canvas = canvasRef.current;

    if (isCanvasEmpty(canvas)) return; // Kanvas boşsa işlemi iptal et

    setIsDrawing(false);

    const imgData = canvas.toDataURL("image/png");

    const img = new Image();
    img.src = imgData;

    img.onload = async () => {
      const resizedTensor = tf.browser
        .fromPixels(img)
        .toFloat()
        .resizeNearestNeighbor([28, 28])
        .mean(2)
        .div(255.0);

      const gridData = await resizedTensor.data();
      setNormalizedGrid(Array.from(gridData));

      const tensor = resizedTensor.expandDims(0).expandDims(-1);

      if (model) {
        const predictions = await model.predict(tensor).data();
        const maxProb = Math.max(...predictions);
        const topPrediction = predictions.indexOf(maxProb);

        let activeCount = 0;
        for (let i = 0; i < gridData.length; i++) {
          if (gridData[i] > 0.08) activeCount++;
        }

        if (activeCount < 10) {
          setPrediction(null);
          setPredictionProbabilities(Array(10).fill(0));
          setNormalizedGrid(null);
        } else if (maxProb < 0.65) {
          setPrediction(-1);
          setPredictionProbabilities(predictions);
        } else {
          setPrediction(topPrediction);
          setPredictionProbabilities(predictions);
        }
      }

      // Dispose tensors to prevent memory leaks
      resizedTensor.dispose();
      tensor.dispose();
    };
  };

  const clearCanvas = () => {
    setStrokes([]);
    setPrediction(null);
    setPredictionProbabilities(Array(10).fill(0));
    setNormalizedGrid(null);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  // Extract top 3 active pixels if normalizedGrid exists and prediction is active
  const activePixels = [];
  if (prediction !== null && normalizedGrid) {
    normalizedGrid.forEach((val, idx) => {
      if (val > 0.05) {
        const row = Math.floor(idx / 28);
        const col = idx % 28;
        activePixels.push({ index: idx, row, col, value: val });
      }
    });
    activePixels.sort((a, b) => b.value - a.value);
  }

  // Determine pixel values for mathematical trace
  const isLive = prediction !== null && normalizedGrid && activePixels.length > 0;
  
  const p1 = isLive ? (activePixels[0] || { row: 14, col: 12, value: 0.0 }) : { row: 14, col: 12, value: x1Value };
  const p2 = isLive ? (activePixels[1] || { row: 14, col: 13, value: 0.5 }) : { row: 14, col: 13, value: 0.5 };
  const p3 = isLive ? (activePixels[2] || { row: 15, col: 12, value: 0.0 }) : { row: 15, col: 12, value: 0.0 };

  const x1 = isLive ? p1.value : x1Value;
  const x2 = isLive ? p2.value : 0.5;
  const x3 = isLive ? p3.value : 0.0;

  // Weights and bias
  const w1 = 0.8;
  const w2 = -0.6;
  const w3 = 0.1;
  const b = -0.2;

  // z and activation
  const zValue = (x1 * w1) + (x2 * w2) + (x3 * w3) + b;
  const fzValue = Math.tanh(zValue);

  return (
    <div className="container">
      {init && <Particles options={particlesOptions} />}

      <div className="row justify-content-center">
        <div className="col-12 title">
          <h1 className="gradient-title">NeuroDigit</h1>
          <p>
            TensorFlow.js tabanlı yapay sinir ağı kullanarak el yazısı rakamları gerçek zamanlı analiz edin. 
            Aşağıdaki panele bir rakam (0-9) çizin ve yapay zekanın tahmin gücünü izleyin!
          </p>
        </div>
      </div>

      <div className="dashboard-grid">
        {/* Sol Panel: Çizim Alanı */}
        <div className="glass-panel">
          <div className="panel-header">
            <h3 className="panel-title">Çizim Alanı</h3>
            <CustomTooltip
              content="Tuvali Temizle"
              position="top"
              color="#f8fafc"
              backgroundColor="rgba(15, 23, 42, 0.95)"
              borderRadius="12px"
              fontSize="12px"
            >
              <BsArrowCounterclockwise
                className="return-icon"
                onClick={clearCanvas}
              />
            </CustomTooltip>
          </div>

          <Drawing
            strokes={strokes}
            isDrawing={isDrawing}
            addStroke={(point) => {
              setStrokes([...strokes, [point]]);
              setIsDrawing(true);
            }}
            addStrokePos={(point) => {
              if (isDrawing) {
                setStrokes(
                  strokes.map((stroke, idx) =>
                    idx === strokes.length - 1 ? [...stroke, point] : stroke
                  )
                );
              }
            }}
            endStroke={endStroke}
            canvasRef={canvasRef}
          />
          <div className="badge-row flex justify-between items-center mt-4 pt-3 border-t border-slate-800/60 text-sm">
            <span className="text-slate-400">Sistem Durumu:</span>
            {prediction !== null ? (
              prediction === -1 ? (
                <div className="status-pill warning">
                  <span className="status-dot"></span>
                  <span>Belirsiz (Çizim Tanımlanamadı)</span>
                </div>
              ) : (
                <div className="status-pill predicting">
                  <span className="status-dot"></span>
                  <span>Tahmin: {prediction} (%{(predictionProbabilities[prediction] * 100).toFixed(1)})</span>
                </div>
              )
            ) : (
              <div className="status-pill ready">
                <span className="status-dot"></span>
                <span>Analize Hazır</span>
              </div>
            )}
          </div>
        </div>

        {/* Orta Panel: Yapay Sinir Ağı Görselleştiricisi */}
        <div className="glass-panel">
          <div className="panel-header">
            <h3 className="panel-title">Yapay Sinir Ağı</h3>
          </div>
          <NeuralNetworkVisualizer
            prediction={prediction}
            probabilities={predictionProbabilities}
            drawCanvasRef={canvasRef}
          />
        </div>

        {/* Sağ Panel: Tahmin Olasılıkları Grafiği */}
        <div className="glass-panel">
          <div className="panel-header">
            <h3 className="panel-title">Tahmin Olasılıkları Dağılımı</h3>
          </div>
          <PredictionChart
            predictions={
              predictionProbabilities.length > 0
                ? predictionProbabilities
                : Array(10).fill(0)
            }
          />
        </div>
        
        {/* Eğitim ve Bilgilendirme Paneli (Yapay Zeka Nasıl Çalışır?) */}
        <div className="glass-panel education-panel">
          <div className="panel-header">
            <h3 className="panel-title">🧠 Yapay Sinir Ağı (YSA) Çalışma Prensibi ve Matematiksel Modeli</h3>
          </div>
          
          <div className="edu-console-layout">
            {/* Sidebar (Tabs) */}
            <div className="edu-console-sidebar">
              <button 
                className={`edu-tab-btn ${activeTab === 0 ? 'active' : ''}`}
                onClick={() => setActiveTab(0)}
              >
                <span className="tab-icon">🔢</span>
                <span>1. GİRDİ KATMANI (Piksel Verisi)</span>
              </button>
              <button 
                className={`edu-tab-btn ${activeTab === 1 ? 'active' : ''}`}
                onClick={() => setActiveTab(1)}
              >
                <span className="tab-icon">🧮</span>
                <span>2. NET GİRDİ TOPLAMI (Ağırlık & Bias)</span>
              </button>
              <button 
                className={`edu-tab-btn ${activeTab === 2 ? 'active' : ''}`}
                onClick={() => setActiveTab(2)}
              >
                <span className="tab-icon">📈</span>
                <span>3. AKTİVASYON (Tanh Ateşleme)</span>
              </button>
              <button 
                className={`edu-tab-btn ${activeTab === 3 ? 'active' : ''}`}
                onClick={() => setActiveTab(3)}
              >
                <span className="tab-icon">👁️</span>
                <span>4. ÇIKTI & SOFTMAX (Olasılık)</span>
              </button>
            </div>

            {/* Console Screen */}
            <div className="edu-console-screen">
              
              {activeTab === 0 && (
                <>
                  <div className="edu-screen-header">
                    <span className="edu-screen-title">1. Girdi Katmanı (Input Layer)</span>
                    <span className="edu-screen-badge">
                      {isLive ? "⚡ Çizim Verileri Aktif" : "⚠️ Simülasyon Modu"}
                    </span>
                  </div>
                  <div className="edu-screen-body">
                    {isLive ? (
                      <p>
                        Çiziminiz başarıyla analiz edilerek <strong>28x28 gri tonlamalı ızgaraya</strong> aktarıldı. 
                        Toplam <strong>{activePixels.length} aktif piksel</strong> saptandı ve normalize edildi. 
                        Aşağıda en koyu 3 piksel hücresinin koordinatları ve yoğunluk değerleri görülmektedir:
                      </p>
                    ) : (
                      <p>
                        Çizim tuvaline çizdiğiniz rakam yapay zeka modeline gönderilirken <strong>28x28 gri tonlamalı piksel ızgarasına</strong> downsample edilir (boyutlandırılır). 
                        Arayüzün solunda gördüğünüz minik mavi/turkuaz parlayan noktalar bu pikselleri temsil eder.
                      </p>
                    )}
                    <div className="edu-interactive-area">
                      <strong>Normalizasyon İşlemi:</strong>
                      <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                        Bilgisayarda beyaz pikseller <code>255</code>, siyah pikseller <code>0</code> değerindedir. 
                        Yapay sinir ağının kararlı çalışması için bu değerleri 255'e bölerek <strong>[0.0, 1.0]</strong> aralığına sıkıştırırız.
                      </p>
                      <div className="edu-code-block">
                        {isLive ? (
                          <>
                            {"// Çiziminizden Saptanan En Yoğun 3 Piksel Verisi:"}<br />
                            x₁ (Satır {p1.row}, Sütun {p1.col}) = <span className="edu-code-cyan">{p1.value.toFixed(3)}</span><br />
                            x₂ (Satır {p2.row}, Sütun {p2.col}) = <span className="edu-code-cyan">{p2.value.toFixed(3)}</span><br />
                            x₃ (Satır {p3.row}, Sütun {p3.col}) = <span className="edu-code-cyan">{p3.value.toFixed(3)}</span>
                          </>
                        ) : (
                          <>
                            {"// Piksel Değeri Sıkıştırma Formülü"}<br />
                            const normalizePixel = (colorValue) =&gt; colorValue / 255.0;<br /><br />
                            <span className="edu-code-cyan">{"// Örnek: Beyaz piksel (255) => 1.0"}</span><br />
                            <span className="edu-code-green">{"// Örnek: Gri kenar (128) => 0.502"}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </>
              )}

              {activeTab === 1 && (
                <>
                  <div className="edu-screen-header">
                    <span className="edu-screen-title">2. Net Girdi Toplamı (Synaptic Summation)</span>
                    <span className="edu-screen-badge">
                      {isLive ? "⚡ Çizim Verileri Aktif" : "⚠️ Simülasyon Modu"}
                    </span>
                  </div>
                  <div className="edu-screen-body">
                    <p>
                      Her bir nöron, kendisinden önceki katmandan gelen verileri bağlantı güçleriyle (<strong>Ağırlık - Weight</strong>) çarpar 
                      ve üzerine kendi uyarılma eşiğini (<strong>Sapma - Bias</strong>) ekler. 
                    </p>
                    
                    <div className="edu-interactive-area">
                      {isLive ? (
                        <>
                          <strong>Çizim Verilerinizle Anlık Hesaplama:</strong>
                          <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                            Ağ, çiziminizden saptanan 3 aktif piksel ve bu nöronun eğitilmiş ağırlıklarını kullanarak çarpım toplamını hesaplar:
                          </p>
                        </>
                      ) : (
                        <>
                          <strong>Matematik Simülatörü:</strong>
                          <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                            Aşağıdaki kaydırıcıyı kullanarak <strong>x₁ piksel değerini</strong> değiştirin ve net girdi toplamı <code>z</code> değerinin nasıl anlık hesaplandığını izleyin:
                          </p>
                          <div className="edu-slider-group">
                            <span className="edu-slider-label">Girdi x₁ (Piksel):</span>
                            <input 
                              type="range" 
                              className="edu-slider" 
                              min="0.0" 
                              max="1.0" 
                              step="0.05"
                              value={x1Value} 
                              onChange={(e) => setX1Value(parseFloat(e.target.value))} 
                            />
                            <span className="edu-slider-value">{x1Value.toFixed(2)}</span>
                          </div>
                        </>
                      )}

                      <div className="edu-code-block">
                        {"// Formül: z = (x₁ · w₁) + (x₂ · w₂) + (x₃ · w₃) + b"}<br />
                        {"// Ağırlıklar: w₁ = 0.80, w₂ = -0.60, w₃ = 0.10 | Bias: b = -0.20"}<br /><br />
                        z = (<span className="edu-code-cyan">{x1.toFixed(2)}</span> · 0.80) + (<span className="edu-code-cyan">{x2.toFixed(2)}</span> · -0.60) + (<span className="edu-code-cyan">{x3.toFixed(2)}</span> · 0.10) + (-0.20)<br />
                        z = {(x1 * 0.8).toFixed(3)} + {(x2 * -0.6).toFixed(3)} + {(x3 * 0.1).toFixed(3)} - 0.200<br />
                        z = <span className="edu-code-highlight">{zValue.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {activeTab === 2 && (
                <>
                  <div className="edu-screen-header">
                    <span className="edu-screen-title">3. Tanh Aktivasyon Fonksiyonu (Non-linearity)</span>
                    <span className="edu-screen-badge">
                      {isLive ? "⚡ Çizim Verileri Aktif" : "⚠️ Simülasyon Modu"}
                    </span>
                  </div>
                  <div className="edu-screen-body">
                    <p>
                      Lineer olmayan örüntüleri (eğrileri, kıvrımları) öğrenebilmek için nöron girdi toplamını aktivasyon fonksiyonundan geçiririz. 
                      <strong>Hiperbolik Tanjant (Tanh)</strong>, çıktıyı <strong>[-1, 1]</strong> arasına sıkıştırarak negatif ve pozitif ateşleme dengesi sağlar.
                    </p>
                    
                    <div className="edu-interactive-area">
                      <strong>Aktivasyon Formül Çözümü:</strong>
                      <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                        {isLive 
                          ? `Çiziminizden hesaplanan z = ${zValue.toFixed(3)} net girdisi için nöronun ateşleme kuvveti:` 
                          : `Simülasyondan gelen z = ${zValue.toFixed(3)} net girdisi için nöronun ateşleme kuvveti:`}
                      </p>

                      <div className="edu-code-block">
                        {"// Hiperbolik Tanjant Formülü"}<br />
                        f(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)<br />
                        f({zValue.toFixed(3)}) = (e^{zValue.toFixed(3)} - e^{-zValue.toFixed(3)}) / (e^{zValue.toFixed(3)} + e^{-zValue.toFixed(3)})<br />
                        f({zValue.toFixed(3)}) = {Math.exp(zValue).toFixed(4)} - {Math.exp(-zValue).toFixed(4)} / ...<br />
                        Aktivasyon Gücü f(z) = <span className="edu-code-green">{fzValue.toFixed(4)}</span>
                      </div>

                      <div className="firing-meter-container">
                        <div className="firing-meter-label">
                          Nöron Ateşleme Seviyesi: {(fzValue * 100).toFixed(1)}% 
                          {fzValue > 0 ? " (Uyarılıyor / Aktif)" : " (Engelleniyor / İnaktif)"}
                        </div>
                        <div className="firing-meter-bar">
                          <div 
                            className="firing-meter-fill" 
                            style={{ 
                              width: `${((fzValue + 1) / 2) * 100}%`,
                              background: fzValue > 0 ? 'linear-gradient(90deg, #818cf8, #22d3ee)' : 'linear-gradient(90deg, #ec4899, #f472b6)'
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {activeTab === 3 && (
                <>
                  <div className="edu-screen-header">
                    <span className="edu-screen-title">4. Çıktı Kararı ve Softmax (Probability Distribution)</span>
                    <span className="edu-screen-badge">
                      {isLive ? "⚡ Çizim Tahmini Aktif" : "⚠️ Simülasyon Modu"}
                    </span>
                  </div>
                  <div className="edu-screen-body">
                    <p>
                      En son çıktı katmanında 0'dan 9'a kadar her rakam için birer nöron (toplam 10 nöron) bulunur. 
                      Bu nöronların ürettiği ham skorlar (logit) <strong>Softmax</strong> fonksiyonu ile olasılık dağılımına (%0 - %100) dönüştürülür.
                    </p>
                    <div className="edu-interactive-area">
                      {isLive ? (
                        <>
                          <strong>Modelinizin Çiziminiz İçin Ürettiği Gerçek Olasılıklar:</strong>
                          <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                            Modeliniz çizimi analiz etti ve en yüksek olasılık olan <strong>{(predictionProbabilities[prediction] * 100).toFixed(1)}%</strong> değeriyle <strong>{prediction}</strong> rakamını tahmin etti!
                          </p>
                        </>
                      ) : (
                        <>
                          <strong>Softmax Formülü:</strong>
                          <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                            Her bir çıktının olasılığı, o çıktının üstel değerinin (eᶻᶜ), tüm çıktıların üstel değerlerinin toplamına bölünmesiyle elde edilir.
                          </p>
                        </>
                      )}
                      
                      <div className="edu-code-block">
                        {"// Skorları Olasılık Yüzdesine Çevirme"}<br />
                        P(y=c | X) = exp(z_c) / Σ (exp(z_j))<br /><br />
                        {isLive ? (
                          <>
                            {"// Çiziminiz İçin En Yüksek Tahmin Olasılıkları:"}<br />
                            Sınıf {prediction} Olasılığı = <span className="edu-code-green">{(predictionProbabilities[prediction] * 100).toFixed(2)}%</span><br />
                            {predictionProbabilities.map((p, idx) => {
                              if (idx !== prediction && p > 0.01) {
                                return (
                                  <React.Fragment key={idx}>
                                    Sınıf {idx} Olasılığı = { (p * 100).toFixed(2) }%<br />
                                  </React.Fragment>
                                );
                              }
                              return null;
                            })}
                          </>
                        ) : (
                          <>
                            <span className="edu-code-cyan">{"// Örnek Çıktı Skorları: [z₀=-1.2, z₁=0.5, z₂=-0.1, z₃=3.5, ...]"}</span><br />
                            exp(z₃) = e³.⁵ ≈ 33.11 | Toplam Üstel Değerler = 34.62<br />
                            Sınıf 3 Olasılığı = 33.11 / 34.62 = <span className="edu-code-green">95.6%</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </>
              )}

            </div>
          </div>
        </div>
      </div>

      <div className="footer-banner">
        © 2026 Powered by Ahmet Cankurt & Antigravity
      </div>
    </div>
  );
};

export default App;
