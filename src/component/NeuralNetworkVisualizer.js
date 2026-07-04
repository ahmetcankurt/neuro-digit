import React, { useEffect, useRef } from "react";

// Helper to generate deterministic weights based on seed
const seededRandom = (seed) => {
  const x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
};

// Initialize weights once (784 inputs -> 8 hidden -> 8 hidden -> 10 outputs)
const initWeights = (state) => {
  // Defensive check: if weights are already initialized to the correct size, return early
  if (state.W1 && state.W1.length === 784 && state.initialized) return;

  let seed = 1234.5;
  // Generate 784 x 8 weights
  state.W1 = Array(784).fill(0).map(() => 
    Array(8).fill(0).map(() => {
      const val = seededRandom(seed++) * 3.0 - 1.5; // [-1.5, 1.5]
      return val;
    })
  );

  // Generate 8 x 8 weights
  state.W2 = Array(8).fill(0).map(() => 
    Array(8).fill(0).map(() => {
      const val = seededRandom(seed++) * 3.0 - 1.5;
      return val;
    })
  );

  // Generate 8 x 10 weights
  state.W3 = Array(8).fill(0).map(() => 
    Array(10).fill(0).map(() => {
      const val = seededRandom(seed++) * 3.0 - 1.5;
      return val;
    })
  );

  state.initialized = true;
};

const NeuralNetworkVisualizer = ({ prediction, probabilities, drawCanvasRef }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const stateRef = useRef({
    W1: [],
    W2: [],
    W3: [],
    inputGrid: Array(784).fill(0),
    h1Activations: Array(8).fill(0),
    h2Activations: Array(8).fill(0),
    prevPrediction: null,
    waveStage: -1, // -1: idle, 0: L0->L1, 1: L1->L2, 2: L2->L3, 3: completed
    waveProgress: 0,
    pulses: [],
    initialized: false,
  });

  useEffect(() => {
    initWeights(stateRef.current);

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resizeCanvas = () => {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = 380; // Increased height to 380px for spacious distribution
    };
    resizeCanvas();

    const getNodesPositions = () => {
      const nodes = [];
      const w = canvas.width;
      const h = canvas.height;

      // Layer 0: 28x28 Grid of nodes representing the exact input image resolution
      const l0Nodes = [];
      const gridWidth = 84;
      const l1X = w * 0.38;
      const gridStartX = (l1X / 2) - (gridWidth / 2); // Center horizontally in column 0
      const cellSpacing = gridWidth / 27; // 28 nodes spacing
      const gridStartY = (h - gridWidth) / 2;

      for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
          l0Nodes.push({
            id: `L0_N${r}_${c}`,
            x: gridStartX + c * cellSpacing,
            y: gridStartY + r * cellSpacing,
            layer: 0,
            index: r * 28 + c,
            activation: 0,
          });
        }
      }
      nodes.push(l0Nodes);

      // Layer 1: Hidden 1 (8 nodes)
      const l1Nodes = [];
      const l1Spacing = (h - 80) / 7; // Distribute nodes vertically with large spacing
      for (let i = 0; i < 8; i++) {
        l1Nodes.push({
          id: `L1_N${i}`,
          x: l1X,
          y: 25 + i * l1Spacing,
          layer: 1,
          index: i,
          activation: 0,
        });
      }
      nodes.push(l1Nodes);

      // Layer 2: Hidden 2 (8 nodes)
      const l2Nodes = [];
      const l2X = w * 0.65;
      const l2Spacing = (h - 80) / 7;
      for (let i = 0; i < 8; i++) {
        l2Nodes.push({
          id: `L2_N${i}`,
          x: l2X,
          y: 25 + i * l2Spacing,
          layer: 2,
          index: i,
          activation: 0,
        });
      }
      nodes.push(l2Nodes);

      // Layer 3: Output (10 nodes representing digits 0-9)
      const l3Nodes = [];
      const l3X = w - 60;
      const l3Spacing = (h - 90) / 9; // Distribute output nodes with ample vertical space
      for (let i = 0; i < 10; i++) {
        l3Nodes.push({
          id: `L3_N${i}`,
          x: l3X,
          y: 25 + i * l3Spacing,
          layer: 3,
          index: i,
          activation: 0,
        });
      }
      nodes.push(l3Nodes);

      return nodes;
    };

    let nodes = getNodesPositions();

    const handleResize = () => {
      resizeCanvas();
      nodes = getNodesPositions();
    };
    window.addEventListener("resize", handleResize);

    // Pixel Downsampler: Canvas -> 28x28 Grayscale Matrix
    const downsampleDrawing28x28 = () => {
      const drawCanvas = drawCanvasRef?.current;
      if (!drawCanvas) return Array(784).fill(0);
      
      const drawCtx = drawCanvas.getContext("2d");
      const w = drawCanvas.width;
      const h = drawCanvas.height;
      const imgData = drawCtx.getImageData(0, 0, w, h).data;
      
      const grid = Array(784).fill(0);
      const cellW = w / 28;
      const cellH = h / 28;
      
      for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
          const startX = Math.floor(c * cellW);
          const startY = Math.floor(r * cellH);
          const endX = Math.floor((c + 1) * cellW);
          const endY = Math.floor((r + 1) * cellH);
          
          let totalAlpha = 0;
          let samples = 0;
          
          for (let y = startY; y < endY; y += 3) { // Sample every 3rd pixel to optimize
            for (let x = startX; x < endX; x += 3) {
              const index = (y * w + x) * 4;
              if (imgData[index] > 30) {
                totalAlpha += imgData[index];
              }
              samples++;
            }
          }
          const avg = totalAlpha / (samples * 255);
          grid[r * 28 + c] = Math.min(avg * 4.0, 1.0); // Boost contrast
        }
      }
      return grid;
    };

    // Animation frames
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const state = stateRef.current;

      // Draw Sci-Fi Tech Grid Background
      ctx.strokeStyle = "rgba(99, 102, 241, 0.025)";
      ctx.lineWidth = 0.5;
      const gridSize = 25;
      for (let x = 0; x < canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // Draw Vertical Column Guidelines
      ctx.strokeStyle = "rgba(99, 102, 241, 0.05)";
      ctx.lineWidth = 0.8;
      ctx.setLineDash([5, 5]);
      const l1X_guide = canvas.width * 0.38;
      const l2X_guide = canvas.width * 0.65;
      const l3X_guide = canvas.width - 60;
      const colsX = [l1X_guide / 2, l1X_guide, l2X_guide, l3X_guide];

      colsX.forEach((colX) => {
        ctx.beginPath();
        ctx.moveTo(colX, 15);
        ctx.lineTo(colX, canvas.height - 25);
        ctx.stroke();
      });
      ctx.setLineDash([]); // Reset line dash

      // 1. Downsample the drawing canvas in real-time
      const inputGrid = downsampleDrawing28x28();
      state.inputGrid = inputGrid;

      // 2. Perform feedforward math inside the visualizer with safety checks
      // Hidden Layer 1 activations (784 -> 8)
      const h1 = Array(8).fill(0);
      for (let j = 0; j < 8; j++) {
        let sum = 0;
        for (let i = 0; i < 784; i++) {
          const weight = (state.W1[i] && state.W1[i][j] !== undefined) ? state.W1[i][j] : 0;
          sum += inputGrid[i] * weight;
        }
        h1[j] = Math.tanh(sum); // range [-1, 1]
      }
      state.h1Activations = h1;

      // Hidden Layer 2 activations (8 -> 8)
      const h2 = Array(8).fill(0);
      for (let k = 0; k < 8; k++) {
        let sum = 0;
        for (let j = 0; j < 8; j++) {
          const weight = (state.W2[j] && state.W2[j][k] !== undefined) ? state.W2[j][k] : 0;
          sum += h1[j] * weight;
        }
        h2[k] = Math.tanh(sum);
      }
      state.h2Activations = h2;

      // Apply activations to visual nodes
      // Layer 0 (Input)
      if (nodes[0]) {
        nodes[0].forEach((node, idx) => {
          node.activation = inputGrid[idx] || 0;
        });
      }

      // Layer 1 (Hidden 1)
      if (nodes[1]) {
        nodes[1].forEach((node, idx) => {
          node.activation = Math.abs(h1[idx]) || 0;
        });
      }

      // Layer 2 (Hidden 2)
      if (nodes[2]) {
        nodes[2].forEach((node, idx) => {
          node.activation = Math.abs(h2[idx]) || 0;
        });
      }

      // Layer 3 (Output)
      if (nodes[3]) {
        nodes[3].forEach((node, idx) => {
          const prob = (probabilities && probabilities[idx]) || 0;
          node.activation = prob;
        });
      }

      // Update feedforward wave
      if (state.waveStage >= 0 && state.waveStage <= 2) {
        state.waveProgress += 0.08; // Wave propagation speed
        if (state.waveProgress >= 1.0) {
          state.waveStage += 1;
          state.waveProgress = 0;
        }
      }

      // Track active pathway connections to highlight
      const activeConnections = [];
      if (prediction !== null && prediction !== undefined && prediction >= 0) {
        const predIdx = prediction;
        
        // 1. Output -> Hidden 2 connections
        const h2Weights = [];
        for (let j = 0; j < 8; j++) {
          const weight = (state.W3[j] && state.W3[j][predIdx] !== undefined) ? state.W3[j][predIdx] : 0;
          const strength = Math.abs(h2[j] * weight);
          h2Weights.push({ index: j, strength });
        }
        h2Weights.sort((a, b) => b.strength - a.strength);
        const topH2Indices = h2Weights.slice(0, 3).map((w) => w.index);
        
        topH2Indices.forEach((h2Idx) => {
          const weight = (state.W3[h2Idx] && state.W3[h2Idx][predIdx] !== undefined) ? state.W3[h2Idx][predIdx] : 0;
          activeConnections.push({
            layer: 2,
            from: h2Idx,
            to: predIdx,
            weight: weight,
            strength: h2[h2Idx] || 0,
          });

          // 2. Hidden 2 -> Hidden 1 connections
          const h1Weights = [];
          for (let i = 0; i < 8; i++) {
            const weight = (state.W2[i] && state.W2[i][h2Idx] !== undefined) ? state.W2[i][h2Idx] : 0;
            const strength = Math.abs(h1[i] * weight);
            h1Weights.push({ index: i, strength });
          }
          h1Weights.sort((a, b) => b.strength - a.strength);
          const topH1Indices = h1Weights.slice(0, 2).map((w) => w.index);

          topH1Indices.forEach((h1Idx) => {
            const weight = (state.W2[h1Idx] && state.W2[h1Idx][h2Idx] !== undefined) ? state.W2[h1Idx][h2Idx] : 0;
            activeConnections.push({
              layer: 1,
              from: h1Idx,
              to: h2Idx,
              weight: weight,
              strength: h1[h1Idx] || 0,
            });

            // 3. Hidden 1 -> Input Grid connections
            for (let i = 0; i < 784; i++) {
              if (inputGrid[i] > 0.15) { // Connect to active input pixels
                const weight = (state.W1[i] && state.W1[i][h1Idx] !== undefined) ? state.W1[i][h1Idx] : 0;
                activeConnections.push({
                  layer: 0,
                  from: i,
                  to: h1Idx,
                  weight: weight,
                  strength: inputGrid[i],
                });
              }
            }
          });
        });
      }

      // Draw all connection lines (edges)
      for (let l = 0; l < 3; l++) {
        const currentLayer = nodes[l];
        const nextLayer = nodes[l + 1];
        if (!currentLayer || !nextLayer) continue;

        currentLayer.forEach((fromNode) => {
          // Optimization: Skip drawing inactive lines for Layer 0 (784 input pixels)
          if (l === 0) return;

          nextLayer.forEach((toNode) => {
            const isActive = activeConnections.some(
              (conn) => conn.layer === l && conn.from === fromNode.index && conn.to === toNode.index
            );

            // Draw inactive lines very faintly
            if (!isActive) {
              ctx.beginPath();
              ctx.moveTo(fromNode.x, fromNode.y);
              ctx.lineTo(toNode.x, toNode.y);
              ctx.strokeStyle = "rgba(255, 255, 255, 0.02)";
              ctx.lineWidth = 0.5;
              ctx.stroke();
            }
          });
        });
      }

      // Draw active pathway connection lines with glowing colors and weights
      const topWeightsToLabel = [];
      activeConnections.forEach((conn) => {
        if (!nodes[conn.layer] || !nodes[conn.layer + 1]) return;
        const fromNode = nodes[conn.layer][conn.from];
        const toNode = nodes[conn.layer + 1][conn.to];
        if (!fromNode || !toNode) return;

        const isPositive = conn.weight >= 0;
        const color1 = isPositive ? "rgba(34, 211, 238, 0.6)" : "rgba(236, 72, 153, 0.6)"; // Cyan vs Pink
        const color2 = isPositive ? "rgba(34, 211, 238, 0.15)" : "rgba(236, 72, 153, 0.15)";
        
        const gradient = ctx.createLinearGradient(fromNode.x, fromNode.y, toNode.x, toNode.y);
        
        if (state.waveStage === conn.layer) {
          const t = state.waveProgress;
          gradient.addColorStop(0, color2);
          gradient.addColorStop(Math.max(0, t - 0.1), color2);
          gradient.addColorStop(t, color1);
          gradient.addColorStop(Math.min(1.0, t + 0.1), color2);
          gradient.addColorStop(1.0, color2);
        } else if (state.waveStage > conn.layer || state.waveStage === 3) {
          gradient.addColorStop(0, color1);
          gradient.addColorStop(1, color2);
        } else {
          gradient.addColorStop(0, isPositive ? "rgba(34, 211, 238, 0.06)" : "rgba(236, 72, 153, 0.06)");
          gradient.addColorStop(1, "rgba(255, 255, 255, 0.01)");
        }

        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.strokeStyle = gradient;
        // Layer 0 active lines are thinner since there are more of them
        ctx.lineWidth = conn.layer === 0 ? 1.0 : 1.8;
        ctx.stroke();

        if (state.waveStage === 3 || state.waveStage > conn.layer) {
          topWeightsToLabel.push({
            x: (fromNode.x + toNode.x) / 2,
            y: (fromNode.y + toNode.y) / 2,
            weight: conn.weight,
            importance: Math.abs(conn.weight) * conn.strength,
            isPositive,
          });
        }
      });

      // Render weight labels for top connections
      topWeightsToLabel.sort((a, b) => b.importance - a.importance);
      topWeightsToLabel.slice(0, 4).forEach((label) => {
        ctx.font = "bold 8px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.fillRect(label.x - 18, label.y - 6, 36, 12);
        ctx.fillStyle = label.isPositive ? "#22d3ee" : "#f472b6";
        ctx.fillText(`${label.isPositive ? "+" : ""}${label.weight.toFixed(2)}`, label.x, label.y);
      });

      // Draw wave pulses (during feedforward)
      if (state.waveStage >= 0 && state.waveStage <= 2) {
        const l = state.waveStage;
        const t = state.waveProgress;
        
        activeConnections
          .filter((conn) => conn.layer === l)
          .forEach((conn) => {
            if (!nodes[l] || !nodes[l + 1]) return;
            const fromNode = nodes[l][conn.from];
            const toNode = nodes[l + 1][conn.to];
            if (!fromNode || !toNode) return;

            const px = fromNode.x + (toNode.x - fromNode.x) * t;
            const py = fromNode.y + (toNode.y - fromNode.y) * t;

            ctx.beginPath();
            ctx.arc(px, py, l === 0 ? 2.5 : 3.5, 0, Math.PI * 2);
            ctx.fillStyle = conn.weight >= 0 ? "#22d3ee" : "#ec4899";
            ctx.shadowBlur = 6;
            ctx.shadowColor = conn.weight >= 0 ? "#22d3ee" : "#ec4899";
            ctx.fill();
            ctx.shadowBlur = 0;
          });
      }

      // Draw nodes
      nodes.forEach((layer, layerIdx) => {
        layer.forEach((node) => {
          const isL0 = layerIdx === 0;
          const isOutput = layerIdx === 3;
          
          let radius = isOutput ? 12 : (isL0 ? 1.0 : 6);
          let nodeColor = "rgba(71, 85, 105, 0.35)";
          let outerGlow = "transparent";
          let scale = 1.0;

          if (isL0) {
            // 28x28 nodes drawing
            if (node.activation > 0.05) {
              nodeColor = `rgba(56, 189, 248, ${node.activation * 0.95})`; // Glowing Cyan
              outerGlow = "rgba(56, 189, 248, 0.5)";
              radius = 1.5;
            } else {
              nodeColor = "rgba(255, 255, 255, 0.06)";
            }
          } else if (isOutput) {
            const prob = node.activation;
            if (prediction !== null && node.index === prediction) {
              const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, 14);
              grad.addColorStop(0, "#ffffff");
              grad.addColorStop(0.3, "#c084fc");
              grad.addColorStop(1, "#7e22ce");
              nodeColor = grad;
              outerGlow = "rgba(168, 85, 247, 0.85)";
              radius = 14;
              if (state.waveStage === 3) {
                const pulseRadius = radius + Math.sin(Date.now() * 0.01) * 3;
                ctx.beginPath();
                ctx.arc(node.x, node.y, pulseRadius, 0, Math.PI * 2);
                ctx.strokeStyle = "rgba(168, 85, 247, 0.3)";
                ctx.lineWidth = 1.5;
                ctx.stroke();
              }
            } else if (prob > 0.15) {
              const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, 12 + prob * 2);
              grad.addColorStop(0, "rgba(255, 255, 255, 0.9)");
              grad.addColorStop(0.3, "rgba(56, 189, 248, 0.8)");
              grad.addColorStop(1, "rgba(15, 23, 42, 0.75)");
              nodeColor = grad;
              outerGlow = "rgba(56, 189, 248, 0.5)";
              radius = 12 + prob * 2;
            } else {
              nodeColor = "rgba(15, 23, 42, 0.65)";
              radius = 11;
            }

            // Circular progress gauge (Softmax probability dials)
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius + 4, 0, Math.PI * 2);
            ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
            ctx.lineWidth = 1.0;
            ctx.stroke();

            if (prob > 0.01) {
              ctx.beginPath();
              ctx.arc(node.x, node.y, radius + 4, -Math.PI / 2, -Math.PI / 2 + (prob * Math.PI * 2));
              ctx.strokeStyle = prediction !== null && node.index === prediction 
                ? "rgba(168, 85, 247, 0.95)" 
                : "rgba(34, 211, 238, 0.75)";
              ctx.lineWidth = 2.0;
              ctx.stroke();
            }
          } else {
            // Hidden Layer Nodes
            if (node.activation > 0.05) {
              const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, radius * scale);
              grad.addColorStop(0, "rgba(255, 255, 255, 0.9)");
              grad.addColorStop(0.3, "rgba(34, 211, 238, 0.85)");
              grad.addColorStop(1, "rgba(56, 189, 248, 0.15)");
              nodeColor = grad;
              outerGlow = "rgba(56, 189, 248, 0.6)";
              scale = 1.0 + node.activation * 0.15;
            } else {
              nodeColor = "rgba(30, 41, 59, 0.65)";
            }

            // Rotating active neon arcs
            if (node.activation > 0.05) {
              const startAngle = (Date.now() * 0.003) % (Math.PI * 2);
              ctx.beginPath();
              ctx.arc(node.x, node.y, radius * scale + 3, startAngle, startAngle + Math.PI * 0.7);
              ctx.strokeStyle = "rgba(34, 211, 238, 0.7)";
              ctx.lineWidth = 1.2;
              ctx.stroke();
            } else {
              ctx.beginPath();
              ctx.arc(node.x, node.y, radius + 2, 0, Math.PI * 2);
              ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
              ctx.lineWidth = 0.8;
              ctx.stroke();
            }
          }

          // Draw Outer Ring for active nodes
          if (outerGlow !== "transparent" && !isL0) {
            ctx.shadowBlur = isOutput ? 12 : 6;
            ctx.shadowColor = outerGlow;
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius * scale + 2, 0, Math.PI * 2);
            ctx.strokeStyle = isOutput ? outerGlow : "rgba(56, 189, 248, 0.2)";
            ctx.lineWidth = 1.2;
            ctx.stroke();
          }

          // Draw Inner Circle
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius * scale, 0, Math.PI * 2);
          ctx.fillStyle = nodeColor;
          ctx.fill();
          
          if (isOutput) {
            ctx.strokeStyle = prediction !== null && node.index === prediction 
              ? "#ffffff" 
              : "rgba(255, 255, 255, 0.12)";
            ctx.lineWidth = prediction !== null && node.index === prediction ? 2 : 1;
            ctx.stroke();
          }
          ctx.shadowBlur = 0;

          // Output labels
          if (isOutput) {
            ctx.font = "bold 11px 'Plus Jakarta Sans', sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = prediction !== null && node.index === prediction 
              ? "#ffffff" 
              : "rgba(255, 255, 255, 0.7)";
            ctx.fillText(node.index.toString(), node.x, node.y);
          }
        });
      });

      // Draw static layer titles below
      const w = canvas.width;
      const l1X = w * 0.38;
      const l2X = w * 0.65;
      const l3X = w - 60;
      ctx.font = "bold 9px 'Plus Jakarta Sans', sans-serif";
      ctx.textAlign = "center";
      
      const labels = [
        { text: "28x28 GİRDİ", x: l1X / 2 },
        { text: "GİZLİ 1", x: l1X },
        { text: "GİZLİ 2", x: l2X },
        { text: "ÇIKTI", x: l3X }
      ];

      labels.forEach((label) => {
        ctx.fillStyle = "rgba(99, 102, 241, 0.45)";
        ctx.fillText(`[ ${label.text} ]`, label.x, canvas.height - 10);
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationRef.current);
    };
  }, [prediction, probabilities, drawCanvasRef]);

  // Trigger wave when prediction changes
  useEffect(() => {
    const state = stateRef.current;
    if (prediction !== state.prevPrediction) {
      if (prediction !== null && prediction !== undefined) {
        state.waveStage = 0;
        state.waveProgress = 0;
      } else {
        state.waveStage = -1;
        state.waveProgress = 0;
      }
      state.prevPrediction = prediction;
    }
  }, [prediction]);

  return (
    <div className="w-100 h-100 d-flex align-items-center justify-content-center">
      <canvas
        ref={canvasRef}
        className="w-100"
        style={{
          display: "block",
          maxHeight: "380px",
          background: "transparent",
        }}
      />
    </div>
  );
};

export default NeuralNetworkVisualizer;
