import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const PredictionChart = ({ predictions }) => {
  // Eğer tahminler yoksa varsayılan bir veri kümesi göster
  const defaultPredictions = Array(10).fill(0); // 10 sıfırlı bir dizi
  
  const data = {
    labels: [...Array(10).keys()], // 0-9 arası rakamlar
    datasets: [
      {
        label: "Tahmin Olasılığı",
        data: predictions || defaultPredictions,
        backgroundColor: (context) => {
          const chart = context.chart;
          const { ctx, chartArea } = chart;
          if (!chartArea) {
            return "rgba(56, 189, 248, 0.5)"; // fallback
          }
          const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
          gradient.addColorStop(0, "rgba(99, 102, 241, 0.2)"); // Indigo transparan
          gradient.addColorStop(1, "rgba(56, 189, 248, 0.8)"); // Cyan parlama
          return gradient;
        },
        borderColor: "#38bdf8",
        borderWidth: 1.5,
        borderRadius: 8,
        borderSkipped: false,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false, // Gereksiz lejantı gizle
      },
      tooltip: {
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        titleColor: "#f1f5f9",
        bodyColor: "#38bdf8",
        borderColor: "rgba(255, 255, 255, 0.08)",
        borderWidth: 1,
        titleFont: {
          family: "'Plus Jakarta Sans', sans-serif",
          weight: "600"
        },
        bodyFont: {
          family: "'Plus Jakarta Sans', sans-serif"
        },
        callbacks: {
          label: (context) => ` Olasılık: ${(context.raw * 100).toFixed(2)}%`
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: "rgba(255, 255, 255, 0.05)",
          drawBorder: false,
        },
        ticks: {
          color: "rgba(148, 163, 184, 0.8)",
          font: {
            family: "'Plus Jakarta Sans', sans-serif",
            size: 12,
            weight: "600"
          }
        }
      },
      y: {
        beginAtZero: true,
        max: 1.0,
        grid: {
          color: "rgba(255, 255, 255, 0.05)",
          drawBorder: false,
        },
        ticks: {
          color: "rgba(148, 163, 184, 0.8)",
          font: {
            family: "'Plus Jakarta Sans', sans-serif",
            size: 11,
          },
          callback: (value) => `${(value * 100).toFixed(0)}%`
        }
      }
    }
  };

  return (
    <div className="chart-container">
      <Bar data={data} options={options} />
    </div>
  );
};

export default PredictionChart;
